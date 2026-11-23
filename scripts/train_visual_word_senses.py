"""Train the matched visual spellers on a single GPU and archive complete fits."""

import argparse
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from torch.nn import functional as F
from ilm.visual_lm.idiom_ink import choose_landmarks
from ilm.visual_lm.visual_word_sense import VisualGlossWriter, retinal_features, WIDTH
from scripts.prepare_visual_word_senses import WORK, PUBLIC, SEED, save, sha
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_span_choice import gpu_owner
import json

EPOCHS = 8
BATCH = 256


def corpus(split):
    reg = json.loads((PUBLIC / "split.json").read_text())
    file = WORK / (split + ".npz")
    if (
        sha(file) != reg["data"][split]["inputs"]
        or sha(WORK / "pixels.npy") != reg["pixels"]
    ):
        raise ValueError("Image data binding changed")
    with np.load(file) as d:
        return d["x"], d["y"]


def images():
    return torch.from_numpy(np.unpackbits(np.load(WORK / "pixels.npy"), axis=-1)).cuda()


def new_model(pixels, y):
    train_cells = np.unique(y[y != 0])
    raw = pixels[0, train_cells].cpu().numpy()
    points = choose_landmarks(raw)
    k = next(
        i
        for i in range(1, len(points) + 1)
        if len(np.unique(raw[:, points[:i]], axis=0)) == len(train_cells)
    )
    assert k <= 14
    return VisualGlossWriter(points[:k]).cuda(), torch.from_numpy(raw).cuda()


def motor():
    _, y = corpus("train")
    pix = images()
    torch.manual_seed(SEED)
    model, physical = new_model(pix, y)
    shape = physical[:, model.landmarks].float()
    opt = torch.optim.AdamW(model.motor.parameters(), lr=0.007)
    start = monotonic()
    energy = snapshot()
    for step in range(1, 1501):
        logits = model.motor(shape)
        losses = F.binary_cross_entropy_with_logits(
            logits, physical.float(), reduction="none"
        )
        loss = losses.mean() + 0.5 * losses.topk(16, dim=1).values.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 and torch.equal(model.motor(shape) > 0, physical.bool()):
            break
    assert torch.equal(model.motor(shape) > 0, physical.bool())
    dest = WORK / "initial.safetensors"
    if dest.exists():
        raise FileExistsError(dest)
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(dest)
    )
    save(
        PUBLIC / "motor.json",
        dict(
            initial=sha(dest),
            steps=step,
            exact=len(physical),
            landmarks=model.landmarks.tolist(),
            seconds=monotonic() - start,
            energy=consumed(energy, snapshot()),
        ),
    )


def train(arm):
    own = WORK / arm
    own.mkdir(exist_ok=False)
    x, y = corpus("train")
    pix = images()
    feat = retinal_features(pix)
    proof = json.loads((PUBLIC / "motor.json").read_text())
    initial = WORK / "initial.safetensors"
    assert sha(initial) == proof["initial"]
    torch.manual_seed(SEED)
    model = VisualGlossWriter(proof["landmarks"]).cuda()
    model.load_state_dict(load_file(str(initial)))
    model.motor.requires_grad_(False)
    # The auxiliary training labels are equivalence classes of entire target
    # images. The released recurrent speller does not load this head.
    physical = pix[0, :, model.landmarks].long()
    code = (physical * (2 ** torch.arange(physical.shape[1], device="cuda"))).sum(1)
    code[0] = model.stop
    y_codes = code[torch.from_numpy(y).cuda().long()]
    unique_gloss, sense = np.unique(y, axis=0, return_inverse=True)
    aux = torch.nn.Linear(WIDTH, len(unique_gloss)).cuda()
    opt = torch.optim.AdamW(
        [*filter(lambda p: p.requires_grad, model.parameters()), *aux.parameters()],
        lr=0.001,
        weight_decay=0.005,
    )
    xg = torch.from_numpy(x).cuda().long()
    labels = torch.from_numpy(sense).cuda().long()
    length = torch.from_numpy((y != 0).sum(1) + 1).cuda()
    times = torch.arange(y.shape[1], device="cuda")[None, :]
    # Equal head images must share a permutation group even across encodings.
    # The preparation already merged identical printed examples; regroup the
    # remaining source positions by the actual central bitmap.
    gids = np.unique(pix[0].cpu().numpy(), axis=0, return_inverse=True)[1][
        x[:, x.shape[1] // 2]
    ]
    groups = {g: ix for g, ix in zip(*_group_indices(gids))}
    step = 0
    trace = []
    start = monotonic()
    energy = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(len(x))
        mapping = np.arange(len(x))
        if arm == "shuffled":
            shuffled = np.random.default_rng(SEED * 3 + epoch)
            for ix in groups.values():
                mapping[ix] = shuffled.permutation(ix)
        running = 0.0
        for j in range(0, len(order), BATCH):
            ids = order[j : j + BATCH]
            targets = torch.from_numpy(mapping[ids]).cuda()
            face = step % 2
            xf = feat[face, xg[ids]]
            yc = y_codes[targets]
            valid = times < length[targets, None]
            ctx = model.encode(xf)
            logits = model.teacher(ctx, yc)
            loss = F.cross_entropy(
                logits.transpose(1, 2), yc.masked_fill(~valid, -100)
            ) + 0.5 * F.cross_entropy(aux(ctx), labels[targets])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *aux.parameters()], 2.0
            )
            opt.step()
            running += float(loss.detach()) * len(ids)
            step += 1
            if step % 256 == 0 and (
                monotonic() - start > 1500
                or torch.cuda.max_memory_allocated() > 6 * 2**30
            ):
                raise RuntimeError("Resource ceiling")
        trace.append(running / len(x))
        print(arm, epoch + 1, trace[-1], flush=True)
    dest = own / "model.safetensors"
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(dest)
    )
    save(
        PUBLIC / (arm + "_fit.json"),
        dict(
            weights=sha(dest),
            initial=sha(initial),
            training_examples=len(x),
            presentations=len(x) * EPOCHS,
            steps=step,
            parameters=sum(p.numel() for p in model.parameters()),
            auxiliary=sum(p.numel() for p in aux.parameters()),
            seconds=monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            energy=consumed(energy, snapshot()),
            loss=trace,
        ),
    )


def _group_indices(ids):
    order = np.argsort(ids, kind="stable")
    starts = np.flatnonzero(np.diff(ids[order])) + 1
    return np.unique(ids), np.split(order, starts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("motor", "aligned", "shuffled"))
    args = ap.parse_args()
    with gpu_owner():
        torch.set_float32_matmul_precision("high")
        if args.arm == "motor":
            motor()
        else:
            train(args.arm)
