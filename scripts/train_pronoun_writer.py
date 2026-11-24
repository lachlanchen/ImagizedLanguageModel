"""Fit the small visual context residual through the frozen English image speller."""

from collections import defaultdict
from time import monotonic
import argparse

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.gloss_residual import GlossResidual
from ilm.visual_lm.gloss_focus import cell_features
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_pronoun_writer import SEED, WORK, PUBLIC
from scripts.prepare_asked_gloss import dump, read, sha
from scripts.train_asked_gloss import packed_batch
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, RELEASE


def train(arm):
    spec = read(PUBLIC / "split.json")
    assert sha(WORK / "train.json") == spec["train_sha256"]
    rows = read(WORK / "train.json")
    folder = WORK / arm
    folder.mkdir(exist_ok=False)
    face = Face(FACES[0])
    start, energy = monotonic(), snapshot()
    writer = restore(device="cuda").requires_grad_(False)
    # cuDNN needs a training-mode GRU to propagate gradients into its input.
    # The one-layer speller has no dropout and its parameters stay frozen.
    writer.speller.train()
    states, targets = [], []
    letters = {}
    for c in "abcdefghijklmnopqrstuvwxyz-":
        bits = np.unpackbits(np.frombuffer(face.cell(c), np.uint8))[
            writer.landmarks.cpu().numpy()
        ]
        letters[c] = int(bits @ (2 ** np.arange(len(bits))))
    for s in range(0, len(rows), 128):
        batch = rows[s : s + 128]
        raw, _ = packed_batch([r[0] for r in batch], face, "cuda")
        with torch.no_grad():
            states.append(writer.encode(cell_features(raw)))
        for r in batch:
            targets.append([letters[c] for c in r[1]] + [writer.stop])
    x = torch.cat(states)
    y = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(w) for w in targets], batch_first=True, padding_value=writer.stop
    ).cuda()
    length = torch.tensor(list(map(len, targets)), device="cuda")
    times = torch.arange(y.shape[1], device="cuda")[None]
    ordinary = torch.tensor([r[2] != "object_person" for r in rows], device="cuda")
    motion = [i for i, r in enumerate(rows) if r[2] == "motion"]
    torch.manual_seed(SEED)
    net = GlossResidual().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    losses, steps = [], 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(12):
        rng = np.random.default_rng(SEED + epoch)
        # Motion is rare. Give the counter-sense enough presentations in both arms.
        order = rng.permutation(
            np.concatenate([np.arange(len(rows)), np.tile(motion, 5)])
        )
        target = np.arange(len(order))
        if arm == "exchanged":
            groups = defaultdict(list)
            for j, i in enumerate(order):
                groups[face.cell(rows[i][0][8])].append(j)
            for indices in groups.values():
                target[indices] = rng.permutation(indices)
        loss_sum = 0.0
        for a in range(0, len(order), 128):
            ids = order[a : a + 128]
            labels = order[target[a : a + 128]]
            ctx = net(x[ids])
            codes = y[labels]
            valid = times < length[labels, None]
            logits = writer.teacher(ctx, codes)
            loss = F.cross_entropy(logits[valid], codes[valid])
            # Penalize deviations for retained lexical targets. Shuffle the
            # anchor designation with the target together in the control.
            loss = (
                loss + 0.1 * ((ctx - x[ids]).square().mean(1) * ordinary[labels]).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(ids)
            steps += 1
        losses.append(loss_sum / len(order))
        if monotonic() - start > 300 or torch.cuda.max_memory_allocated() > 6 * 2**30:
            raise RuntimeError("Residual fit exceeded its ceiling")
    out = folder / "adapter.safetensors"
    save_file({k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(out))
    fit = dict(
        sha256=sha(out),
        parent=sha(RELEASE / "writer.safetensors"),
        parameters=sum(p.numel() for p in net.parameters()),
        examples=len(rows),
        examples_per_pass=len(order),
        passes=12,
        steps=steps,
        seconds=monotonic() - start,
        peak_bytes=torch.cuda.max_memory_allocated(),
        energy=consumed(energy, snapshot()),
        loss=losses,
    )
    dump(folder / "weights.json", fit)
    dump(PUBLIC / (arm + "_fit.json"), fit)
    print(arm, steps, losses[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=["aligned", "exchanged", "both"])
    args = p.parse_args()
    with gpu_owner():
        for name in ("aligned", "exchanged") if args.arm == "both" else (args.arm,):
            train(name)
