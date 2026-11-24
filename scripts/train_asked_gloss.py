"""Train a pictured-question focus from answer-image likelihood; run matched controls."""

import argparse
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.gloss_focus import GlossFocus, encoding_windows, cell_features
from scripts.prepare_asked_gloss import WORK, ROOT, QUESTIONS, dump, read, sha
from scripts.write_visual_gloss import restore
from scripts.measure_writer_energy import consumed, snapshot
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

SEED = 241127
ARMS = ("answer", "locations", "exchanged")


def rows_and_split():
    split = read(WORK / "split.json")
    if sha(WORK / "rows.json") != split["rows_sha256"]:
        raise ValueError("Clause split changed")
    return read(WORK / "rows.json"), split


def question_batch(face, device):
    strings = [q for pair in QUESTIONS for q in pair]
    return packed_batch(strings, face, device)


def packed_batch(strings, face, device):
    lengths = torch.tensor([len(s) for s in strings], device=device)
    raw = np.zeros((len(strings), max(map(len, strings)), 128), np.uint8)
    for i, s in enumerate(strings):
        raw[i, : len(s)] = np.frombuffer(face(s), np.uint8).reshape(-1, 128)
    return torch.from_numpy(raw).to(device), lengths


@torch.no_grad()
def cache():
    rows, split = rows_and_split()
    train = [r for r in rows if r["split"] == "train"]
    writer = restore(device="cuda").requires_grad_(False)
    face = Face(FACES[0])
    n = len(train)
    width = max(len(r["page"]) for r in train)
    ctx = torch.zeros(n, width, 160, dtype=torch.float16)
    ll = torch.full((n, 2, width), -torch.inf)
    started = monotonic()
    energy = snapshot()
    for a in range(0, n, 96):
        batch = train[a : a + 96]
        b = len(batch)
        pix, lens = packed_batch([r["page"] for r in batch], face, "cuda")
        f = encoding_windows(writer, cell_features(pix))
        w = f.shape[1]
        ctx[a : a + b, :w] = f.half().cpu()
        valid = torch.arange(w, device="cuda")[None] < lens[:, None]
        owners = torch.arange(b, device="cuda")[:, None].expand(b, w)[valid]
        for role in range(2):
            raw, lengths = packed_batch(
                [r["gloss"][role][0] + " " for r in batch], face, "cuda"
            )
            shifts = torch.arange(7, -1, -1, device="cuda")
            bit = ((raw[..., None].long() >> shifts) & 1).flatten(-2)[
                ..., writer.landmarks
            ]
            codes = (bit * (2 ** torch.arange(bit.shape[-1], device="cuda"))).sum(-1)
            times = torch.arange(codes.shape[1], device="cuda")[None]
            codes.masked_fill_(times >= lengths[:, None] - 1, writer.stop)
            parts = []
            for x, ix in zip(f[valid].split(384), owners.split(384), strict=True):
                targets = codes[ix]
                scores = writer.teacher(x, targets).log_softmax(-1)
                terms = scores.gather(-1, targets[..., None]).squeeze(-1)
                parts.append((terms * (times < lengths[ix, None])).sum(1))
            tile = f.new_full((b, w), -torch.inf)
            tile[valid] = torch.cat(parts)
            ll[a : a + b, role, :w] = tile.cpu()
    out = WORK / "emissions.npz"
    np.savez(out, context=ctx.numpy(), logp=ll.numpy())
    dump(
        WORK / "cache.json",
        dict(
            split=split["rows_sha256"],
            cache=sha(out),
            seconds=monotonic() - started,
            energy=consumed(energy, snapshot()),
        ),
    )


def fit(arm, questions=None, reader_type=GlossFocus):
    rows, split = rows_and_split()
    rows = [r for r in rows if r["split"] == "train"]
    cacheinfo = read(WORK / "cache.json")
    if (
        split["rows_sha256"] != cacheinfo["split"]
        or sha(WORK / "emissions.npz") != cacheinfo["cache"]
    ):
        raise ValueError("Stale or incomplete visual emission cache")
    with np.load(WORK / "emissions.npz") as d:
        x = torch.from_numpy(d["context"]).cuda()
        logp = torch.from_numpy(d["logp"]).cuda()
    lens = torch.tensor([len(r["page"]) for r in rows], device="cuda")
    truth = torch.tensor([r["focus"] for r in rows], device="cuda")
    work = WORK / arm
    work.mkdir(exist_ok=False)
    torch.manual_seed(SEED)
    reader = reader_type().cuda()
    opt = torch.optim.AdamW(reader.parameters(), lr=0.001)
    templates = questions if questions is not None else QUESTIONS[:4]
    raw, ql = packed_batch([q for pair in templates for q in pair], Face(FACES[0]), "cuda")
    q = cell_features(raw)
    role = torch.arange(2, device="cuda")[None]
    steps = 0
    trace = []
    energy = snapshot()
    t = monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(12):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(len(rows))
        flip = torch.as_tensor(
            rng.random(len(rows)) < 0.5, device="cuda", dtype=torch.long
        )
        phr = torch.as_tensor(rng.integers(0, len(templates), (len(rows), 2)), device="cuda")
        loss_sum = 0.0
        for part in np.array_split(order, (len(rows) + 127) // 128):
            ix = torch.as_tensor(part, device="cuda")
            qs = reader.read_questions(q, ql)[2 * phr[ix] + role]
            logits = reader(x[ix].float(), lens[ix], qs)
            if arm == "locations":
                loss = F.cross_entropy(logits.flatten(0, 1), truth[ix].flatten())
            else:
                ev = logp[ix]
                if arm.endswith("exchanged"):
                    ev = ev[
                        torch.arange(len(ix), device="cuda")[:, None],
                        role ^ flip[ix, None],
                    ]
                loss = -torch.logsumexp(logits.log_softmax(-1) + ev, -1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 2.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(ix)
            steps += 1
        trace.append(loss_sum / len(rows))
        if monotonic() - t > 900 or torch.cuda.max_memory_allocated() > 8 * 2**30:
            raise RuntimeError("Training resource limit")
    out = work / "focus.safetensors"
    save_file(
        {k: v.cpu().contiguous() for k, v in reader.state_dict().items()}, str(out)
    )
    dump(
        work / "fit.json",
        dict(
            arm=arm,
            seconds=monotonic() - t,
            energy=consumed(energy, snapshot()),
            steps=steps,
            examples=len(rows),
            passes=12,
            peak_bytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in reader.parameters()),
            sha256=sha(out),
            cache=cacheinfo["cache"],
            loss=trace,
        ),
    )
    print(arm, steps, trace[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", *ARMS, "all"))
    action = parser.parse_args().action
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with gpu_owner():
        if action in ("cache", "all"):
            if (WORK / "emissions.npz").exists():
                raise FileExistsError("Use a fresh work directory for caching")
            cache()
        for arm in ARMS:
            if action in (arm, "all"):
                fit(arm)
