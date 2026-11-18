"""Train an autoregressive visual reader and writer from real aligned Chinese passages."""

from collections import Counter
import fcntl
import hashlib
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.bitmap_alignment import blocks
from ilm.visual_lm.causal_glyph_translator import (
    CausalGlyphTranslator,
    COPY,
    EXEMPLAR,
    DRAW,
    END,
)
from ilm.visual_lm.idiom_ink import visual_input
from scripts.train_aligned_visual_translation import (
    ROOT,
    read_rows,
    sha,
    tiles,
    write_new,
    MODELS,
)

OUT = ROOT / "artifacts/causal_glyph_translation_v1"
SEED = 90421


def prepare(rows, canonical, *, examples=None):
    canonical_rows = [
        (canonical[r["x"]].tolist(), canonical[r["z"]].tolist()) for r in rows
    ]
    if examples is None:
        frequent = Counter(v for _, seq in canonical_rows for v in seq)
        examples = np.array([v for v, _ in frequent.most_common(2048)])
    index = {int(v): i for i, v in enumerate(examples)}
    x = np.zeros((len(rows), max(len(src) for src, _ in canonical_rows)), np.int32)
    y = np.zeros((len(rows), max(len(tgt) for _, tgt in canonical_rows)), np.int32)
    column = np.zeros_like(y)
    mode = np.full((len(rows), y.shape[1] + 1), END, np.int8)
    addresses = np.zeros_like(y)
    xl, yl = [], []
    for b, (src, tgt) in enumerate(canonical_rows):
        x[b, : len(src)] = src
        y[b, : len(tgt)] = tgt
        xl.append(len(src))
        yl.append(len(tgt))
        c = [i for i, part in enumerate(blocks(src, tgt)) for _ in part]
        column[b, : len(tgt)] = c
        mode[b, : len(tgt)] = [
            COPY if v == src[i] else EXEMPLAR if v in index else DRAW
            for v, i in zip(tgt, c)
        ]
        addresses[b, : len(tgt)] = [index.get(v, 0) for v in tgt]
    return examples, [x, y, column, mode, addresses, np.array(xl), np.array(yl)]


def train():
    start = time.monotonic()
    OUT.mkdir()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    rows = read_rows("train")
    packed, canonical = np.unique(tiles(), axis=0, return_inverse=True)
    packed_hash = hashlib.sha256(packed.tobytes()).hexdigest()
    examples, data = prepare(rows, canonical)
    target_lengths = data[-1]
    bank = torch.from_numpy(np.unpackbits(packed, axis=1)).bool()
    parent = load_file(str(MODELS / "editor.safetensors"))
    model = CausalGlyphTranslator(parent["landmarks"], bank[examples])
    model.motor.load_state_dict(
        {k[6:]: v for k, v in parent.items() if k.startswith("motor.")}
    )
    model.cuda()
    features = visual_input(bank.cuda(), model.landmarks)
    x, y, col, route, address, n, m = [torch.as_tensor(v, device="cuda") for v in data]
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.001
    )
    rng = np.random.default_rng(SEED)
    losses = []
    began = time.monotonic()
    for epoch in range(8):
        batches = []
        for bucket in np.array_split(rng.permutation(len(rows)), 16):
            bucket = bucket[np.argsort(target_lengths[bucket], kind="stable")]
            batches.extend(bucket[i : i + 96] for i in range(0, len(bucket), 96))
        rng.shuffle(batches)
        total = 0.0
        for batch in batches:
            idx = torch.as_tensor(batch, device="cuda")
            l, r = int(n[idx].max()), int(m[idx].max())
            source = features[x[idx, :l].long()]
            target = features[y[idx, :r].long()]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(
                    source,
                    n[idx],
                    target,
                    m[idx],
                    col[idx, :r].long(),
                    route[idx, : r + 1].long(),
                    address[idx, :r].long(),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            total += loss.item() * len(batch)
        losses.append(total / len(rows))
        if torch.cuda.max_memory_allocated() >= 10 * 2**30:
            raise RuntimeError("Device memory limit")
        with open("/proc/meminfo") as f:
            if (
                int(
                    next(
                        line for line in f if line.startswith("MemAvailable:")
                    ).split()[1]
                )
                < 24 * 2**20
            ):
                raise RuntimeError("Host reserve")
        print(
            "epoch",
            epoch + 1,
            "loss",
            losses[-1],
            "elapsed",
            time.monotonic() - began,
            flush=True,
        )
    ckpt = OUT / "translator.safetensors"
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
        str(ckpt),
        metadata={"format": "ilm-causal-glyph-v1"},
    )
    write_new(
        OUT / "fit.json",
        dict(
            checkpoint_sha256=sha(ckpt),
            bank_sha256=packed_hash,
            rows=len(rows),
            targets=sum(map(len, (r["z"] for r in rows))),
            epochs=8,
            steps=len(batches) * 8,
            loss=losses,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            training_seconds=time.monotonic() - began,
            total_seconds=time.monotonic() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train()
