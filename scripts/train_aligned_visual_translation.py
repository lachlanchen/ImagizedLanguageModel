"""Fit equal-exposure visual translators before and after monotone source alignment."""

import copy
import fcntl
import hashlib
import json
import resource
import time

import numpy as np
import torch
from ilm.visual_lm.bitmap_alignment import blocks
from safetensors.torch import load_file, save_file

from ilm.visual_lm.block_translation import BlockTranslator
from ilm.visual_lm.idiom_ink import visual_input
from scripts.align_visual_translation import (
    ROOT,
    WORK,
    load,
    require,
    sha,
    tiles,
    write_new,
)
from scripts.infer_visual_phrase import MODELS

SEED = 7821


def encode(rows, arm):
    size = max(len(r["x"]) for r in rows)
    x = np.zeros((len(rows), size), np.int32)
    counts = np.zeros_like(x)
    y = np.zeros((*x.shape, BlockTranslator.slots), np.int32)
    lengths = []
    clipped = 0
    for i, row in enumerate(rows):
        parts = blocks(row["x"], row[arm])
        x[i, : len(parts)] = row["x"]
        counts[i, : len(parts)] = list(map(len, parts))
        clipped += max(map(len, parts)) > BlockTranslator.slots
        lengths.append(len(parts))
        for j, part in enumerate(parts):
            kept = part[: BlockTranslator.slots]
            y[i, j, : len(kept)] = kept
    return [
        torch.tensor(v, device="cuda") for v in (x, np.asarray(lengths), counts, y)
    ], int(clipped)


def read_rows(split):
    path = WORK / (split + ".jsonl")
    require(
        sha(path) == load(WORK / "preparation.json")["splits"][split]["sha256"],
        "Changed split",
    )
    with path.open() as f:
        return [json.loads(line) for line in f]


def run():
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    bank = tiles()
    require(
        hashlib.sha256(bank.tobytes()).hexdigest()
        == load(WORK / "preparation.json")["bank_sha256"],
        "Bank changed",
    )
    images = torch.as_tensor(np.unpackbits(bank, axis=1), device="cuda").bool()
    ids = np.unique(bank, axis=0, return_inverse=True)[1]
    shapes = torch.as_tensor(ids, device="cuda")
    motor = load_file(str(MODELS / "editor.safetensors"))
    net = BlockTranslator(motor["landmarks"])
    net.motor.load_state_dict(
        {k[6:]: v for k, v in motor.items() if k.startswith("motor.")}
    )
    base = net.cuda()
    features = visual_input(images, base.landmarks)
    bits = images[:, base.landmarks]
    rows = read_rows("train")[:32768]
    out = WORK / "fit"
    out.mkdir()
    results = {}
    for arm in ("y", "z"):
        net = copy.deepcopy(base)
        opt = torch.optim.AdamW(
            (p for p in net.parameters() if p.requires_grad), lr=0.001
        )
        data, clipped = encode(rows, arm)
        schedule = hashlib.sha256()
        rng = np.random.default_rng(SEED)
        losses = []
        began = time.monotonic()
        for epoch in range(8):
            order = rng.permutation(len(rows))
            schedule.update(order.astype("<i4").tobytes())
            total = 0.0
            for offset in range(0, len(order), 96):
                idx = torch.as_tensor(order[offset : offset + 96], device="cuda")
                x, lengths, counts, y = (v[idx].long() for v in data)
                width = int(lengths.max())
                x, counts, y = x[:, :width], counts[:, :width], y[:, :width]
                loss = net.loss(
                    features[x],
                    lengths,
                    counts,
                    bits[y],
                    shapes[x][:, :, None] == shapes[y],
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 1, error_if_nonfinite=True
                )
                opt.step()
                total += loss.item() * len(idx)
            losses.append(total / len(rows))
            print(arm, "epoch", epoch + 1, "loss", losses[-1], flush=True)
            require(
                torch.cuda.max_memory_allocated() < 8 * 2**30
                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 6 * 2**20,
                "Memory ceiling",
            )
            require(
                int(
                    next(
                        v
                        for v in open("/proc/meminfo")
                        if v.startswith("MemAvailable:")
                    ).split()[1]
                )
                >= 24 * 2**20,
                "Host reserve",
            )
        path = out / (arm + ".safetensors")
        save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()},
            str(path),
            metadata={"format": "ilm-block-translation-v1"},
        )
        results[arm] = dict(
            loss=losses,
            seconds=time.monotonic() - began,
            clipped_sentences=clipped,
            schedule=schedule.hexdigest(),
            sha256=sha(path),
        )
        del data, opt, net
    require(
        results["y"]["schedule"] == results["z"]["schedule"],
        "Training schedule differs",
    )
    write_new(
        out / "fit.json",
        dict(
            rows=len(rows),
            changed=sum(r["y"] != r["z"] for r in rows),
            arms=results,
            parameters=sum(p.numel() for p in base.parameters()),
            trainable=sum(p.numel() for p in base.parameters() if p.requires_grad),
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
