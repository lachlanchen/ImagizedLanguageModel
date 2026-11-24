"""Fit matched additive and two-reading heads on the same complete pictured contexts."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.fact_binding import (
    BindingHead,
    frozen_features,
    pack_paths,
    score_paths,
)
from scripts.prepare_fact_binding import MODEL, PUB, WORK, ROOT, dump, read, sha
from scripts.prepare_complete_fact_sets import MODEL as BASE
from scripts.train_visual_cursor import pad
from scripts.train_paired_evidence import gpu_owner
from scripts.measure_writer_energy import consumed, snapshot
from scripts.write_multiple_ink_runs import load_reader

SEED = 719023
STEPS = 2000
BATCH = 12


def tensors(rows, device):
    return [
        torch.as_tensor(pad([r[k] for r in rows]), device=device).long()
        for k in ("qi", "pi")
    ]


def corpus(split):
    info = read(PUB / "corpus.json")
    for name in (split + ".json", "tiles.npy"):
        if sha(WORK / name) != info["files"][name]:
            raise ValueError("Binding corpus changed: " + name)
    return read(WORK / (split + ".json")), np.load(WORK / "tiles.npy")


def train(arm, output_dir=None):
    rows, tiles = corpus("train")
    rng = np.random.default_rng(SEED)
    n = STEPS * BATCH
    ids = np.concatenate(
        [rng.permutation(len(rows)) for _ in range(1 + n // len(rows))]
    )[:n].reshape(-1, BATCH)
    dest = (output_dir or MODEL) / arm
    dest.mkdir(parents=True, exist_ok=True)
    weights = dest / "head.safetensors"
    receipt = dest / "fit.json"
    bindings = dict(
        schedule=sha256(ids.astype("<i8").tobytes()).hexdigest(),
        corpus=sha(PUB / "corpus.json"),
        parent=sha(BASE / "contrast/survival.safetensors"),
        trainer=sha(Path(__file__)),
        math=sha(ROOT / "ilm/visual_lm/fact_binding.py"),
        protocol=sha(ROOT / "references/fact_binding_protocol.md"),
    )
    with gpu_owner():
        if receipt.exists():
            result = read(receipt)
            if bindings != result["bindings"] or sha(weights) != result["weights"]:
                raise ValueError("Completed fit has different bindings")
            return
        torch.manual_seed(SEED)
        reader = load_reader("cuda", True, BASE / "contrast").requires_grad_(False)
        bank = reader.ink(
            torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        )
        head = BindingHead().cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=0.0015)
        trace = []
        torch.cuda.synchronize()
        start, meter = monotonic(), snapshot()
        for i, group in enumerate(ids, 1):
            batch = [rows[j] for j in group]
            q, p = tensors(batch, "cuda")
            feat, role, count = frozen_features(reader, bank, q, p)
            h = head(feat)
            bounds = torch.from_numpy(pack_paths([r["paths"] for r in batch])).cuda()
            score, raw = score_paths(role, count, h, bounds, arm == "mixture")
            loss = F.softplus(1 + score[:, 2:] - score[:, :2]).mean()
            loss = (
                loss
                + 0.002 * h[p >= 2].square().mean()
                + 0.0005 * raw[:, :2].square().mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            if i == 1 or i % 250 == 0:
                trace.append(dict(step=i, loss=loss.item()))
                print(arm, trace[-1], flush=True)
                if (
                    monotonic() - start > 1800
                    or torch.cuda.max_memory_allocated() > 12 * 2**30
                ):
                    raise RuntimeError("Fit exceeded registered resource ceilings")
        torch.cuda.synchronize()
        seconds, joules = monotonic() - start, consumed(meter, snapshot())["joules"]
        tmp = weights.with_suffix(".tmp")
        save_file(head.cpu().state_dict(), str(tmp))
        tmp.replace(weights)
        dump(
            receipt,
            dict(
                arm=arm,
                bindings=bindings,
                weights=sha(weights),
                seconds=seconds,
                joules=joules,
                trainable=sum(t.numel() for t in head.parameters()),
                frozen=sum(t.numel() for t in reader.parameters()),
                peak_bytes=torch.cuda.max_memory_allocated(),
                trace=trace,
                steps=STEPS,
                presentations=n,
                distinct=len(set(ids.flatten())),
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("linear", "mixture"))
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    train(args.arm, args.output_dir)
