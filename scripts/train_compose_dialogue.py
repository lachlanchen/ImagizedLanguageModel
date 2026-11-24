"""Train free-running glyph and byte writers with controlled composition curricula."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.prepare_compose_dialogue import MODEL, PUB, prepared
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_dialogue_surface import ROOT_MODEL, representation, corpus
from scripts.train_dialogue_ink import dialogue_loss
from scripts.train_dialogue_evidence import SEED
from scripts.train_dialogue_subword import timed
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot


def fit(mode, arm, dest, data, pools, native):
    if dest.exists():
        raise FileExistsError(dest)
    net = load(ROOT_MODEL / mode, "cuda").train()
    face, unit, _ = representation(mode)
    torch.cuda.reset_peak_memory_stats()
    start = monotonic()
    meter = snapshot()
    ds = corpus(data, "train", net, face, unit)
    palette, labels = ds.glyph_targets()
    if not torch.equal(palette, net.palette.cpu()):
        raise ValueError("Writable surfaces changed")
    prep = timed(start, meter)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0002)
    uniform = np.random.default_rng(SEED)
    rng = np.random.default_rng(SEED + 1)
    bank = np.random.default_rng(SEED + 2)
    both = hashlib.sha256()
    seen = set()
    trace = []
    torch.cuda.reset_peak_memory_stats()
    start = monotonic()
    meter = snapshot()
    for step in range(1, 1501):
        general = uniform.integers(native, size=32)[:16]
        # Generate both complete stratum draws before subsampling to preserve paired cases.
        copies = rng.choice(pools["assembly"], 8)
        spelled = rng.choice(pools["spell"], 8)
        if arm == "assembly":
            ix = np.concatenate([general, copies, spelled])
        else:
            ix = np.concatenate(
                [
                    general,
                    copies[:4],
                    spelled[:4],
                    bank.integers(native, len(ds.rows), 8),
                ]
            )
        both.update(ix.astype("<i8").tobytes())
        seen.update(ix.tolist())
        loss, *_ = dialogue_loss(net, ds, ix, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        if step % 250 == 0:
            trace.append([step, float(loss.detach())])
            print(mode, arm, trace[-1], flush=True)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Resource ceiling exceeded")
    cost = timed(start, meter)
    dest.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=dest.parent) as stage:
        tmp = Path(stage) / "completed"
        tmp.mkdir()
        f = tmp / "writer.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
            str(f),
        )
        dump(
            tmp / "weights.json",
            dict(
                sha256=sha(f),
                parent=sha(ROOT_MODEL / mode / "writer.safetensors"),
                mode=mode,
                curriculum=arm,
                parameters=sum(p.numel() for p in net.parameters()),
                seed=SEED,
                steps=1500,
                presentations=48000,
                distinct=len(seen),
                batches_sha256=both.hexdigest(),
                input_sha256=sha(PUB / "split.json"),
                max_passage_words=448,
                trace=trace,
                preparation=prep,
                fit=cost,
            ),
        )
        tmp.rename(dest)


def train(root):
    old = read_data()
    p = prepared()
    native = sum(r[0] == "train" for r in old["rows"])
    for arm in ("assembly", "lists"):
        data = (
            old
            if arm == "assembly"
            else dict(
                pages=old["pages"] | p["extra"]["pages"],
                rows=old["rows"] + p["extra"]["rows"],
            )
        )
        for mode in ("ink", "byte"):
            dest = root / mode / arm
            if dest.exists():
                m = json.loads((dest / "weights.json").read_text())
                if (
                    m["mode"],
                    m["curriculum"],
                    m["steps"],
                    m["parent"],
                    m["input_sha256"],
                    m["sha256"],
                ) != (
                    mode,
                    arm,
                    1500,
                    sha(ROOT_MODEL / mode / "writer.safetensors"),
                    sha(PUB / "split.json"),
                    sha(dest / "writer.safetensors"),
                ):
                    raise ValueError(f"Different saved fit: {dest}")
            else:
                fit(mode, arm, dest, data, p["pools"], native)
        schedules = [
            json.loads((root / m / arm / "weights.json").read_text())["batches_sha256"]
            for m in ("ink", "byte")
        ]
        if schedules[0] != schedules[1]:
            raise ValueError("Unmatched visual and byte exposures")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=MODEL)
    a = p.parse_args()
    with gpu_owner():
        train(a.output)
