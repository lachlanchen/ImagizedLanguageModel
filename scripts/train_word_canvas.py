"""Fit matched masked and causal image and byte writers on an identical schedule."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import hashlib
import json
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.word_canvas import WordCanvas, canvas_loss
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.measure_writer_energy import snapshot
from scripts.prepare_compose_dialogue import MODEL as PARENTS, prepared, PUB as SOURCE
from scripts.prepare_word_canvas import MODEL, PUB
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_dialogue_ink import dialogue_loss
from scripts.train_dialogue_subword import timed
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_span_choice import gpu_owner

SEED = 617243
STEPS = 3200


def fit(data, pools, native, mode, arm, root):
    parent = PARENTS / mode / "lists"
    dest = root / mode / arm
    binding = dict(
        parent=sha(parent / "writer.safetensors"),
        input_sha256=sha(SOURCE / "split.json"),
        reservation=sha(PUB / "reservation.json"),
        mode=mode,
        arm=arm,
        seed=SEED,
        steps=STEPS,
    )
    if dest.exists():
        meta = json.loads((dest / "weights.json").read_text())
        if any(meta[k] != v for k, v in binding.items()) or meta["sha256"] != sha(
            dest / "writer.safetensors"
        ):
            raise ValueError("Different existing fit")
        return meta["schedule"]
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    clock = monotonic()
    meter = snapshot()
    net = load(parent, "cuda")
    if arm == "masked":
        draft = WordCanvas(net.palette.cpu(), mode).cuda()
        shared = {k: v for k, v in net.state_dict().items() if k in draft.state_dict()}
        missing, extra = draft.load_state_dict(shared, strict=False)
        if extra or any(
            not k.startswith(
                (
                    "length_head.",
                    "length_embedding.",
                    "position.",
                    "ink_projection.",
                    "mask_image",
                    "context.",
                )
            )
            for k in missing
        ):
            raise ValueError("Unexpected inherited tensors")
        net = draft
    net.train()
    face, unit, _ = representation(mode)
    ds = corpus(data, "train", net, face, unit)
    palette, labels = ds.glyph_targets()
    if not torch.equal(palette, net.palette.cpu()):
        raise ValueError("Changed glyph inventory")
    prep = timed(clock, meter)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0002)
    schedule = hashlib.sha256()
    trace = []
    seen = set()
    torch.cuda.reset_peak_memory_stats()
    clock = monotonic()
    meter = snapshot()
    for step in range(1, STEPS + 1):
        ix = np.concatenate(
            (
                rng.integers(native, size=16),
                rng.choice(pools["assembly"], 8),
                rng.choice(pools["spell"], 4),
                rng.integers(native, len(ds.rows), size=4),
            )
        )
        schedule.update(ix.astype("<i8").tobytes())
        seen.update(ix.tolist())
        opt.zero_grad(set_to_none=True)
        loss = (
            canvas_loss(net, ds, ix, labels)
            if arm == "masked"
            else dialogue_loss(net, ds, ix, labels)[0]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3, error_if_nonfinite=True)
        opt.step()
        if step % 400 == 0:
            trace.append([step, float(loss.detach())])
            print(mode, arm, trace[-1], flush=True)
            if (
                monotonic() - clock > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Fit exceeds resource ceiling")
    cost = timed(clock, meter)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as stage:
        path = Path(stage) / "complete"
        path.mkdir()
        f = path / "writer.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
            str(f),
        )
        meta = binding | dict(
            sha256=sha(f),
            schedule=schedule.hexdigest(),
            parameters=sum(p.numel() for p in net.parameters()),
            max_passage_words=448,
            presentations=STEPS * 32,
            distinct=len(seen),
            preparation=prep,
            fit=cost,
            trace=trace,
        )
        dump(path / "weights.json", meta)
        path.rename(dest)
    return schedule.hexdigest()


def train(output):
    reg = read_data()
    p = prepared()
    native = sum(r[0] == "train" for r in reg["rows"])
    data = dict(
        rows=reg["rows"] + p["extra"]["rows"], pages=reg["pages"] | p["extra"]["pages"]
    )
    schedules = [
        fit(data, p["pools"], native, m, a, output)
        for a in ("masked", "causal")
        for m in ("ink", "byte")
    ]
    if len(set(schedules)) != 1:
        raise ValueError("Unmatched presentations")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=MODEL)
    with gpu_owner():
        train(p.parse_args().output)
