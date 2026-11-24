"""Train the fitting-word memory on real conversational and list answer images."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.word_memory import WordMemory, fit_loss
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.measure_writer_energy import snapshot
from scripts.prepare_compose_dialogue import prepared, PUB as TRAIN_SOURCE
from scripts.prepare_dialogue_ink import dump, sha
from scripts.prepare_word_canvas import MODEL as PARENT
from scripts.prepare_word_memory import MODEL, PUB
from scripts.train_dialogue_subword import timed
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_span_choice import gpu_owner

SEED = 812641
STEPS = 3200


def fit(data, native, pools, mode, output):
    dest = output / mode
    parent = PARENT / mode / "masked"
    binding = dict(
        mode=mode,
        arm="word_memory",
        seed=SEED,
        steps=STEPS,
        reservation=sha(PUB / "reservation.json"),
        split=sha(TRAIN_SOURCE / "split.json"),
        parent=sha(parent / "writer.safetensors"),
    )
    if dest.exists():
        meta = json.loads((dest / "weights.json").read_text())
        if any(meta[k] != v for k, v in binding.items()) or meta["sha256"] != sha(
            dest / "writer.safetensors"
        ):
            raise ValueError("Existing checkpoint mismatch")
        return meta["schedule"]
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    clock, meter = monotonic(), snapshot()
    base = load(parent, "cuda")
    face, unit, _ = representation(mode)
    ds = corpus(data, "train", base, face, unit)
    palette, labels = ds.glyph_targets()
    if not torch.equal(base.palette.cpu(), palette):
        raise ValueError("Changed palette")
    freq = Counter(ds.index[w] for row in ds.rows for w in row[5].split())
    idx = torch.tensor([i for i, _ in freq.most_common(2048)], device=labels.device)
    cells = labels[idx].to(torch.uint8)
    lengths = (cells != 0).sum(1).to(torch.uint8)
    net = WordMemory(palette, mode, cells, lengths, ds.bank.features[idx]).cuda().eval()
    missing, extra = net.load_state_dict(base.state_dict(), strict=False)
    allowed = {
        "prototype_cells",
        "prototype_lengths",
        "prototype_features",
        "keys",
        "prior",
        "retrieve.weight",
        "retrieve.bias",
        "gate.weight",
        "gate.bias",
    }
    if (
        extra
        or set(missing) != allowed
        or net.prototype_words != tuple(ds.bank.raw[i] for i in idx.tolist())
    ):
        raise ValueError("Memory initialization mismatch")
    for name, p in net.named_parameters():
        p.requires_grad_(name in allowed)
    params = [p for p in net.parameters() if p.requires_grad]
    net.lookup(ds.bank)
    prep = timed(clock, meter)
    opt = torch.optim.AdamW(params, lr=0.002)
    rng = np.random.default_rng(SEED)
    seen = set()
    trace = []
    schedule = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    clock, meter = monotonic(), snapshot()
    for step in range(1, STEPS + 1):
        ix = np.concatenate(
            (
                rng.integers(native, size=16),
                rng.choice(pools["assembly"], 8),
                rng.choice(pools["spell"], 4),
                rng.integers(native, len(ds.rows), size=4),
            )
        )
        seen.update(ix.tolist())
        schedule.update(ix.astype("<i8").tobytes())
        loss = fit_loss(net, ds, ix, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3, error_if_nonfinite=True)
        opt.step()
        if step % 400 == 0:
            trace.append([step, float(loss.detach())])
            print(mode, trace[-1], flush=True)
            if (
                monotonic() - clock > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Fit exceeds ceiling")
    cost = timed(clock, meter)
    if any(
        not torch.equal(net.state_dict()[k], v) for k, v in base.state_dict().items()
    ):
        raise ValueError("Frozen parent changed")
    output.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output) as stage:
        path = Path(stage) / "finished"
        path.mkdir()
        f = path / "writer.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
            str(f),
        )
        dump(
            path / "weights.json",
            binding
            | dict(
                sha256=sha(f),
                schedule=schedule.hexdigest(),
                parameters=sum(p.numel() for p in net.parameters()),
                trainable=sum(p.numel() for p in params),
                max_passage_words=448,
                presentations=STEPS * 32,
                distinct=len(seen),
                prototype_count=len(idx),
                bytes_on_disk=f.stat().st_size,
                preparation=prep,
                fit=cost,
                trace=trace,
            ),
        )
        path.rename(dest)
    return schedule.hexdigest()


def train(output):
    torch.set_num_threads(2)
    print(torch.cuda.get_device_name(), torch.version.cuda, flush=True)
    reg = read_data()
    prepared_data = prepared()
    native = sum(r[0] == "train" for r in reg["rows"])
    extra = prepared_data["extra"]
    data = dict(rows=reg["rows"] + extra["rows"], pages=reg["pages"] | extra["pages"])
    schedules = [
        fit(data, native, prepared_data["pools"], mode, output)
        for mode in ("ink", "byte")
    ]
    if len(set(schedules)) != 1:
        raise ValueError("Training presentations differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL)
    options = parser.parse_args()
    with gpu_owner():
        train(options.output)
