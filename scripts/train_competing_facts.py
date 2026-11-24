"""Compare whole-answer interval likelihood and same-page answer preferences."""

import argparse
from pathlib import Path
from hashlib import sha256
from time import monotonic
import math
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_competing_facts import (
    KINDS,
    MODEL,
    PUB,
    WORK,
    PAIRED,
    dump,
    read,
    sha,
)
from scripts.train_paired_evidence import byte_features, gpu_owner
from scripts.train_visual_cursor import pad
from scripts.write_paired_evidence import reader
from ilm.visual_lm.paired_evidence import EvidenceWriter

STEPS = 4000
SEED = 291407


def alternative_mass(spans, positions, lengths):
    """All occurrences of each stopped physical fact represent one event."""
    valid = positions >= 0
    sample = torch.arange(len(spans), device=spans.device)[:, None]
    probs = spans[sample, positions.clamp_min(0), lengths[:, None] - 1]
    return probs.masked_fill(~valid, -torch.inf).logsumexp(-1)


def sample(groups, arrays, step):
    rng = np.random.default_rng([SEED, step])
    group = groups[KINDS[(step - 1) % 4]]
    q = []
    s = []
    positions = []
    lengths = []
    for gi in rng.integers(len(group), size=16):
        page = group[gi]
        facts = page["facts"]
        a = rng.choice([i for i, f in enumerate(facts) if f["rivals"]])
        b = rng.choice(facts[a]["rivals"])
        d = page["doc"]
        source = arrays["p"][arrays["po"][d] : arrays["po"][d + 1]]
        for f in (facts[a], facts[b]):
            idx = rng.choice(f["rows"])
            q.append(arrays["q"][arrays["qo"][idx] : arrays["qo"][idx + 1]])
            s.append(source)
            positions.append(f["at"])
            lengths.append(len(f["y"]))
    offset = np.full((len(q), max(map(len, positions))), -1, dtype=np.int32)
    for i, at in enumerate(positions):
        offset[i, : len(at)] = at
    return pad(q), pad(s), offset, np.array(lengths, np.int32)


def trained(arm, device, run=None):
    folder = run or MODEL
    path = folder / (arm + ".safetensors")
    if sha(path) != read((run or PUB) / (arm + ".json"))["weights"]:
        raise ValueError("Checkpoint differs")
    net = EvidenceWriter().to(device)
    net.load_state_dict(load_file(str(path), device=device))
    return net.eval()


def fit(
    arm,
    groups,
    arrays,
    run,
    *,
    model_dir=MODEL,
    metrics_dir=PUB,
    initial=None,
    resume_dir=None,
):
    from scripts.measure_writer_energy import snapshot, consumed

    dest = (run or model_dir) / (arm + ".safetensors")
    receipt = (run or metrics_dir) / (arm + ".json")
    if receipt.exists():
        if sha(dest) != read(receipt)["weights"]:
            raise ValueError("Changed trained reader")
        return
    byte = arm == "bytes"
    net = (
        initial
        if initial is not None
        else reader("byte-evidence" if byte else "evidence", "cuda")
    )
    net.train()
    learn = (
        "context",
        "ask",
        "fuse",
        "reason",
        "evidence",
        "where",
        "summary",
        "length",
        "left",
        "right",
        "gain",
    )
    params = []
    for name, p in net.named_parameters():
        p.requires_grad_(name.split(".")[0] in learn)
        if p.requires_grad:
            params.append(p)
    data = (
        byte_features(arrays["chars"])
        if byte
        else torch.from_numpy(observations({"tiles": arrays["tiles"]}, "ink"))
    )
    bank = net.ink(data.cuda()).detach()
    opt = torch.optim.AdamW(params, lr=0.00035, weight_decay=0.015, fused=True)
    pair = torch.arange(32, device="cuda") ^ 1
    trace = []
    schedule = bytes(32)
    first = 1
    earlier_seconds = earlier_joules = 0.0
    snapshot_path = resume_dir / (arm + "-resume.pt") if resume_dir else None
    if snapshot_path:
        bindings = dict(
            arm=arm,
            training=sha(WORK / "train.json"),
            base=sha(PAIRED / "corpus.npz"),
            split=sha(metrics_dir / "split.json"),
            implementation=sha(Path(__file__)),
            model=sha(
                Path(__file__).resolve().parents[1]
                / "ilm/visual_lm/endpoint_interaction.py"
            ),
        )
        if snapshot_path.exists():
            state = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            if state["bindings"] != bindings:
                raise ValueError("Resume inputs changed")
            first = state["step"] + 1
            net.load_state_dict(state["net"])
            opt.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])
            trace = state["trace"]
            schedule = bytes.fromhex(state["schedule"])
            earlier_seconds = state["seconds"]
            earlier_joules = state["joules"]
    torch.cuda.synchronize()
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for step in range(first, STEPS + 1):
        batch = sample(groups, arrays, step)
        digest = sha256(schedule)
        for a in batch:
            digest.update(a.tobytes())
        schedule = digest.digest()
        q, s, at, lens = [torch.as_tensor(v, device="cuda").long() for v in batch]
        opt.param_groups[0]["lr"] = (
            0.00035
            * min(1, step / 100)
            * (0.12 + 0.88 * (1 + math.cos(math.pi * step / STEPS)) / 2)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spans = net.read(bank, q, s)[0]
            pos = alternative_mass(spans, at, lens)
            neg = alternative_mass(spans, at[pair], lens[pair])
            loss = -pos.mean()
            if arm in ("contrast", "bytes"):
                loss = loss + 0.75 * F.softplus(math.log(4) + neg - pos).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 200 == 0:
            trace.append(
                [step, float(loss.detach()), float((pos > neg).float().mean())]
            )
            print(arm, trace[-1], round(monotonic() - start, 1), flush=True)
            if (
                torch.cuda.max_memory_allocated() > 12 * 2**30
                or earlier_seconds + monotonic() - start > 720
            ):
                raise RuntimeError("Trial resource ceiling")
        if snapshot_path and step % 1000 == 0:
            tmp = snapshot_path.with_suffix(".tmp")
            energy = consumed(meter, snapshot())["joules"]
            torch.save(
                dict(
                    bindings=bindings,
                    step=step,
                    net=net.state_dict(),
                    optimizer=opt.state_dict(),
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                    trace=trace,
                    schedule=schedule.hex(),
                    seconds=earlier_seconds + monotonic() - start,
                    joules=earlier_joules + energy,
                ),
                tmp,
            )
            tmp.replace(snapshot_path)
    torch.cuda.synchronize()
    secs = monotonic() - start
    energy = consumed(meter, snapshot())
    dest.parent.mkdir(exist_ok=True, parents=True)
    save_file(net.cpu().state_dict(), str(dest))
    dump(
        receipt,
        dict(
            arm=arm,
            updates=STEPS,
            draws=32 * STEPS,
            seconds=earlier_seconds + secs,
            joules=earlier_joules + energy["joules"],
            peak_bytes=torch.cuda.max_memory_allocated(),
            weights=sha(dest),
            corpus=sha(PUB / "corpus.json"),
            schedule=schedule.hex(),
            trainable=sum(p.numel() for p in params),
            trace=trace,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    meta = read(PUB / "corpus.json")
    if (
        sha(WORK / "train.json") != meta["files"]["train.json"]
        or sha(PAIRED / "corpus.npz") != meta["parents"]
    ):
        raise ValueError("Changed training source")
    groups = read(WORK / "train.json")
    with np.load(PAIRED / "corpus.npz") as dat:
        arrays = {k: dat[k] for k in ("p", "po", "q", "qo", "tiles", "chars")}
    with gpu_owner():
        for arm in ("likelihood", "contrast", "bytes"):
            fit(arm, groups, arrays, args.run_dir)


if __name__ == "__main__":
    main()
