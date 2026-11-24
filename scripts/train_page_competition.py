"""Train on complete optical reference placements and unrestricted whole-page alternatives."""

import argparse
from hashlib import sha256
from math import cos, pi
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.fact_binding import fixed
from ilm.visual_lm.multiple_ink_runs import best_path
from ilm.visual_lm.page_competition import best_reference, spans, mistake_loss
from ilm.visual_lm.run_survival import utilities, supervised
from scripts.evaluate_multiple_ink_runs import grade
from scripts.prepare_page_competition import WORK, PUB, MODEL, ROOT, read, sha, dump
from scripts.prepare_complete_fact_sets import MODEL as PREVIOUS
from scripts.train_visual_cursor import pad
from scripts.train_paired_evidence import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from scripts.write_multiple_ink_runs import load_reader

STEPS = 4608
SEED = 731021


def schedule(rows):
    rng = np.random.default_rng(SEED)
    ids = []
    for lang, n in (("en", 4), ("zh", 12)):
        eligible = np.asarray(
            [i for i, r in enumerate(rows) if r["lang"] == lang], np.int64
        )
        size = STEPS * n
        if size < len(eligible):
            raise ValueError("Schedule does not include the full corpus")
        seq = np.concatenate(
            [rng.permutation(eligible) for _ in range(1 + size // len(eligible))]
        )
        ids.append(seq[:size].reshape(-1, n))
    return np.concatenate(ids, axis=1)


def train(arm, run=None):
    device = "cuda"
    corpus = read(PUB / "corpus.json")
    for name, h in corpus["files"].items():
        if sha(WORK / name) != h:
            raise ValueError("Training data changed")
    rows = read(WORK / "training.json")
    tiles = np.load(WORK / "training-tiles.npy")
    ids = schedule(rows)
    dest = run or MODEL / arm
    dest.mkdir(parents=True, exist_ok=True)
    weights = dest / "survival.safetensors"
    receipt = dest / "survival.json"
    state = (run or WORK) / (arm + "-resume.pt")
    bindings = dict(
        arm=arm,
        corpus=sha(PUB / "corpus.json"),
        fresh=sha(PUB / "fresh.json"),
        schedule=sha256(ids.tobytes()).hexdigest(),
        parent=sha(PREVIOUS / "contrast/survival.safetensors"),
        trainer=sha(Path(__file__)),
        objective=sha(ROOT / "ilm/visual_lm/page_competition.py"),
        protocol=sha(ROOT / "references/page_competition_protocol.md"),
    )
    with gpu_owner():
        if receipt.exists():
            r = read(receipt)
            if r["bindings"] != bindings or sha(weights) != r["weights"]:
                raise ValueError("Inconsistent released checkpoint")
            return
        torch.manual_seed(SEED)
        model = load_reader(device, True, PREVIOUS / "contrast").train()
        model.ink.requires_grad_(False)
        bank = model.ink(
            torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        ).detach()
        opt = torch.optim.AdamW(
            [w for w in model.parameters() if w.requires_grad],
            lr=0.0002,
            weight_decay=0.02,
            fused=True,
        )
        begin = 0
        seconds = joules = 0.0
        trace = []
        if state.exists():
            s = torch.load(state, map_location="cpu", weights_only=True)
            if s["bindings"] != bindings:
                raise ValueError("Resume identity mismatch")
            model.load_state_dict(s["model"])
            opt.load_state_dict(s["optimizer"])
            torch.set_rng_state(s["cpu"])
            torch.cuda.set_rng_state(s["cuda"])
            begin, seconds, joules, trace = (
                s["step"],
                s["seconds"],
                s["joules"],
                s["trace"],
            )
        torch.cuda.reset_peak_memory_stats()
        start, meter = monotonic(), snapshot()
        for step in range(begin, STEPS):
            batch = [rows[i] for i in ids[step]]
            q, p = [
                torch.as_tensor(pad([r[k] for r in batch]), device=device).long()
                for k in ("qi", "pi")
            ]
            keep = p >= 2
            opt.param_groups[0]["lr"] = (
                0.0002
                * min(1, (step + 1) / 100)
                * (0.15 + 0.85 * (1 + cos((step + 1) * pi / STEPS)) / 2)
            )
            with torch.autocast(device, dtype=torch.bfloat16):
                raw, count = model(bank, q, p)
            role = utilities(raw.float(), keep)
            rs, cs = [fixed(x.detach()).cpu().numpy() for x in (role, count)]
            good, free, gaps, wrong = [], [], [], []
            for j, r in enumerate(batch):
                u = rs[j, : len(r["pi"])]
                tags, pos = best_reference(u, cs[j], r["options"])
                good.append(tags)
                if arm == "competition":
                    predicted, value = best_path(u, cs[j])
                    if value < pos:
                        raise ValueError("Unrestricted path below positive")
                    parts = [r["p"][a:b] for a, b in spans(predicted)]
                    free.append(predicted)
                    gaps.append((value - pos) / 1000.0)
                    wrong.append(not grade(parts, r["refs"])["exact"])
            y = torch.as_tensor(pad(good), device=device).long()
            local = supervised(raw, y, keep) + 0.18 * F.cross_entropy(
                count.float(), (y == 1).sum(1)
            )
            loss = local
            if arm == "competition":
                z = torch.as_tensor(pad(free), device=device).long()
                added = mistake_loss(
                    role,
                    count,
                    z,
                    y,
                    keep,
                    torch.tensor(wrong, device=device),
                    torch.tensor(gaps, device=device),
                )
                loss = loss + 0.12 * added
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            n = step + 1
            if n == 1 or n % 256 == 0:
                tr = dict(step=n, local=local.item())
                if wrong:
                    tr.update(errors=sum(wrong), mean_gap=float(np.mean(gaps)))
                trace.append(tr)
                print(arm, tr, flush=True)
            if n % 768 == 0:
                torch.cuda.synchronize()
                elapsed = seconds + monotonic() - start
                energy = joules + consumed(meter, snapshot())["joules"]
                if elapsed > 3000 or torch.cuda.max_memory_allocated() > 12 * 2**30:
                    raise RuntimeError("Registered resource ceiling exceeded")
                tmp = state.with_suffix(".tmp")
                torch.save(
                    dict(
                        step=n,
                        seconds=elapsed,
                        joules=energy,
                        trace=trace,
                        model=model.state_dict(),
                        optimizer=opt.state_dict(),
                        cpu=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state(),
                        bindings=bindings,
                    ),
                    tmp,
                )
                tmp.replace(state)
        tmp = weights.with_suffix(".tmp")
        save_file(model.cpu().state_dict(), str(tmp))
        tmp.replace(weights)
        dump(
            receipt,
            dict(
                weights=sha(weights),
                bindings=bindings,
                steps=STEPS,
                presentations=ids.size,
                distinct=len(rows),
                seconds=elapsed,
                joules=energy,
                peak_bytes=torch.cuda.max_memory_allocated(),
                trace=trace,
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "competition"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    train(args.arm, args.run_dir)
