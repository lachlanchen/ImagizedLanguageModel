"""Decompose whole-surface preference and test which fork decisions survive free writing."""

from collections import Counter
from time import monotonic
import numpy as np
import torch
from scripts.prepare_teacher_translation import ROOT, BASE, WORK, PUB, read, dump, sha
from scripts.evaluate_teacher_translation import (
    pack,
    stopped,
    MODES,
    restore,
    material,
    assessment,
)
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from ilm.visual_lm.portable_writer import save_grid


def load(label):
    if label in MODES:
        return restore(MODES[label], "cuda")
    arm = label.split("-")[0]
    ck = torch.load(WORK / (arm + "-resume.pt"), weights_only=True, map_location="cpu")
    if (
        len(ck["history"]) != 12
        or ck["binding"]
        != read(ROOT / "models/teacher_translation" / arm / "fit.json")["binding"]
    ):
        raise ValueError("Final pass recovery state differs")
    net = restore(MODES[arm], "cpu")
    net.load_state_dict(ck["net"])
    return net.cuda().eval()


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


@torch.inference_mode()
def measure(label, rows, optical):
    net = load(label)
    f, inv, local = optical
    records = [None] * len(rows)
    for t in (0, 1):
        indices = [i for i, r in enumerate(rows) if r["t"] == t]
        for j in range(0, len(indices), 32):
            ix = indices[j : j + 32]
            rr = [rows[i] for i in ix]
            x = pack([r["input"] for r in rr], "cuda")
            s = net.generate(f, x, inv[t], t).cpu()
            teacher = [r["teacher_cells"] + [1] for r in rr]
            corpus = [r["target"] + [1] for r in rr]
            width = max(map(len, teacher + corpus))

            def score(seqs):
                # Equal batch dimensions keep both paths on the same kernels.
                y = torch.zeros(len(ix), width, device="cuda", dtype=torch.long)
                for b, seq in enumerate(seqs):
                    y[b, : len(seq)] = torch.tensor(seq, device="cuda")
                raw = net(f, x, y, inv[t], t).float().log_softmax(-1)
                chosen = raw.gather(-1, local[t][y].clamp_min(0).unsqueeze(-1)).squeeze(
                    -1
                )
                return chosen.mul(1e6).round().long().cpu().numpy()

            is_selected = label in MODES
            if is_selected:
                tc, cc = score(teacher), score(corpus)
            for b, i in enumerate(ix):
                seq, done = stopped(s[b])
                free = seq + ([1] if done else [])
                a, c = teacher[b], corpus[b]
                q = first_difference(a, c)
                r = dict(
                    free=seq, stop=done, teacher_exact=free == a, corpus_exact=free == c
                )
                if q is not None and is_selected:
                    sa, sc = tc[b, : len(a)], cc[b, : len(c)]
                    prefix = int((sa[:q] - sc[:q]).sum())
                    fork = int(sa[q] - sc[q])
                    tail = int(sa[q + 1 :].sum() - sc[q + 1 :].sum())
                    whole = int(sa.sum() - sc.sum())
                    assert whole == prefix + fork + tail
                    r.update(
                        fork=q,
                        prefix_residual=prefix,
                        routing=fork,
                        tail=tail,
                        odds=whole,
                        reached=len(free) > q and free[:q] == a[:q],
                        teacher_branch=free[: q + 1] == a[: q + 1],
                        corpus_branch=free[: q + 1] == c[: q + 1],
                    )
                records[i] = r
    return records


def main():
    configure_cuda_cli()
    selected = read(WORK / "selected.json")
    held = [dict(r, teacher_cells=r["target"]) for r in assessment()[:96]]
    full = {}
    summary = {}
    with gpu_owner():
        optical = material("cuda")
        for label in (*MODES, "corpus-final", "distilled-final"):
            rows = selected if label in MODES else selected + held
            full[label] = rr = measure(label, rows, optical)
            n = len(selected)
            trained = rr[:n]
            agg = {
                k: sum(r[k] for r in trained)
                for k in ("teacher_exact", "corpus_exact", "stop")
            }
            fork = [r for r in rr if "fork" in r]
            if fork:
                agg.update(
                    {
                        k: sum(r[k] for r in fork)
                        for k in ("reached", "teacher_branch", "corpus_branch")
                    }
                )
                agg["mean_log_odds_nats"] = {
                    k: np.mean([r[k] for r in fork]) / 1e6
                    for k in ("odds", "routing", "tail")
                }
                agg["max_abs_prefix_residual_micronats"] = max(
                    abs(r["prefix_residual"]) for r in fork
                )
            if len(rr) > n:
                agg["held_exact"] = sum(r["corpus_exact"] for r in rr[n:])
            summary[label] = agg
    pairs = [
        (a, b)
        for a, b in zip(full["distilled"], full["corpus"], strict=True)
        if "fork" in a
    ]
    errors = [
        (a["odds"] - b["odds"]) - (a["routing"] - b["routing"] + a["tail"] - b["tail"])
        for a, b in pairs
    ]
    out = WORK / "fork-scores.json"
    dump(out, full)
    dump(
        PUB / "forks.json",
        dict(
            rows=len(selected),
            forks=len(pairs),
            models=summary,
            max_intervention_identity_residual_micro_nats=max(map(abs, errors)),
            scores=sha(out),
            protocol=sha(ROOT / "references/teacher_surface_fork_protocol.md"),
            program=sha(ROOT / "scripts/measure_translation_forks.py"),
            proof=sha(ROOT / "formal/span_list/TeacherSurfaceFork.lean"),
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
