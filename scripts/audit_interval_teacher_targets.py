"""Directly normalize complete intervals and verify the implied next-event scores."""

import json
from math import log
from time import monotonic

import numpy as np
import torch
from scripts.interval_teacher_corpus import WORK, PUBLIC, load_rows
from scripts.train_causal_glyph_writer import record, sha, in_cells
from ilm.visual_lm.interval_teacher import interval_targets


def audit():
    torch.set_num_threads(2)
    rows = load_rows()
    receipt = json.loads((WORK / "teacher.json").read_text())
    archive = WORK / "teacher.npz"
    assert sha(archive) == receipt["sha256"]
    stats = dict(
        examples=0,
        mode_matches=0,
        multiple_occurrences=0,
        initial_mass=0.0,
        terminal_mass=0.0,
        whole_mass=0.0,
        sampled_prefix_checks=0,
        max_log_probability_error=0.0,
    )
    began = monotonic()
    with np.load(archive) as data:
        raw = data["logits"]
        for k, (idx, left, right) in enumerate(
            zip(data["indices"], data["offsets"][:-1], data["offsets"][1:], strict=True)
        ):
            r = rows[idx]
            p = in_cells(r.p)
            y = in_cells(r.y)
            T = len(y)
            # Exact double-precision interval masses, independent of the training routine.
            a, b = raw[left:right].astype(np.float64).T
            w = np.exp(a - a.max())[:, None] * np.lib.stride_tricks.sliding_window_view(
                np.pad(np.exp(b - b.max()), (0, 47)), 48
            )
            start, delta = divmod(int(w.argmax()), 48)
            stats["mode_matches"] += int(y == p[start : start + delta + 1])
            stats["examples"] += 1
            js = [i for i, v in enumerate(p) if v == y[0]]
            initial = w[js].sum() / w.sum()
            js = [j for j in js if p[j : j + T] == y]
            assert js
            exact = w[js, T - 1].sum()
            terminal = exact / w[js, T - 1 :].sum()
            whole = exact / w.sum()
            stats["multiple_occurrences"] += int(len(js) > 1)
            stats["initial_mass"] += initial
            stats["whole_mass"] += whole
            stats["terminal_mass"] += terminal
            if k % 37 == 0:
                labels = {v: j + 1 for j, v in enumerate(dict.fromkeys(p))}
                ids = torch.tensor([[labels[v] for v in p]])
                copies, stop = interval_targets(
                    ids,
                    ids != 0,
                    torch.tensor([[*[labels[v] for v in y], 0]]),
                    torch.tensor([T]),
                    torch.tensor(raw[None, left:right].astype("f")),
                )
                logp = log(float(stop[0, T]))
                for t, cell in enumerate(y):
                    possible = [j for j in range(len(p) - t) if p[j + t] == cell]
                    logp += log(float(copies[0, t, possible].sum()))
                gap = abs(logp - log(whole))
                stats["max_log_probability_error"] = max(
                    stats["max_log_probability_error"], gap
                )
                if gap > 2e-4:
                    raise ValueError(
                        "Prefix factorization diverges from full interval enumeration"
                    )
                stats["sampled_prefix_checks"] += 1
    for key in ("initial_mass", "whole_mass", "terminal_mass"):
        stats[key] /= stats["examples"]
    stats["seconds"] = monotonic() - began
    PUBLIC.mkdir(exist_ok=True, parents=True)
    record(PUBLIC / "teacher_targets.json", stats)
    print(stats, flush=True)


if __name__ == "__main__":
    audit()
