"""Recount the matched replay control and attach its measured values to Figure 72."""

import numpy as np
from scripts.evaluate_independent_entry_reader import views
from scripts.evaluate_paired_student import read_streams
from scripts.train_independent_entry_reader import OUT, corpus, load, sha, write_json


def finalize():
    primary = OUT / "primary_figure.json"
    if primary.exists():
        raise FileExistsError("Figure already finalized")
    base = load(OUT / "figure.json")
    replay = load(OUT / "replay_evaluation.json")
    evaluated = load(OUT / "evaluation.json")["conditions"]
    ref = corpus("development")["answers"]
    exact = {}
    source_rows = {}
    for view, qs, ids, group in views():
        r = replay[view]
        archive = OUT / "outputs" / (view + "_replay.npz")
        assert sha(archive) == r["stream_sha"]
        raw = read_streams(view + "_replay", OUT / "outputs")
        hits = np.asarray(
            [
                s and p.tobytes() == ref[i]
                for s, p, i in zip(r["stops"], raw, ids, strict=True)
            ]
        )
        assert (
            np.array_equal(np.flatnonzero(hits), r["rows"])
            and hits.reshape(-1, group).all(-1).sum() == r["complete"]
        )
        unique = {}
        for i, (q, t) in enumerate(zip(qs, ids, strict=True)):
            unique.setdefault((t // 8, q, ref[t]), i)
        exact[view] = int(hits[list(unique.values())].sum())
        source_rows[view] = len(hits)
    table = [base["table"][0][:3] + ["Replay"] + base["table"][0][3:]]
    for row, view in zip(
        base["table"][1:],
        ("independent", "before", "correction", "reference", "simple"),
        strict=True,
    ):
        table.append(row[:3] + [f"{exact[view]:,}"] + row[3:])
    # Resample whole families, separately within each sampled stage combination.
    import json
    from scripts.prepare_pictured_entry_answers import DEFAULT

    meta = load(DEFAULT / "development/index.json")
    elicited = [
        json.loads(l) for l in (OUT / "questions.jsonl").read_text().splitlines()
    ]
    groups = {}
    for g in range(24):
        stages = tuple(
            sorted({meta[elicited[i]["row"]]["stage"] for i in range(g * 4, g * 4 + 4)})
        )
        groups.setdefault(stages, []).append(g)
    assert len(groups) == 6 and all(len(v) == 4 for v in groups.values())
    shared = evaluated["independent"]["shared"]
    r = replay["independent"]
    diff = np.zeros(96, np.int8)
    diff[shared["rows"]] += 1
    diff[r["rows"]] -= 1
    diff = diff.reshape(24, 4).sum(1)
    rng = np.random.default_rng(77412)
    sample = np.concatenate(
        [rng.choice(g, (10000, len(g))) for g in groups.values()], axis=1
    )
    ci = np.quantile(diff[sample].sum(1) / 96, [0.025, 0.975]).tolist()
    write_json(
        OUT / "replay_recount.json",
        dict(
            exact_distinct=exact,
            stream_counts=source_rows,
            group_bootstrap_interval=ci,
            shared_minus_replay=int(diff.sum()),
            replicates=10000,
        ),
    )
    write_json(primary, base)
    base["table"] = table
    base["series"].append([round(x, 5) for x in load(OUT / "replay_fit.json")["loss"]])
    write_json(OUT / "figure.json", base)
    print(ci)


if __name__ == "__main__":
    finalize()
