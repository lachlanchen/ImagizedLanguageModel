"""Build the panel inputs from immutable full-stream output and the paired scores."""

import json
import shutil
from collections import defaultdict
from scipy.stats import binomtest

from scripts.prepare_participant_margin import OUT, PUBLIC
from scripts.prepare_visual_facts import Face, FONTS, write_json
from scripts.evaluate_participant_margin import save_png
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.train_visual_facts import rows


def main():
    res = json.loads((PUBLIC / "results.json").read_text())["arms"]
    res.update(json.loads((PUBLIC / "ablation.json").read_text())["arms"])
    arms = ("previous", "permuted", "matched", "likelihood")
    cohorts = ("new_source", "clause_order")
    series = [
        [round(100 * res[a][c]["all_both"] / res[a][c]["boards"], 1) for c in cohorts]
        for a in arms
    ]
    table = [
        [
            "Question set",
            "Inputs",
            "Previous",
            "Random labels",
            "With margin",
            "Likelihood only",
        ]
    ]
    for title, groups in [
        ("Human questions", ("cmrc", "drcd")),
        ("Programme notes", ("notes",)),
        ("DuIE passages", ("duie",)),
    ]:
        table.append(
            [
                title,
                str(sum(res["matched"][c]["inputs"] for c in groups)),
                *[str(sum(res[a][c]["both"] for c in groups)) for a in arms],
            ]
        )
    write_json(PUBLIC / "figure.json", dict(series=series, table=table))
    led = {a: json.loads((OUT / (a + "_ledger.json")).read_text()) for a in arms}
    analysis = {}
    for group in cohorts:

        def full(arm):
            out = defaultdict(list)
            for x in led[arm]:
                if x["group"] == group:
                    out[x["board"]].append(x["both"])
            return {k: all(vals) for k, vals in out.items()}

        p, m = full("previous"), full("matched")
        win = sum(m[k] and not p[k] for k in p)
        lose = sum(p[k] and not m[k] for k in p)
        analysis[group] = dict(
            paired_gains=win,
            paired_losses=lose,
            two_sided_exact_mcnemar_p=binomtest(win, win + lose).pvalue,
        )
    write_json(PUBLIC / "paired.json", analysis)
    cases = rows("held")
    i = next(
        j
        for j in range(48)
        if led["previous"][j]["both"] and not led["matched"][j]["both"]
    )
    face = Face((FONTS[0], 2))
    x = cases[i]
    save_png(PUBLIC / "human_question.png", face(x["q"]))
    from scripts.train_fact_realizer import target

    for order, j in [("object", 1), ("subject", 0)]:
        values = {
            a: load_streams(OUT / f"{a}_{order}.npz") for a in ("previous", "matched")
        }
        y, st = values["matched"]
        if not st[i] or y[i] != face(target(x, j)):
            for a, (streams, stops) in values.items():
                save_png(PUBLIC / f"human_{a}.png", streams[i])
            write_json(
                PUBLIC / "human_example.json",
                dict(index=i, question=x["q"], reference=target(x, j), order=order),
            )
            break
    for name, newname in [
        ("matched_2179_page", "song_page"),
        ("matched_2179_question", "song_question"),
        ("previous_2179_answer", "song_previous"),
        ("matched_2179_answer", "song_matched"),
    ]:
        shutil.copyfile(
            PUBLIC / "examples" / (name + ".png"), PUBLIC / (newname + ".png")
        )


if __name__ == "__main__":
    main()
