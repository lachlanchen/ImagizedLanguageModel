"""Recount physical distances and frozen meaning grades used in Figure 60."""

import shutil

import numpy as np
from rapidfuzz.distance import Levenshtein
from scripts.report_clause_support import seq, strip
from scripts.train_pictured_revision import ROOT, OUT, load, sha, write_new


def unpack_grades(path):
    data = load(path / "review.json")
    keys = load(path / "mapping.json")
    lines = (path / "grades.tsv").read_text().splitlines()
    assert len(data) == len(keys) == len(lines)
    results = {}
    for i, (r, key, line) in enumerate(zip(data, keys, lines), 1):
        num, grades, flags, note = line.split("\t")
        flags = flags.split(",")
        assert (
            int(num) == i
            and len(r["choices"]) == len(grades) == len(flags)
            and set(grades) <= set("FPCX")
        )
        assert all(set(v) <= set("-gint") for v in flags)
        results[key["row"]] = {
            k: (grades[j], flags[j]) for k, j in key["options"].items()
        }
    return results


def main():
    out = ROOT / "references/evidence/pictured_revision_20261118"
    out.mkdir(exist_ok=False)
    e = OUT / "evaluation"
    q = OUT / "qwen"
    v = OUT / "vision_teacher"
    stats = load(e / "summary.json")
    records = load(e / "rows.json")
    packets = load(e / "packets.json")
    for name, h in stats["files"].items():
        assert sha(e / name) == h

    # Recount each reported edit from the actual packed image streams.
    def streams(name):
        with np.load(e / (name + ".npz")) as raw:
            a = raw["packed"]
            o = raw["offsets"]
            return [tuple(map(bytes, a[i:j])) for i, j in zip(o, o[1:])]

    reference = streams("reference")
    for arm, res in stats["arms"].items():
        distances = [
            Levenshtein.distance(x, y) for x, y in zip(streams(arm), reference)
        ]
        assert (
            distances == [r["dist"][arm] for r in records]
            and sum(distances) == res["edits"]
        )
    ratings = unpack_grades(q)
    teacher = unpack_grades(v)
    arms = ["draft", "revised", "exchanged", "reference", "qwen"]
    readings = {}
    for name, ids in packets.items():

        def gained(i, k):
            return ratings[i][k][0] == "F"

        readings[name] = dict(
            accepted={k: sum(gained(i, k) for i in ids) for k in arms},
            accepted_draft_lost=[
                i for i in ids if gained(i, "draft") and not gained(i, "revised")
            ],
            accepted_revision_gained=[
                i for i in ids if gained(i, "revised") and not gained(i, "draft")
            ],
            newly_flagged={
                flag: [
                    i
                    for i in ids
                    if flag in ratings[i]["revised"][1]
                    and flag not in ratings[i]["draft"][1]
                ]
                for flag in "gint"
            },
        )
    readings["vision"] = {
        k: sum(scores[k][0] == "F" for scores in teacher.values())
        for k in ("typed", "image")
    }
    for path, prefix, files in [
        (OUT, "", ["fit.json"]),
        (e, "", ["summary.json", "packets.json", "rows.json"]),
        (
            q,
            "qwen_",
            [
                "receipt.json",
                "grades.tsv",
                "mapping.json",
                "review.json",
                "translations.json",
            ],
        ),
        (
            v,
            "vision_",
            [
                "receipt.json",
                "grades.tsv",
                "mapping.json",
                "review.json",
                "responses.json",
            ],
        ),
    ]:
        for file in files:
            shutil.copyfile(path / file, out / (prefix + file))
    write_new(out / "readings.json", readings)
    table = [
        [
            "Complete stream",
            "Glyph edits",
            "Error (%)",
            "Exact glosses",
            "Changed exact",
        ]
    ]
    for arm, label in [
        ("source", "Copy classical source"),
        ("draft", "First draft"),
        ("revised", "Revised, true source"),
        ("exchanged", "Revised, other source"),
    ]:
        res = stats["arms"][arm]
        table.append(
            [
                label,
                f'{res["edits"]:,}',
                f'{res["cer"]*100:.2f}',
                str(res["exact"]),
                str(res["changed_exact"]),
            ]
        )
    rows = dict(
        gain=next(
            i
            for i, r in enumerate(records)
            if r["dist"]["draft"] > 0 and r["dist"]["revised"] == 0
        ),
        loss=readings["changed"]["accepted_draft_lost"][0],
    )
    widths = {}
    for label, i in rows.items():
        for arm in ["source", "draft", "revised", "reference"]:
            n = label + "_" + arm
            widths[n] = strip(out, n, seq(e, arm, i))
    series = [
        [
            readings[cohort]["accepted"][k]
            for k in ["draft", "revised", "qwen", "reference"]
        ]
        for cohort in ["all", "changed"]
    ]
    write_new(
        out / "figure.json",
        dict(
            table=table,
            series=series,
            widths=widths,
            rows=rows,
            vision=readings["vision"],
            ocr=load(v / "receipt.json")["transcription_exact"],
        ),
    )


if __name__ == "__main__":
    main()
