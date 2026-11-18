"""Verify the full-corpus likelihood comparison and export the underlying figure data."""

import shutil
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_visible_image_likelihood import (
    EVAL,
    OUT,
    ROOT,
    load,
    sha,
    write_new,
)
from scripts.evaluate_paired_student import read_streams
from scripts.evaluate_visible_speech import WORK
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.report_pictured_revision import unpack_grades
from scripts.report_clause_support import strip


def main():
    d = ROOT / "references/evidence/visible_likelihood_20261119"
    summary = load(EVAL / "scores.json")
    rating = unpack_grades(EVAL)
    rows = load(EVAL / "roster.json")
    fit = load(OUT / "fit.json")
    faithful = {
        k: sum(v[k][0] == "F" for v in rating.values())
        for k in next(iter(rating.values()))
    }
    for filename, digest in summary["files"].items():
        assert sha(EVAL / filename) == digest
    pixels = {k: read_streams(k, EVAL) for k in summary["scores"]}
    dist = load(EVAL / "distances.json")
    for arm, v in pixels.items():
        assert [
            Levenshtein.distance(tuple(map(bytes, p)), tuple(map(bytes, t)))
            for p, t in zip(v, pixels["reference"])
        ] == dist[arm]
    # Bind every readable character in the masked packet to the original ink.
    raster = Rasterizer()
    unknown = 0
    for r, key in zip(load(EVAL / "review.json"), load(EVAL / "mapping.json")):
        i = key["row"]
        for name, j in key["options"].items():
            txt = r["choices"][j]
            assert len(txt) == len(pixels[name][i])
            for c, cell in zip(txt, pixels[name][i]):
                if c == "▧":
                    unknown += 1
                else:
                    assert raster.tile(c) == cell.tobytes()
    alt = load(WORK / "review.json")
    keys = load(WORK / "mapping.json")
    samples = load(WORK / "samples.json")
    ratings = {}
    for i, (line, key) in enumerate(
        zip((WORK / "grades.tsv").read_text().splitlines(), keys), 1
    ):
        n, marks, note = line.split("\t")
        assert int(n) == i
        m = marks.split(",")
        assert len(m) == len(alt[i - 1]["choices"]) and all(
            len(v) == 3 and set(v) <= set("01") for v in m
        )
        ratings[i - 1] = {k: m[j] for k, j in key["options"].items()}
    assert len(ratings) == 8
    speech = {
        k: dict(
            speaker=sum(v[k][0] == "1" for v in ratings.values()),
            polarity_and_predicate=sum(v[k][1] == "1" for v in ratings.values()),
            whole=sum(v[k] == "111" for v in ratings.values()),
            both=sum(
                ratings[i][k] == ratings[i + 1][k] == "111" for i in range(0, 8, 2)
            ),
        )
        for k in ["marginal", "composer"]
    }
    d.mkdir(exist_ok=False)
    for parent, prefix, names in [
        (
            EVAL,
            "",
            [
                "scores.json",
                "roster.json",
                "distances.json",
                "review.json",
                "mapping.json",
                "grades.tsv",
            ],
        ),
        (OUT, "", ["fit.json"]),
        (
            WORK,
            "speech_",
            ["samples.json", "review.json", "mapping.json", "grades.tsv"],
        ),
    ]:
        for filename in names:
            shutil.copyfile(parent / filename, d / (prefix + filename))
    exact = [
        i for i, r in enumerate(rows) if dist["marginal"][i] == 0 and dist["source"][i]
    ]
    # First exact gain that changes at least two non-punctuation characters; the
    # other example is the mission naming Wei Ran, selected to show a wrong role.
    success = next(
        i
        for i in exact
        if Levenshtein.distance(
            "".join(c for c in rows[i]["source"] if c.isalnum()),
            "".join(c for c in rows[i]["target"] if c.isalnum()),
        )
        >= 2
    )
    failed = next(
        i for i, r in enumerate(rows) if r["source"] == "秦使魏冉之赵，出兵助燕击齐。"
    )
    widths = {}
    for tag, i in (("exact", success), ("mission", failed)):
        for k in ["source", "marginal", "composer", "reference"]:
            name = tag + "_" + k
            widths[name] = strip(d, name, pixels[k][i])
    for i in (2, 3):
        for name in ("source", "marginal"):
            shutil.copyfile(WORK / f"{i}_{name}.png", d / f"speech_{i}_{name}.png")
    data = dict(
        accepted=faithful,
        speech=speech,
        exact=exact,
        unresolved_masked_cells=unknown,
        speech_training_sources=sum(r["seen_training"] for r in samples),
    )
    write_new(d / "reading.json", data)
    table = [["Method", "Full glyph edits", "Changed exact", "Faithful / 18"]]
    for k, label in [
        ("source", "Unchanged source"),
        ("composer", "Phrase composer"),
        ("parent", "Parent, pooled decoder"),
        ("single", "Single training path"),
        ("marginal", "All matching paths"),
        ("exchange", "Exchanged source"),
    ]:
        s = summary["scores"][k]
        table.append(
            [
                label,
                f"{s['edits']:,}",
                str(s["changed_exact"]),
                str(faithful.get(k, "—")),
            ]
        )
    write_new(
        d / "figure.json",
        dict(
            table=table,
            widths=widths,
            rows=dict(exact=success, mission=failed),
            series=[
                [
                    speech[k][m]
                    for m in ("speaker", "polarity_and_predicate", "whole", "both")
                ]
                for k in ("marginal", "composer")
            ],
            speech=speech,
        ),
    )
    print(data)


if __name__ == "__main__":
    main()
