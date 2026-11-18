"""Attach the probability diagnosis and autonomous decoding outcomes to Figure 61."""

import json
import shutil
import numpy as np
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_pooled_glyphs import (
    WORK,
    EVAL,
    ROOT,
    read_streams,
    load,
    write_new,
)
from scripts.report_clause_support import seq, strip
from scripts.report_pictured_revision import unpack_grades


def main():
    directory = ROOT / "references/evidence/paired_student_20261119"
    figure = load(directory / "figure.json")
    assert "pooled" not in figure
    results = load(WORK / "scores.json")
    ratings = unpack_grades(WORK)
    faithful = {
        key: sum(row[key][0] == "F" for row in ratings.values())
        for key in next(iter(ratings.values()))
    }
    mask = load(directory / "receipt.json")["complete_renderable"]
    with np.load(WORK / "pooled.npz") as f:
        packed, offsets = f["packed"], f["offsets"]
    output = [tuple(map(bytes, packed[a:b])) for a, b in zip(offsets, offsets[1:])]
    for ref in ("reference", "teacher_target"):
        targets = [tuple(map(bytes, row)) for row in read_streams(ref)]
        d = sum(
            Levenshtein.distance(p, t)
            for i, (p, t) in enumerate(zip(output, targets, strict=True))
            if ref == "reference" or mask[i]
        )
        assert d == results["scores"][ref]["edits"]
    for stem in ("review.json", "mapping.json", "grades.tsv", "scores.json"):
        shutil.copyfile(WORK / stem, directory / ("pooled_" + stem))
    write_new(directory / "pooled_reading.json", faithful)
    for side, i in figure["samples"].items():
        key = side + "_pooled"
        figure["widths"][key] = strip(directory, key, seq(WORK, "pooled", i))
    figure["pooled"] = faithful
    figure["table"].append(
        [
            "Pooled glyph probability",
            *[
                f"{results['scores'][name]['edits']:,}"
                for name in ("reference", "teacher_target")
            ],
            str(results["stopped"]),
        ]
    )
    figure["series"][0].insert(3, faithful["pooled"])
    path = directory / "figure.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(figure, indent=2) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
