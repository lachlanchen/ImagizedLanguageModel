"""Recount whole-image edit distances and publish the masked reading of paired students."""

from collections import Counter
import shutil

import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.evaluate_paired_student import (
    EVAL,
    OUT,
    ROOT,
    load,
    read_streams,
    sha,
    write_new,
)
from scripts.report_clause_support import strip
from scripts.report_pictured_revision import unpack_grades


def main():
    roster = load(EVAL / "roster.json")
    scores = load(EVAL / "scores.json")
    distances = load(EVAL / "distances.json")
    generation = load(EVAL / "generation.json")
    teacher = load(OUT / "target_teacher/receipt.json")
    names = (
        "source",
        "composer",
        "parent",
        "archive",
        "teacher",
        "exchange",
        "reference",
        "teacher_target",
    )
    pixels = {name: read_streams(name) for name in names}
    streams = {
        name: [tuple(map(bytes, row)) for row in rows] for name, rows in pixels.items()
    }
    for name, digest in generation["files"].items():
        assert sha(EVAL / name) == digest
    assert sha(EVAL / "teacher_target.npz") == teacher["packed"]
    for ref, data in scores["scores"].items():
        eligible = (
            np.ones(len(roster), bool)
            if ref == "reference"
            else np.array(teacher["complete_renderable"])
        )
        for name in names:
            d = [
                Levenshtein.distance(x, y)
                for x, y in zip(streams[name], streams[ref], strict=True)
            ]
            assert (
                d == distances[ref][name]
                and np.array(d)[eligible].sum() == data["arms"][name]["edits"]
            )
    grades = unpack_grades(EVAL)
    accepted = {
        name: sum(r[name][0] == "F" for r in grades.values())
        for name in grades[next(iter(grades))]
    }
    changes = Counter((row["archive"][0], row["teacher"][0]) for row in grades.values())
    out = ROOT / "references/evidence/paired_student_20261119"
    out.mkdir(exist_ok=False)
    for folder, nameset in (
        (OUT, ("fit.json",)),
        (
            EVAL,
            (
                "scores.json",
                "grades.tsv",
                "mapping.json",
                "review.json",
                "generation.json",
                "roster.json",
                "distances.json",
            ),
        ),
        (OUT / "target_teacher", ("receipt.json",)),
    ):
        for filename in nameset:
            shutil.copyfile(folder / filename, out / filename)
    report = dict(
        accepted=accepted,
        comparisons={a + b: n for (a, b), n in changes.items()},
        chapters=generation["chapters"],
        groups=len(grades),
    )
    write_new(out / "reading.json", report)
    methods = {
        "source": "Copy source",
        "composer": "Phrase composer",
        "parent": "Parent writer",
        "archive": "Archive replay",
        "teacher": "Teacher + archive",
        "exchange": "Exchanged source",
    }
    table = [["Method", "Archive edits", "Teacher edits", "Learned stops"]]
    for name, label in methods.items():
        counts = [
            scores["scores"][ref]["arms"][name]["edits"]
            for ref in ("reference", "teacher_target")
        ]
        stopped = scores["stops"].get("teacher" if name == "exchange" else name)
        table.append(
            [
                label,
                *[f"{x:,}" for x in counts],
                str(stopped) if stopped is not None else "—",
            ]
        )
    # Display the first compact case improved by supplementary training and
    # the first compact case worsened, always relative to archive-only replay.
    cost = distances["reference"]
    samples = {}
    arms = ("source", "archive", "teacher", "reference")
    for key, sign in (("gain", 1), ("loss", -1)):
        eligible = [
            i
            for i in range(len(roster))
            if sign * (cost["archive"][i] - cost["teacher"][i]) > 0
            and all(0 < len(pixels[n][i]) <= 15 for n in arms)
        ]
        samples[key] = eligible[0]
    widths = {
        key + "_" + name: strip(out, key + "_" + name, pixels[name][i])
        for key, i in samples.items()
        for name in arms
    }
    bars = [
        "parent",
        "archive",
        "teacher",
        "exchange",
        "composer",
        "teacher_target",
        "reference",
    ]
    write_new(
        out / "figure.json",
        dict(
            samples=samples,
            widths=widths,
            table=table,
            series=[[accepted[k] for k in bars]],
            quantitative=scores["scores"]["teacher_target"]["rows"],
            chapters=generation["chapters"],
        ),
    )
    print(report, samples)


if __name__ == "__main__":
    main()
