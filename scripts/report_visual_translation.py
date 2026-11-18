"""Recount frozen readings and export literal ink strips and data for Figure 57."""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scripts.train_aligned_visual_translation import (
    ROOT,
    WORK,
    load,
    read_rows,
    sha,
    tiles,
    write_new,
)

DEFAULT_OUTPUT = WORK / "recount"


def grade(path, label, output):
    grade_lines = (path / "grades.txt").read_text().splitlines()
    blind, key = load(path / "review.json"), load(path / "mapping.json")
    assert len(blind) == len(key) == len(grade_lines) == 24
    counts = defaultdict(Counter)
    for scores, options, row in zip(grade_lines, key, blind):
        assert len(scores) == len(row["choices"]) and set(scores) <= set("FPCX")
        for arm, i in options["options"].items():
            counts[arm][scores[i]] += 1
    for name in ("grades.txt", "review.json", "mapping.json", "summary.json"):
        shutil.copyfile(path / name, output / (label + "_" + name))
    return counts


def main(output):
    output.mkdir(exist_ok=False)
    evaluated, composed = WORK / "evaluation", WORK / "composer"
    rowset = read_rows("heldout")
    groups = {
        label: grade(path, label, output)
        for label, path in [("block", evaluated), ("composer", composed)]
    }
    for src, dst in [
        ("fit/fit.json", "fit.json"),
        ("preparation.json", "preparation.json"),
        ("blind_grades.txt", "alignment_grades.txt"),
        ("blind.json", "alignment_review.json"),
        ("key.json", "alignment_key.json"),
    ]:
        shutil.copyfile(WORK / src, output / dst)
    old = ROOT / "artifacts/first_emission_v1/within_chapter"
    for src in ("blind_review.tsv", "mapping.json", "packet.json", "preparation.json"):
        shutil.copyfile(old / src, output / ("calibration_" + src))
    write_new(output / "readings.json", groups)
    size = WORK / "scale"
    write_new(output / "scale_readings.json", grade(size, "scale", output))
    shutil.copyfile(size / "changed_exact.json", output / "scale_changed_exact.json")
    score = load(evaluated / "summary.json")["arms"]
    table = [
        [
            "First 2,048 passages",
            "Glyph edits",
            "Error (%)",
            "Exact glosses",
            "Accepted / 24",
        ]
    ]
    for key, name in [
        ("copy", "Copy source"),
        ("local", "Local majority"),
        ("y", "Original targets"),
        ("z", "Realigned targets"),
        ("z_reversed", "Reverse context"),
    ]:
        s = score[key]["z"]
        table.append(
            [
                name,
                format(s["edits"], ","),
                format(100 * s["cer"], ".2f"),
                str(s["exact"]),
                str(groups["block"][key]["F"])
                if key in groups["block"]
                else "Unscored",
            ]
        )
    cs = load(composed / "summary.json")["arms"]
    counts = [[cs[k]["edits"] for k in ("copy", "local", "model")]]
    widths = {}

    def strip(name, cells):
        pixels = np.unpackbits(cells, axis=1).reshape(-1, 32, 32)
        Image.fromarray((1 - np.concatenate(pixels, axis=1)) * 255).save(
            output / (name + ".png")
        )
        widths[name] = len(cells)

    bank = tiles()

    def seq(path, num, arm):
        with np.load(path / (arm + ".npz")) as lib:
            first, last = lib["offsets"][num : num + 2]
            if "cells" in lib:
                return lib["cells"][first:last]
            return np.load(path / "bitmap_index.npy")[lib["sequence"][first:last]]

    # First enrolled neural row, first changed exact composer result, first short
    # composer result graded X in the chapter-distinct reading.
    per = load(composed / "per_row.json")
    gained = next(
        i for i, r in enumerate(per) if r["copy"]["z"] and not r["model"]["z"]
    )
    missed = next(
        m["row"]
        for m, g in zip(
            load(composed / "mapping.json"),
            (composed / "grades.txt").read_text().splitlines(),
        )
        if g[m["options"]["model"]] == "X" and len(rowset[2048 + m["row"]]["z"]) <= 16
    )
    for name, path, index, offset, arm in [
        ("neural", evaluated, 0, 0, "z"),
        ("gain", composed, gained, 2048, "model"),
        ("error", composed, missed, 2048, "model"),
    ]:
        r = rowset[offset + index]
        strip(name + "_source", bank[r["x"]])
        strip(name + "_answer", seq(path, index, arm))
        strip(name + "_gloss", bank[r["z"]])
    write_new(
        output / "figure.json",
        dict(
            table=table,
            exchange_counts=counts,
            widths=widths,
            cohort_indices=dict(neural=0, gain=gained, error=missed),
            reviews=groups,
        ),
    )
    write_new(
        output / "streams.json",
        {
            str(p.relative_to(WORK)): sha(p)
            for d in [composed, evaluated]
            for p in sorted(d.iterdir())
            if p.suffix in (".npz", ".npy") or p.name == "per_row.json"
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
