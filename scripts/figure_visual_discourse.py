"""Build the open-writing figure from generated cells and measured histograms."""

import numpy as np
from PIL import Image
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_visual_discourse import MODEL, PUBLIC, read, dump
from scripts.prepare_visual_cursor import PUBLIC as QA


def figure_data():
    tasks = ["book-en", "book-zh", "instruction-en", "instruction-zh"]
    result = read(PUBLIC / "ink-pair-evaluation.json")["open"]
    initial = read(PUBLIC / "initial-open.json")
    series = [
        [round(line, 2) for line in group]
        for group in (
            [initial[k]["bits_per_glyph"] for k in tasks],
            [result[k]["bits_per_glyph"] for k in tasks],
            [result[k]["wrong_prompt_bits_per_glyph"] for k in tasks],
        )
    ]
    table = [
        [
            "Complete written answers",
            "English dialogue",
            "Simplified Chinese",
            "Traditional Chinese",
        ]
    ]
    for arm, name in (
        ("base", "Original writer"),
        ("independent", "Independent source position"),
        ("persistent", "Persistent source position"),
    ):
        scores = read(QA / (arm + "-evaluation.json"))
        table.append(
            [name]
            + [
                str(scores[k]["exact"]) + " / " + str(scores[k]["n"])
                for k in ("coqa", "cmrc", "drcd")
            ]
        )
    return dict(series=series, table=table)


def typeset_cells(file, count, out, wrap):
    tiles = read_image(file, count)
    lines = []
    line = []
    advance = 0
    for cell in tiles:
        ink = np.unpackbits(cell).reshape(32, 32)
        columns = np.where(ink.any(0))[0]
        glyph = ink[:, columns[0] : columns[-1] + 1] if len(columns) else ink[:, :8]
        if line and advance + glyph.shape[1] + 2 > wrap:
            lines.append(np.concatenate(line, axis=1))
            line = []
            advance = 0
        line.extend([glyph, np.zeros((32, 2), np.uint8)])
        advance += glyph.shape[1] + 2
    if line:
        lines.append(np.concatenate(line, axis=1))
    width = max(line.shape[1] for line in lines)
    canvas = np.concatenate(
        [np.pad(line, ((0, 0), (0, width - line.shape[1]))) for line in lines], axis=0
    )
    Image.fromarray(255 * (1 - canvas)).save(out)


def main():
    for lang in ("en", "zh"):
        f = MODEL / "examples" / ("instruction-" + lang)
        r = read(f / "readout.json")
        for kind, field in (("input", "cells"), ("answer", "written")):
            typeset_cells(
                f / (kind + ".png"),
                r[field],
                PUBLIC / (lang + "-" + kind + ".png"),
                740,
            )
    dump(PUBLIC / "figure.json", figure_data())


if __name__ == "__main__":
    main()
