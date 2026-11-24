"""Reproduce the source, actual answer strips, and chart values in Figure 126."""

import torch
import numpy as np
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.answer_questioned_event import components, load_head, answer, RELEASE
from scripts.train_typeface_writer import FACES, Face
from scripts.prepare_questioned_events import PUB, read, dump


def main():
    groups = read(PUB / "description_evaluation/result.json")["groups"]
    a = groups["kyoto_literal"]
    series = [
        [a[arm][key] for key in ("binding", "bound_written")]
        for arm in ("frozen", "exchanged", "adjacent", "aligned")
    ]
    table = [
        [
            "Questions",
            "Pages",
            "Literal training",
            "Definition training",
            "Exchanged definitions",
        ]
    ]
    for name, k in [
        ("Action names", "kyoto_literal"),
        ("Dictionary definitions", "kyoto_webster"),
        ("Natural questions", "kyoto_natural"),
        ("All questions unseen", "kyoto_natural_unseen"),
        ("Pu Songling, action names", "literature_literal"),
    ]:
        g = groups[k]
        table.append(
            [
                name,
                str(g["aligned"]["n"]),
                *[
                    str(g[arm]["bound_written"])
                    for arm in ("aligned", "word_aligned", "word_exchanged")
                ],
            ]
        )
    dump(PUB / "figure.json", dict(series=series, table=table))
    torch.set_num_threads(2)
    parts = components("cpu")
    network = load_head(RELEASE / "aligned", "cpu")
    face = Face(FACES[0])
    letters = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}

    def draw(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128).copy()

    line = draw("舉杯邀明月")
    write_strip(line, PUB / "moon-page.png")
    for term, expected in [("raise", "cup"), ("invite", "moon")]:
        ink, report = answer(
            parts, network, line, draw(f"What is it that they {term} here?")
        )
        assert (
            report["stopped"] and "".join(letters[x.tobytes()] for x in ink) == expected
        )
        write_strip(ink, PUB / ("moon-" + term + ".png"))


if __name__ == "__main__":
    main()
