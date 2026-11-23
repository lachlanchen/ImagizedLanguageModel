"""Bind Figure 103 to independently counted image replies and complete pairs."""

import json
import shutil

from scripts.train_causal_glyph_writer import record
from scripts.train_stroke_scale import PUBLIC


def illustrate():
    scores = json.loads((PUBLIC / "result.json").read_text())["scores"]
    routed = json.loads((PUBLIC / "routed.json").read_text())["scores"]
    arms = [scores["translation"], scores["stroke_scale"], routed["routed"]]
    series = [
        [
            round(100 * arm[g]["complete"] / arm[g]["boards"], 2)
            for g in (
                "development_hana",
                "development_fandol",
                "trial_hana",
                "trial_fandol",
                "trial_regular",
            )
        ]
        for arm in arms
    ]
    table = [
        [
            "XQuAD question",
            "Passages",
            "Original exact",
            "Routed exact",
            "Original pairs",
            "Routed pairs",
        ]
    ]
    for title, k in (
        ("Chinese · regular", "regular_zh"),
        ("Chinese · HanaMin", "hana_zh"),
        ("English · HanaMin", "hana_en"),
        ("Chinese · FandolSong", "fandol_zh"),
        ("English · FandolSong", "fandol_en"),
    ):
        a, b = [routed[arm]["xquad_" + k] for arm in ("translation", "routed")]
        table.append(
            [
                title,
                str(a["boards"]),
                str(a["exact"]),
                str(b["exact"]),
                str(a["complete"]),
                str(b["complete"]),
            ]
        )
    for dst, src in (
        ("first_question", "gain_question"),
        ("first_answer", "gain_answer"),
        ("second_question", "error_question"),
        ("second_answer", "error_answer"),
        ("second_reference", "error_reference"),
    ):
        shutil.copyfile(PUBLIC / "examples" / (src + ".png"), PUBLIC / (dst + ".png"))
    record(PUBLIC / "figure.json", dict(series=series, table=table))


if __name__ == "__main__":
    illustrate()
