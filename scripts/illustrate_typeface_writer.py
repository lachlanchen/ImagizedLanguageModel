"""Use frozen replies and measurements for Figure 102."""

import json
import shutil
from scripts.audit_typeface_writer import ARMS
from scripts.train_typeface_writer import PUBLIC
from scripts.train_causal_glyph_writer import record


def illustrate():
    result = json.loads((PUBLIC / "result.json").read_text())
    scores = result["scores"]
    series = [
        [
            round(100 * scores[a][f]["complete"] / scores[a][f]["boards"], 1)
            for f in ("regular", "bold", "kai", "hana", "trial_regular", "trial_hana")
        ]
        for a in ARMS
    ]
    table = [["Exact answer", "Cases", "Parent", "Map only", "Replay", "Map + replay"]]
    for title, groups in [
        ("HanaMin trial", ["trial_hana"]),
        ("Earlier reading", ["cmrc", "drcd"]),
        ("Historical forms", ["historical"]),
        ("Rephrased forms", ["historical_rephrased"]),
        ("Required absent glyph", ["open_absent_content_ink"]),
    ]:
        table.append(
            [
                title,
                str(sum(scores["parent"][k]["n"] for k in groups)),
                *[str(sum(scores[a][k]["correct"] for k in groups)) for a in ARMS],
            ]
        )
    tag = next(k for k in ("gain", "success", "error") if k in result["examples"])
    for name, src in [
        ("example_question", tag + "_question"),
        ("example_answer", tag + "_mapped"),
        ("error_reference", "error_reference"),
        ("error_answer", "error_mapped"),
    ]:
        shutil.copyfile(PUBLIC / "examples" / (src + ".png"), PUBLIC / (name + ".png"))
    record(PUBLIC / "figure.json", dict(series=series, table=table, example=tag))


if __name__ == "__main__":
    illustrate()
