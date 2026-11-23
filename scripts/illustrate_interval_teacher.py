"""Prepare editable Figure 101 values from the physical score archives."""

from pathlib import Path
import json
import shutil
from scripts.interval_teacher_corpus import PUBLIC
from scripts.train_causal_glyph_writer import record


def illustrate():
    result = json.loads((PUBLIC / "result.json").read_text())
    bench = json.loads((PUBLIC / "latency.json").read_text())
    scores = {a: v["scores"] for a, v in result["arms"].items()}
    labels = ["parent", "replay", "aligned", "distilled", "teacher", "qwen"]
    fresh = [scores[a]["fresh"] if a != "qwen" else bench["qwen"] for a in labels]
    series = [
        [round(100 * v["correct"] / v["n"], 1) for v in fresh],
        [round(100 * v["complete"] / v["boards"], 1) for v in fresh],
    ]
    table = [
        ["Output criterion", "Cases", "Parent", "Replay", "Alignment", "Distillation"]
    ]
    for title, keys, metric in [
        ("Human reading", ["cmrc", "drcd"], "correct"),
        ("Attested forms", ["historical"], "correct"),
        ("Rephrased forms", ["historical_rephrased"], "correct"),
        ("Routed dates", ["dual_learned"], "complete"),
        ("Absent content glyph", ["open_absent_content_ink"], "correct"),
    ]:
        denominator = "n" if metric == "correct" else "boards"
        table.append(
            [
                title,
                str(sum(scores["parent"][key][denominator] for key in keys)),
                *[str(sum(scores[a][key][metric] for key in keys)) for a in labels[:4]],
            ]
        )
    names = [
        ("gain", "question"),
        ("gain", "distilled"),
        ("error", "reference"),
        ("error", "distilled"),
    ]
    for label, kind in names:
        path = PUBLIC / "examples" / f"{label}_{kind}.png"
        if not path.is_file():
            raise ValueError("Missing primary example " + str(path))
        shutil.copyfile(path, PUBLIC / path.name)
    record(PUBLIC / "figure.json", dict(series=series, table=table))


if __name__ == "__main__":
    illustrate()
