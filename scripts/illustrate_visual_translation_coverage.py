"""Assemble complete saved answer strips and native chart data for Figure 68."""

import numpy as np
from PIL import Image

from scripts.evaluate_paired_student import read_streams
from scripts.train_visual_translation_coverage import DEST, load, sha, write_new


def run():
    scores = load(DEST / "scores.json")["scores"]
    replay = load(DEST / "replay.json")["scores"]
    names = {
        "parent": "Starting writer",
        "narrow": "108 reviewed sources",
        "broad": "841 reviewed sources",
        "composer": "Phrase composer",
        "source": "Literal copy",
        "qwen": "Qwen3-8B (text)",
    }
    table = [
        [
            "Method",
            "Edits on 192",
            "Faithful / 18",
            "Both faithful / 8",
            "Both quantities / 8",
        ]
    ]
    for k, title in names.items():
        r = scores[k]
        table.append(
            [
                title,
                f"{r['archive_edits']:,}",
                str(len(r["natural_correct"])),
                str(len(r["both_meanings"])),
                str(len(r["both_quantities"])),
            ]
        )
    train_streams = DEST / "training_replay"
    choices = replay["broad"]["additional_exact"] or replay["broad"]["nested_exact"]
    s = read_streams("source", train_streams)
    t = read_streams("target", train_streams)
    eligible = [
        i for i in choices if s[i].tobytes() != t[i].tobytes() and len(t[i]) <= 28
    ]
    if eligible:
        i = eligible[0]
        model = "broad"
    else:
        i = next(
            i
            for i in replay["narrow"]["nested_exact"]
            if s[i].tobytes() != t[i].tobytes() and len(t[i]) <= 28
        )
        model = "narrow"
    strips = {}
    for label, folder, j, arm in (
        ("supervised", train_streams, i, model),
        ("natural", DEST / "streams", 38, "broad"),
        ("modified", DEST / "streams", 192, "broad"),
    ):
        for key in ("source", arm):
            seq = read_streams(key, folder)[j]
            cells = np.unpackbits(seq, axis=1).reshape(-1, 32, 32)
            path = DEST / (
                label + ("_input" if key == "source" else "_answer") + ".png"
            )
            ink = cells.transpose(1, 0, 2).reshape(32, -1)
            Image.fromarray(255 * (1 - ink)).save(path)
            strips[path.stem] = dict(
                row=j,
                cells=len(seq),
                file=key,
                folder=folder.name,
                archive_sha=sha(folder / (key + ".npz")),
                png_sha=sha(path),
            )
    series = [
        [round(v, 4) for v in load(DEST / (arm + "_fit.json"))["loss_by_194_steps"]]
        for arm in ("narrow", "broad")
    ]
    write_new(DEST / "figure.json", dict(table=table, strips=strips, series=series))


if __name__ == "__main__":
    run()
