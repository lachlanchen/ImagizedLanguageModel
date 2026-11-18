"""Extract the unmodified glyph strips and native chart data for Figure 67."""

import numpy as np
from PIL import Image

from scripts.evaluate_paired_student import read_streams
from scripts.train_screened_visual_targets import OUT, EVIDENCE, load, sha, write_new


def figure():
    replay = load(EVIDENCE / "training_replay.json")["scores"]
    grades = load(EVIDENCE / "scores.json")["scores"]
    grades["teacher"] = load(EVIDENCE / "teacher_scores.json")
    methods = {
        "parent": "Starting writer",
        "book": "Book targets",
        "screened": "Reviewed targets",
        "composer": "Phrase composer",
        "teacher": "Qwen3-8B (text)",
    }
    table = [
        [
            "Writer",
            "Edits on 118",
            "Faithful / 16",
            "Both faithful / 12",
            "Both details / 12",
        ]
    ]
    for key, title in methods.items():
        s = grades[key]
        table.append(
            [
                title,
                f"{s['archive_edits']:,}",
                str(len(s["archive_whole"])),
                str(len(s["whole_pairs"])),
                str(len(s["detail_pairs"])),
            ]
        )
    series = [
        [
            replay[m]["comparison"][target]["exact"]
            for m in ("parent", "book", "screened")
        ]
        for target in ("book_target", "screened_target")
    ]
    strips = [
        ("familiar", "training_replay", 109),
        ("changed", "changes", 127),
        ("unfamiliar", "evaluation", 24),
    ]
    record = {}
    for name, folder, i in strips:
        for stream in ("source", "screened"):
            src = OUT / folder
            seq = read_streams(stream, src)[i]
            cells = np.unpackbits(seq, axis=1).reshape(-1, 32, 32)
            ink = cells.transpose(1, 0, 2).reshape(32, -1)
            image = EVIDENCE / f"{name}_{stream}.png"
            Image.fromarray(255 * (1 - ink)).save(image)
            record[image.stem] = dict(
                cells=len(seq),
                row=i,
                stream=stream,
                folder=folder,
                input_file_sha256=sha(src / (stream + ".npz")),
                png_sha256=sha(image),
            )
    write_new(EVIDENCE / "figure.json", dict(table=table, series=series, strips=record))


if __name__ == "__main__":
    figure()
