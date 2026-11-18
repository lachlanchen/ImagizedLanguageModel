"""Render the approved training-only drafts to paired 32-by-32 visual streams."""

import csv
import numpy as np
from scripts.evaluate_anchor_gap_writer import (
    EVIDENCE,
    Rasterizer,
    load,
    sha,
    write_new,
)


def pack():
    sample = load(EVIDENCE / "training_sample.json")
    teacher = load(EVIDENCE / "teacher_drafts.json")
    with (EVIDENCE / "teacher_review.tsv").open() as f:
        marks = list(csv.DictReader(f, delimiter="\t"))
    render = Rasterizer()
    source, target, rows = [], [], []
    assert len(sample) == len(marks) == len(teacher["drafts"]) == 32
    for r, v, m in zip(sample, teacher["drafts"], marks):
        assert r["row"] == v["row"] == int(m["row"]) and m["use_draft"] in {"0", "1"}
        if m["use_draft"] == "1":
            assert v["done_reason"] == "stop"
            rows.append(r["row"])
            source.append(
                np.frombuffer(
                    b"".join(map(render.tile, r["source"])), np.uint8
                ).reshape(-1, 128)
            )
            target.append(
                np.frombuffer(b"".join(map(render.tile, v["draft"])), np.uint8).reshape(
                    -1, 128
                )
            )
    path = EVIDENCE / "teacher_training_pixels.npz"
    with path.open("xb") as f:
        np.savez_compressed(
            f,
            source=np.concatenate(source),
            target=np.concatenate(target),
            rows=np.array(rows),
            source_offsets=np.cumsum([0] + list(map(len, source))),
            target_offsets=np.cumsum([0] + list(map(len, target))),
        )
    with np.load(path) as z:
        for key, pieces in (("source", source), ("target", target)):
            for i, part in enumerate(pieces):
                a, b = z[key + "_offsets"][i : i + 2]
                assert np.array_equal(z[key][a:b], part)
    write_new(
        EVIDENCE / "teacher_pilot.json",
        dict(
            accepted=len(rows),
            screened=len(sample),
            rows=rows,
            archive_sha256=sha(path),
            teacher_checkpoint=teacher["digest"],
            source_cells=sum(map(len, source)),
            target_cells=sum(map(len, target)),
            student_training_performed=False,
        ),
    )


if __name__ == "__main__":
    pack()
