"""Freeze a chapter-balanced review subset and save the unmodified copy and phrase controls."""

from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_paired_student import save_streams
from scripts.prepare_phrase_boundary_trial import (
    ROOT,
    Rasterizer,
    load,
    order,
    sha,
    write_new,
)
from scripts.prepare_ordered_copy_anchors import OUT, REPORT
from scripts.train_pictured_revision import COMPOSER, composer
from scripts.train_visual_event_roles import raster


def run():
    roster = ROOT / "references/evidence/anchor_writer_holdout_20261119.json"
    split = load(roster)
    assert sha(OUT / "training.npz") == split["training_sha"] == load(REPORT)["sha256"]
    rows = split["rows"]
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["length_band"]].append(i)
    reader_ids = []
    used = set()
    for band in sorted(groups):
        count = 0
        for i in sorted(
            groups[band],
            key=lambda i: order("anchor-meaning-1/" + rows[i]["source_sha"]),
        ):
            ch = rows[i]["chapter"]
            if ch in used:
                continue
            reader_ids.append(i)
            used.add(ch)
            count += 1
            if count == 6:
                break
        assert count == 6
    out = OUT / "controls"
    out.mkdir()
    engine = composer()
    render = Rasterizer()
    streams = defaultdict(list)
    costs = defaultdict(list)
    for r in rows:
        x = raster(r["source"], render)
        y = raster(r["target"], render)
        values = dict(source=x, reference=y, composer=engine.generate(x))
        for k, v in values.items():
            streams[k].append(v)
            if k != "reference":
                costs[k].append(
                    Levenshtein.distance(tuple(map(bytes, v)), tuple(map(bytes, y)))
                )
    for k, v in streams.items():
        save_streams(k, v, out)
    write_new(
        ROOT / "references/evidence/anchor_writer_controls_20261119.json",
        dict(
            roster_sha=sha(roster),
            composer_sha=sha(COMPOSER),
            n=len(rows),
            reader_rows=reader_ids,
            unique_review_chapters=len(used),
            reference_cells=sum(map(len, streams["reference"])),
            per_case_edit_counts=dict(costs),
            sums={k: sum(v) for k, v in costs.items()},
            streams={k: sha(out / (k + ".npz")) for k in streams},
            trained_candidate=False,
            semantic_scoring_performed=False,
        ),
    )


if __name__ == "__main__":
    run()
