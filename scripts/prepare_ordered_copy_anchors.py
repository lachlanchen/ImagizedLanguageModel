"""Prepare all-source training examples for deciding what to preserve and rewrite."""

from collections import Counter
import time

import numpy as np

from ilm.visual_lm.ordered_copy_anchors import forced_matches
from scripts.train_causal_glyph_translator import ROOT, read_rows, tiles, sha, write_new
from scripts.prepare_visual_sequence_corpus import Rasterizer

OUT = ROOT / "artifacts/ordered_copy_anchors_v1"
REPORT = ROOT / "references/evidence/ordered_copy_anchors_20261119.json"


def prepare():
    start = time.monotonic()
    OUT.mkdir()
    rows = read_rows("train")
    cells, canonical = np.unique(tiles(), axis=0, return_inverse=True)
    render = Rasterizer()
    punctuation = {
        render.tile(c) for c in "。．，、？！；：,.?!;:（）()“”‘’《》「」『』—… "
    }
    punct_idx = {i for i, cell in enumerate(cells) if bytes(cell) in punctuation}
    counts = Counter()
    offsets, pairs = [0], []
    source, target, x_offsets, y_offsets = [], [], [0], [0]
    for r in rows:
        x = canonical[r["x"]].tolist()
        y = canonical[r["z"]].tolist()
        best, anchors = forced_matches(x, y)
        assert all(x[i] == y[j] for i, j in anchors)
        assert all(i < ni and j < nj for (i, j), (ni, nj) in zip(anchors, anchors[1:]))
        source.extend(x)
        target.extend(y)
        x_offsets.append(len(source))
        y_offsets.append(len(target))
        pairs.extend(anchors)
        offsets.append(len(pairs))
        counts["source_cells"] += len(x)
        counts["target_cells"] += len(y)
        counts["maximum_copy"] += best
        counts["forced_pairs"] += len(anchors)
        counts["ambiguous_rows"] += best > len(anchors)
        counts["nonpunctuation_pairs"] += sum(x[i] not in punct_idx for i, j in anchors)
        prev_i = prev_j = -1
        rewrites = 0
        for i, j in anchors + [(len(x), len(y))]:
            a, b = x[prev_i + 1 : i], y[prev_j + 1 : j]
            if a != b:
                rewrites += 1
                counts["rewrite_regions"] += 1
                counts["insertion_only"] += not a
                counts["deletion_only"] += not b
                counts["rewrite_source_cells"] += len(a)
                counts["rewrite_target_cells"] += len(b)
            prev_i, prev_j = i, j
        counts["rows_multiple_rewrites"] += rewrites >= 2
    path = OUT / "training.npz"
    with path.open("xb") as f:
        np.savez_compressed(
            f,
            cells=cells,
            pairs=np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
            offsets=offsets,
            source=np.asarray(source, dtype=np.int32),
            target=np.asarray(target, dtype=np.int32),
            x_offsets=x_offsets,
            y_offsets=y_offsets,
        )
    assert (
        x_offsets[-1] == counts["source_cells"]
        and y_offsets[-1] == counts["target_cells"]
    )
    write_new(
        REPORT,
        dict(
            training_rows=len(rows),
            chapters=len({r["group"] for r in rows}),
            counts=dict(counts),
            file=str(path.relative_to(ROOT)),
            sha256=sha(path),
            seconds=time.monotonic() - start,
            heldout_read=False,
            learned_updates=0,
            output_image_generation=False,
        ),
    )


if __name__ == "__main__":
    prepare()
