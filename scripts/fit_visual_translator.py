"""Train a pictured phrase translator from parallel JSONL text or bitmap pages."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ilm.visual_lm.bitmap_alignment import blocks
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.infer_continuous_image_language import image_cells


def train(rows, target_key, output, weight):
    definitions = {}
    pairs = []
    renderer = None

    def read(record, key):
        nonlocal renderer
        path = record.get(key + "_image")
        if path is not None and key in record:
            raise ValueError(f"Choose either {key} or {key}_image")
        if path is not None:
            pixels = image_cells(
                rows.parent / path, record.get(key + "_cells")
            ).reshape(-1, 1024)
            pictures = map(bytes, np.packbits(pixels, axis=1))
        else:
            if renderer is None:
                from scripts.prepare_visual_sequence_corpus import Rasterizer

                renderer = Rasterizer()
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Expected nonempty {key} text or {key}_image")
            pictures = (renderer.tile(c) for c in record[key])
        return tuple(definitions.setdefault(p, len(definitions)) for p in pictures)

    with rows.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                pairs.append((read(record, "source"), read(record, target_key)))
            except (ValueError, KeyError, OSError) as e:
                raise ValueError(f"Invalid example at line {n} of {rows}: {e}") from e
    if not pairs:
        raise ValueError("The parallel training file is empty")
    bank = np.frombuffer(b"".join(definitions), np.uint8).reshape(-1, 128)
    model = VisualPhraseComposer.fit(bank, pairs, blocks, weight=weight)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as stage:
        new = Path(stage) / "model.gz"
        model.save(new)
        os.link(new, output)
    return dict(
        pairs=len(pairs),
        images=len(bank),
        phrases=len(model.phrases),
        bytes=output.stat().st_size,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rows", type=Path, help="One parallel example per line")
    p.add_argument("--target-key", default="target")
    p.add_argument(
        "--weight", type=float, default=0.25, help="Cross-phrase association weight"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Choose a new weight path")
    if not np.isfinite(a.weight):
        p.error("Use a finite weight")
    print(json.dumps(train(a.rows, a.target_key, a.output, a.weight)))
