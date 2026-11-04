"""Exact image-sequence measures; no OCR or nearest-character assignment."""

from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image

from .visual_copy_language import raster_signatures
from .visual_cell_data import pack_visual_cells


def image_keys(images):
    return [tuple(row) for row in raster_signatures(images).cpu().tolist()]


def answer_scores(output, references, *, stopped):
    keys = image_keys(output)
    predicted = Counter(keys)
    exact, f1 = False, 0.0
    for reference in references:
        gold = image_keys(reference)
        exact |= keys == gold and stopped
        overlap = sum((predicted & Counter(gold)).values())
        f1 = max(f1, 2 * overlap / (len(keys) + len(gold)))
    return {
        "exact_answer": float(exact),
        "raster_f1": f1,
        "output_cells": len(keys),
        "stopped": bool(stopped),
    }


def paired_article_bootstrap(values, articles):
    values = np.asarray(values, dtype=float)
    articles = np.asarray(articles)
    rows = [np.flatnonzero(articles == group) for group in sorted(set(articles))]
    totals = np.asarray([values[r].sum() for r in rows])
    counts = np.asarray([len(r) for r in rows])
    draws = np.random.default_rng(20265819).integers(0, len(rows), (2000, len(rows)))
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "articles": len(rows),
        "examples": len(values),
        "replicates": 2000,
        "seed": 20265819,
    }


def save_answer_image(images, path):
    # A stop-only answer has no glyphs. The JSON/raw tensor records zero cells;
    # its white PNG is a display placeholder, never scored as a generated glyph.
    if len(images):
        pack_visual_cells(images.cpu(), columns=len(images), gutter=0).save(path)
    else:
        Image.new("L", (32, 32), 255).save(path)
