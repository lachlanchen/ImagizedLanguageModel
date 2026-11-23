"""Compare translation and stroke/scale augmentation with identical sampled glyphs."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_typeface_writer import (
    BASE,
    FACES,
    PREVIOUS,
    ROOT,
    SEED,
    Face,
    fit_correspondence,
    training_rows,
)
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner

WORK = ROOT / "artifacts/stroke_scale_v1"
PUBLIC = ROOT / "references/evidence/stroke_scale_20261124"
FANDOL = (
    Path(
        "/usr/share/texlive/texmf-dist/fonts/opentype/public/fandol/FandolSong-Regular.otf"
    ),
    0,
)
ARMS = ("translation", "stroke_scale")


def distort(ink, rng):
    batch = len(ink)
    draws = torch.rand(batch, 4, generator=rng, device=ink.device)
    theta = torch.zeros(batch, 2, 3, device=ink.device)
    inv_scale = 1 / (0.78 + 0.44 * draws[:, :2])
    theta[:, 0, 0], theta[:, 1, 1] = inv_scale.T
    source = ink[:, None].float()
    grid = F.affine_grid(theta, source.shape, align_corners=False)
    values = F.grid_sample(source, grid, align_corners=False)[:, 0]
    threshold = 0.32 + 0.18 * (draws[:, 2] * 3).floor()
    scaled = values >= threshold[:, None, None]
    use = (draws[:, 3] >= 0.25) & scaled.flatten(1).any(1)
    return torch.where(use[:, None, None], scaled, ink)


class PresentedViews:
    def __init__(self):
        self.images = {}
        self.exposure = Counter()
        self.schedule = hashlib.sha256()

    def __call__(self, ink, labels, fonts, dy, dx):
        targets = labels.cpu().numpy()
        self.schedule.update(targets.tobytes())
        self.schedule.update(fonts.cpu().numpy().tobytes())
        self.schedule.update(bytes((dy + 1, dx + 1)))
        packed = np.packbits(ink.cpu().numpy().reshape(len(ink), -1), axis=1)
        for row, label in zip(packed, targets.tolist(), strict=True):
            key = row.tobytes()
            if key not in self.images:
                self.images[key] = label
            elif self.images[key] != label:
                self.images[key] = -1
            self.exposure[key] += 1

    def summary(self):
        conflicts = [key for key, label in self.images.items() if label == -1]
        return dict(
            unique=len(self.images),
            conflicting_bitmaps=len(conflicts),
            occurrences_on_conflicting_bitmaps=sum(self.exposure[k] for k in conflicts),
            exposures=sum(self.exposure.values()),
            schedule_sha256=self.schedule.hexdigest(),
        )


def train():
    WORK.mkdir()
    PUBLIC.mkdir(exist_ok=True, parents=True)
    expected = json.loads((PREVIOUS / "distilled/page_fit.json").read_text())["weights"]
    assert all(sha(BASE / k) == h for k, h in expected.items())
    faces = [Face(f) for f in FACES]
    rows = training_rows()
    schedules = {}
    for arm in ARMS:
        rng = torch.Generator(device="cuda").manual_seed(SEED + 1)
        observer = PresentedViews()
        fit_correspondence(
            rows,
            faces,
            path=WORK / (arm + ".safetensors"),
            report=PUBLIC / (arm + "_fit.json"),
            transform=(lambda x: distort(x, rng)) if arm == "stroke_scale" else None,
            observe=observer,
        )
        schedules[arm] = observer.summary()
        del observer
    assert (
        schedules[ARMS[0]]["schedule_sha256"] == schedules[ARMS[1]]["schedule_sha256"]
    )
    record(PUBLIC / "presentations.json", schedules)


if __name__ == "__main__":
    with gpu_owner():
        train()
