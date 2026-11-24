"""Canonicalize printed glyph images by symmetric stroke distance in pixel space."""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F


def descriptors(cells, device):
    masks = np.zeros((len(cells), 28, 28), np.uint8)
    distances = np.ones_like(masks, np.float32)
    for i, raw in enumerate(cells):
        im = np.unpackbits(raw).reshape(32, 32)
        r, c = im.nonzero()
        if not len(r):
            continue
        clip = Image.fromarray(255 * im[r.min() : r.max() + 1, c.min() : c.max() + 1])
        scale = 26 / max(clip.size)
        w, h = (max(1, round(x * scale)) for x in clip.size)
        tile = np.asarray(clip.resize((w, h), Image.Resampling.BILINEAR)) > 100
        y, x = (28 - h) // 2, (28 - w) // 2
        masks[i, y : y + h, x : x + w] = tile
        distances[i] = np.minimum(distance_transform_edt(1 - masks[i]), 4) / 4
    ink = torch.from_numpy(masks).to(device, dtype=torch.float32)
    pooled = F.normalize(F.avg_pool2d(ink[:, None], 2).flatten(1), dim=-1)
    flat = ink.flatten(1)
    return (
        flat / flat.sum(1, keepdim=True).clamp_min(1),
        torch.from_numpy(distances).to(device).flatten(1),
        pooled,
    )


class StrokeMatcher:
    """Index physical exemplars. Select the closest bitmap for each unique input cell."""

    def __init__(self, bank, device="cpu"):
        if bank.dtype != np.uint8 or bank.ndim != 2 or bank.shape[1] != 128:
            raise ValueError("Expected packed glyph bank")
        self.bank = bank
        self.device = device
        self.exact = {cell.tobytes(): i for i, cell in enumerate(bank)}
        self.reference = descriptors(bank, device)

    @torch.inference_mode()
    def match(self, cells):
        unique, reverse = np.unique(cells.reshape(-1, 128), axis=0, return_inverse=True)
        best = np.array([self.exact.get(raw.tobytes(), -1) for raw in unique], np.int64)
        missing = np.flatnonzero(best < 0)
        ri, rd, rp = self.reference
        for start in range(0, len(missing), 128):
            ids = missing[start : start + 128]
            ink, dist, pool = descriptors(unique[ids], self.device)
            score = (ink @ rd.T + dist @ ri.T) * 0.5 + 1 - pool @ rp.T
            best[ids] = score.argmin(-1).cpu().numpy()
        return self.bank[best[reverse]].reshape(cells.shape)
