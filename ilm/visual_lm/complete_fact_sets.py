"""Rank intact image paths against missing facts and equal-length replacement ink."""

import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.run_survival import utilities


def source_runs(page, tags):
    """Locate labelled runs; reserve every occurrence of the same physical glyph strings."""
    starts = np.flatnonzero(tags == 1)
    ends = starts + 1
    reserved = np.zeros(len(page), np.bool_)
    raw = page.tobytes()
    for j, i in enumerate(starts):
        while ends[j] < len(page) and tags[ends[j]] == 2:
            ends[j] += 1
        needle = page[i : ends[j]].tobytes()
        at = raw.find(needle)
        while at >= 0:
            if at % page.itemsize == 0:
                k = at // page.itemsize
                reserved[k : k + ends[j] - i] = True
            at = raw.find(
                needle, at + page.itemsize if at % page.itemsize == 0 else at + 1
            )
    return starts, ends, reserved


def competing_margins(u, counts, keep, starts, ends, reserved, chosen):
    """Return full minus best rival in each admissible family, plus availability."""
    n, width = keep.shape
    device = u.device
    rows = torch.arange(n, device=device)
    valid = ends > starts
    k = valid.sum(1)
    begin = u[:, :, 1] - u[:, :, 0]
    interior = u[:, :, 2] - u[:, :, 0]
    prefix = F.pad(interior.cumsum(1), (1, 0))

    def gains(left, right):
        return (
            begin.gather(1, left) + prefix.gather(1, right) - prefix.gather(1, left + 1)
        )

    gain = gains(starts, ends)
    ck = counts[rows, k]
    m = torch.arange(1, starts.shape[1] + 1, device=device)[None, :]
    dropped = gain.masked_fill(~valid, float("inf")).sort(1).values.cumsum(1)
    diffs = dropped + ck[:, None] - counts.gather(1, (k[:, None] - m).clamp_min(0))
    omission = diffs.masked_fill(m > k[:, None], float("inf")).amin(1)
    length = (ends - starts)[rows, chosen]
    left = torch.arange(width, device=device).expand(n, -1)
    right = (left + length[:, None]).clamp_max(width)
    occupancy = F.pad((reserved | ~keep).int().cumsum(1), (1, 0))
    legal = (right - left == length[:, None]) & (
        occupancy.gather(1, right) == occupancy[:, :-1]
    )
    other = gains(left, right).masked_fill(~legal, -float("inf")).amax(1)
    available = legal.any(1)
    other = torch.where(available, other, 0.0)
    replacement = gain[rows, chosen] - other
    addition = ck - counts[rows, (k + 1).clamp_max(counts.shape[1] - 1)] - other
    return torch.stack((omission, replacement, addition), 1), torch.stack(
        (k > 0, available, available & (k + 1 < counts.shape[1])), 1
    )


def complete_loss(roles, counts, keep, starts, ends, reserved, chosen):
    margins, valid = competing_margins(
        utilities(roles, keep), counts, keep, starts, ends, reserved, chosen
    )
    penalties = F.softplus(1.0 - margins) * valid
    return (penalties.sum(1) / valid.sum(1)).mean()
