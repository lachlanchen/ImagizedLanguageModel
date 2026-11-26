"""Produce eight-glyph drafts from the frozen optical writer and align by edit distance."""

from time import monotonic
import numpy as np
import torch
from scripts.reserve_visual_prefix import (
    ROOT,
    BASE,
    BOOK,
    PREVIOUS,
    WORK,
    PUB,
    read,
    dump,
    sha,
)
from scripts.write_ordered_translation import restore
from scripts.train_paired_ink import material
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def training_rows():
    receipt = read(ROOT / "references/evidence/book_breadth_v1/corpus.json")
    source = BOOK / "cells.npy"
    replacement = PREVIOUS / "replacement.npz"
    if (
        sha(source) != receipt["files"]["cells.npy"]
        or sha(replacement)
        != read(ROOT / "references/evidence/book_rehearsal_v1/design.json")[
            "replacement"
        ]
    ):
        raise ValueError("Physical training corpus changed")
    cells = np.load(source)
    with np.load(replacement) as z:
        cells[z["indices"]] = z["cells"]
    return cells


def endpoints(draft, ref):
    """Return the optimal reference endpoint and distance for each independently padded row."""
    n = len(draft)
    nd = (draft != 0).sum(1)
    nr = (ref != 0).sum(1)
    cols = np.arange(ref.shape[1] + 1)
    cost = np.broadcast_to(cols, (n, len(cols))).copy()
    for i in range(1, draft.shape[1] + 1):
        row = cost.copy()
        row[:, 0] = i
        for j in cols[1:]:
            row[:, j] = np.minimum(
                np.minimum(cost[:, j] + 1, row[:, j - 1] + 1),
                cost[:, j - 1] + (draft[:, i - 1] != ref[:, j - 1]),
            )
        cost = np.where((nd >= i)[:, None], row, cost)
    valid = cols[None, :] <= np.minimum(nr, 2 * nd)[:, None]
    # Lexicographic minimization of edit cost, displacement from the draft length, then endpoint.
    scale = len(cols) + draft.shape[1] + 1
    score = (cost * scale + np.abs(cols - nd[:, None])) * scale + cols
    j = np.where(valid, score, np.iinfo(np.int64).max).argmin(1)
    return j, cost[np.arange(n), j]


@torch.inference_mode()
def rollout(net, bank, prototype, source, prefix, inventory):
    ctx, valid = net.encode(bank, source)
    # Direction-specific start embedding is supplied as the first prefix vector.
    h, past = net.decode(prefix, ctx, valid)
    result = []
    alive = torch.ones(len(source), device=source.device, dtype=torch.bool)
    for k in range(8):
        got = inventory[(h[:, -1] @ prototype.T).argmax(-1)]
        alive = alive & (got != 1)
        result.append(got.masked_fill(~alive, 0))
        if k < 7:
            h, past = net.decode(bank[got, None], ctx, valid, past, prefix.shape[1] + k)
    return torch.stack(result, dim=1)


def main():
    configure_cuda_cli()
    cells = training_rows()
    lengths = np.count_nonzero(cells, axis=2)
    rng = np.random.default_rng(431109)
    active = rng.random(lengths.shape) < 0.5
    cuts = rng.choice([0, 4, 12, 24], size=lengths.shape).astype(np.uint8)
    cuts[(cuts >= lengths) | ~active] = 0
    drafts = np.zeros((*lengths.shape, 8), np.uint16)
    ends = np.zeros_like(cuts)
    costs = np.zeros_like(cuts)
    start = monotonic()
    with gpu_owner():
        net = restore("book-rehearsal", "cuda")
        features, inv, _ = material("cuda")
        with torch.inference_mode():
            bank = net.optical(features)
            for t in (0, 1):
                proto = net.prototypes(bank, inv[t])
                for cut in (0, 4, 12, 24):
                    ids = np.flatnonzero(active[:, t] & (cuts[:, t] == cut))
                    ids = ids[np.argsort(lengths[ids, 1 - t], kind="stable")]
                    for k in range(0, len(ids), 256):
                        pick = ids[k : k + 256]
                        src = torch.tensor(
                            cells[pick, 1 - t, : lengths[pick, 1 - t].max()].astype(
                                np.int64
                            ),
                            device="cuda",
                        )
                        gold = torch.tensor(
                            cells[pick, t, :cut].astype(np.int64), device="cuda"
                        )
                        prefix = torch.cat(
                            (net.begin[t].expand(len(pick), 1, -1), bank[gold]), dim=1
                        )
                        draft = (
                            rollout(net, bank, proto, src, prefix, inv[t]).cpu().numpy()
                        )
                        drafts[pick, t] = draft
                        end, cost = endpoints(draft, cells[pick, t, cut : cut + 16])
                        ends[pick, t], costs[pick, t] = end, cost
    target = WORK / "drafts.npz"
    np.savez(target, active=active, cuts=cuts, drafts=drafts, ends=ends, costs=costs)
    dump(
        PUB / "drafts.json",
        dict(
            rows=len(cells),
            drafts=int(active.sum()),
            draft_cells=int(np.count_nonzero(drafts)),
            divergent=int(np.count_nonzero(costs)),
            stopped_within_eight=int((active & (drafts[:, :, 7] == 0)).sum()),
            zero_reference_end=int((active & (ends == 0)).sum()),
            seconds=monotonic() - start,
            archive=sha(target),
            parent=sha(ROOT / "models/book_breadth/rehearsal/weights.safetensors"),
            program=sha(ROOT / "scripts/prepare_visual_rollins.py"),
            protocol=sha(ROOT / "references/visual_prefix_recovery_protocol.md"),
        ),
    )


if __name__ == "__main__":
    main()
