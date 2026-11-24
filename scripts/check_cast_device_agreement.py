"""Diagnose the single cross-device disagreement in the frozen GAP PNG replay."""

import numpy as np
import torch

from scripts.check_mention_transfer import datasets
from scripts.prepare_story_mentions import PUBLIC, dump
from scripts.story_mention_options import select
from scripts.train_semantic_page import gpu_owner
from scripts.train_story_cast import load_cast, prepare


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    rows = datasets()[0]["gap"]
    key = "test-879"
    ix = next(i for i, r in enumerate(rows) if r["id"] == key)
    row = rows[ix]
    start = ix - ix % 48
    j = ix - start
    result = {}
    with gpu_owner():
        for device in ("cpu", "cuda"):
            obs, (ids, slot, bounds, n, _), meta = prepare(
                rows[start : start + 48], "ink", device
            )
            net = load_cast("ink", device)
            slots, spans, q = meta[j]
            m = len(slots)
            x = obs[ids.long()]
            runs = {
                "full": ((x, slot, bounds, n), j),
                "padded": (
                    (x[j : j + 1], slot[j : j + 1], bounds[j : j + 1], n[j : j + 1]),
                    0,
                ),
                "single": (
                    (
                        x[j : j + 1, :m],
                        slot[j : j + 1, :m],
                        bounds[j : j + 1],
                        n[j : j + 1],
                    ),
                    0,
                ),
            }
            for label, (args, index) in runs.items():
                z = net(*args)[index].cpu().numpy()
                weights = np.exp(z.astype("float64") - z.max())
                masses = {"null": float(weights[-1])}
                for i, ((a, b), tag) in enumerate(zip(spans[: q[0]], slots[: q[0]])):
                    if tag >= 0:
                        name = row["text"][a:b]
                        masses[name] = masses.get(name, 0.0) + float(weights[i])
                result[f"{device}-{label}"] = dict(
                    selected=select(z, slots, spans, q[0]), relative_masses=masses
                )
    dump(
        PUBLIC / "cast-device-disagreement.json",
        dict(source="GAP " + key, truth=row["truth"], conditions=result),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
