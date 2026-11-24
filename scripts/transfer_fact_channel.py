"""Use the deployed pictured-page pipeline to continue books and return historical specimens."""

from collections import defaultdict
import numpy as np
import torch
from scripts.prepare_fact_channel import MODEL, PUB, Ink, dump, read
from scripts.train_region_writer import gpu_owner
from scripts.write_fact_channel import VisualAnswer, bitmap_ids
from scripts.evaluate_recurrent_history import historical_pages
from scripts.train_visual_discourse import Corpus
from scripts.evaluate_visual_discourse import cohorts
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    ink = Ink(np.load(MODEL / "palette.npy"))
    specs = []
    labels, images, targets = historical_pages()
    for i, r in enumerate(labels):
        q, p = (bitmap_ids(ink, img) for img in images[2 * i : 2 * i + 2])
        specs.append(
            (
                r["language"] + "-period",
                q,
                [p[j : j + 256] for j in range(0, len(p), 256)],
                bitmap_ids(ink, targets[i]),
            )
        )
    corpus = Corpus()
    for t, rows in cohorts(corpus)[1].items():
        if not t.startswith("book-"):
            continue
        q = ink.encode(
            "请续写下一行。" if t.endswith("zh") else "Continue with the next line."
        )
        for r in rows:
            page = bitmap_ids(ink, corpus.tiles[r["x"]])
            specs.append(
                (
                    t,
                    q,
                    [page[j : j + 256] for j in range(0, len(page), 256)],
                    bitmap_ids(ink, corpus.tiles[r["y"][:8]]),
                )
            )
    tiles = ink.array()
    scores = {}
    with gpu_owner(), torch.inference_mode():
        for arm in ("direct", "human", "predicted"):
            net = VisualAnswer(arm, "cuda")
            result = defaultdict(list)
            for j in range(0, len(specs), 12):
                batch = specs[j : j + 12]
                values, _ = net.answer(
                    tiles, [q for t, q, p, y in batch], [p for t, q, p, y in batch]
                )
                for (t, q, p, y), (ids, end) in zip(batch, values, strict=True):
                    result[t].append(
                        ids[:8] == y if t.startswith("book-") else end and ids == y
                    )
            scores[arm] = {k: dict(n=len(v), correct=sum(v)) for k, v in result.items()}
    dump(PUB / "transfer.json", scores)
    print(scores)


if __name__ == "__main__":
    configure_cuda_cli()
    main()
