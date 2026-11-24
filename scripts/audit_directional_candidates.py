"""Bound any final reranker by its actual finite candidate images on held pages."""

from collections import Counter, defaultdict

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_directional_reading import PUB, WORK, dump, read, sha
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.train_fact_channel import join_regions, pack
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_directional_reading import DirectionalAnswer


def main():
    configure_cuda_cli()
    assert (
        sha(WORK / "confirmation.json")
        == read(PUB / "reservation.json")["files"]["confirmation.json"]
    )
    rows = read(WORK / "confirmation.json")
    old = unpack(WORK / "signed.json.gz")["original"]
    ink = Ink(np.load(BASE / "palette.npy"))
    q = [ink.encode(r["q"]) for r in rows]
    pages = [join_regions([ink.encode(b) for b in r["blocks"]]) for r in rows]
    answers = [{tuple(ink.encode(a)) for a in r["answers"]} for r in rows]
    features = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
    counts = defaultdict(Counter)
    locations = []
    with gpu_owner(), torch.inference_mode():
        net = DirectionalAnswer("signed", "cuda")
        for i in range(0, len(q), 12):
            scores, _ = net.reader.lattice(
                features, pack(q[i : i + 12], "cuda"), pack(pages[i : i + 12], "cuda")
            )
            flat = scores.flatten(1)
            ix = flat.argsort(descending=True, stable=True)[:, :32]
            ix = ix.masked_fill(~flat.gather(1, ix).isfinite(), -1)
            for k, indices in enumerate(ix.tolist(), i):
                items = []
                for j in indices:
                    if j >= 0:
                        s, n = divmod(j, scores.size(-1))
                        items.append(tuple(pages[k][s : s + n + 1]))
                tile = bytes.fromhex(old[k]["fact_hex"])
                assert any(
                    b"".join(ink.tiles[t] for t in item) == tile for item in items
                )
                ranks = [v for v, item in enumerate(items, 1) if item in answers[k]]
                exact = any(
                    tile == b"".join(ink.tiles[t] for t in a) for a in answers[k]
                )
                g = counts[rows[k]["task"]]
                g["questions"] += 1
                g["top1"] += 1 in ranks
                g["top8"] += any(v <= 8 for v in ranks)
                g["top32"] += bool(ranks)
                g["final_exact"] += exact
                g["no_correct_candidate"] += not ranks
                g["available_but_not_selected"] += bool(ranks) and not exact
                locations.append(
                    dict(
                        first_accepted_rank=min(ranks, default=None),
                        emitted_exact=exact,
                    )
                )
    dump(
        PUB / "candidates.json",
        dict(scores=counts, rows=locations, corpus=sha(WORK / "confirmation.json")),
    )


if __name__ == "__main__":
    main()
