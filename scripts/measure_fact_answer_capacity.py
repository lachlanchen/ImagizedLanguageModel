"""Enumerate attainable stopped answers of a frozen optical reader and writer."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_visual_discourse import unpack
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_fact_channel import Ink, MODEL, PUB, WORK, dump, read, sha
from scripts.train_fact_channel import join_regions, load, pack, split_span
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


@torch.inference_mode()
def measure():
    rows = read(WORK / "confirmation.json")
    corpus = read(PUB / "corpus.json")
    if sha(WORK / "confirmation.json") != corpus["files"]["confirmation.json"]:
        raise ValueError("Confirmation corpus mismatch")
    previous = unpack(WORK / "fresh-human.json.gz")["original"]
    reader = load("reader", "cpu")
    writer = load("human", "cuda")
    ink = Ink(np.load(MODEL / "palette.npy"))
    n = len(ink.tiles)
    queries = [ink.encode(row["q"]) for row in rows]
    pages = [[ink.encode(block) for block in row["blocks"]] for row in rows]
    answers = [{tuple(ink.encode(s)) for s in row["answers"]} for row in rows]
    facts = [refs if row["task"] == "exam" else {tuple(ink.encode(row["facts"]["answer"]))}
             for row, refs in zip(rows, answers, strict=True)]
    tiles = ink.array()
    features = torch.from_numpy(observations({"tiles": tiles}, "ink"))
    writer_features = features.to("cuda")
    records = []
    summary = defaultdict(lambda: defaultdict(int))
    pairs = defaultdict(list)
    start_time = monotonic()
    meter = snapshot()
    for row, prev, q, regions, refs, annotations in zip(
        rows, previous, queries, pages, answers, facts, strict=True
    ):
        page = join_regions(regions)
        scores = reader(features, pack([q], "cpu"), pack([page], "cpu"))[0].numpy()
        flat = scores.ravel()
        # Choose the first indices at the cut score as well as at equal higher scores.
        k = min(32, int(np.isfinite(flat).sum()))
        cut = np.partition(flat, len(flat) - k)[-k]
        above = np.flatnonzero(flat > cut)
        selected = np.r_[above, np.flatnonzero(flat == cut)[:k-len(above)]]
        order = selected[np.lexsort((selected, -flat[selected]))]
        spans = [split_span(regions, int(i // scores.shape[1]), int(i % scores.shape[1])+1)
                 for i in order]
        unique = list(dict.fromkeys(tuple(s) for _, _, s in spans))
        context, h = writer.encode(writer_features, n, pack([q]*len(unique), "cuda"),
                                   pack(unique, "cuda"))
        generated = dict(zip(unique, writer.write(context, h, 256), strict=True))
        candidates = []
        for region, offset, cells in spans:
            output, stopped = generated[tuple(cells)]
            pixels = tiles[np.asarray(output, dtype=int)].tobytes()
            candidates.append(dict(region=region, offset=offset, length=len(cells),
                                   fact=tuple(cells) in annotations, stopped=stopped,
                                   exact=bool(stopped and tuple(output) in refs),
                                   output_sha256=sha256(pixels).hexdigest()))
        first = candidates[0]
        if (first["region"], first["offset"], first["length"], first["stopped"], first["output_sha256"]) != (
            prev["selected"], prev["offset"], prev["fact_cells"], prev["stop"], prev["sha256"]
        ):
            raise ValueError(f"Rank one failed physical replay on row {len(records)}")
        group = summary[row["task"]]
        group["n"] += 1
        group["ranks"] += len(candidates)
        group["rank_one_exact"] += first["exact"]
        possible = any(c["exact"] for c in candidates)
        joint = any(c["fact"] and c["exact"] for c in candidates)
        group["any_exact"] += possible
        group["any_fact"] += any(c["fact"] for c in candidates)
        group["any_joint"] += joint
        group["missed_in_support"] += possible and not first["exact"]
        group["outside_support"] += not possible
        pairs[row["task"], row["family"]].append((possible, joint))
        records.append(dict(task=row["task"], family=row["family"], candidates=candidates))
    for (task, _), pair in pairs.items():
        if len(pair) != 2:
            raise ValueError("Expected two questions per page")
        summary[task]["both_attainable"] += all(a for a, _ in pair)
        summary[task]["both_joint"] += all(b for _, b in pair)
    for totals in summary.values():
        if totals["rank_one_exact"] + totals["missed_in_support"] + totals["outside_support"] != totals["n"]:
            raise ValueError("Capacity identity failed")
    dump(WORK / "answer-capacity-rows.json", records)
    receipt = dict(summary=summary, seconds=monotonic()-start_time,
                   device_joules=consumed(meter, snapshot())["joules"],
                   rows_sha256=sha(WORK / "answer-capacity-rows.json"),
                   rank_one_matches=len(records),
                   weights={a: sha(MODEL / a / "weights.safetensors") for a in ["reader","human"]},
                   binding={str(f): sha(f) for f in [Path(__file__), PUB / "corpus.json",
                            Path("formal/span_list/FactAnswerCapacity.lean"),
                            Path("references/fact_channel_capacity_protocol.md")]})
    dump(PUB / "answer-capacity.json", receipt)
    print(dict(summary), flush=True)


if __name__ == "__main__":
    configure_cuda_cli()
    torch.set_num_threads(3)
    with gpu_owner():
        measure()
