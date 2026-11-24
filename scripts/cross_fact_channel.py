"""Exchange real predicted facts across sibling questions and ablate the surrounding region."""

from collections import defaultdict
from hashlib import sha256
import numpy as np
import torch
from scripts.prepare_fact_channel import MODEL, WORK, PUB, Ink, read, dump
from scripts.evaluate_visual_discourse import unpack
from scripts.evaluate_region_writer import contains_fact
from scripts.write_fact_channel import VisualAnswer
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    rows = read(WORK / "confirmation.json")
    cached = unpack(WORK / "fresh-predicted.json.gz")["original"]
    ink = Ink(np.load(MODEL / "palette.npy"))
    queries = [ink.encode(r["q"]) for r in rows]
    pages = [[ink.encode(b) for b in r["blocks"]] for r in rows]
    facts = []
    for i, v in enumerate(cached):
        b = v["selected"]
        k = v["offset"]
        n = v["fact_cells"]
        f = pages[i][b][k : k + n]
        assert b"".join(ink.tiles[j] for j in f).hex() == v["fact_hex"]
        facts.append((b, k, f))
    pairs = defaultdict(list)
    for i, r in enumerate(rows):
        pairs[(r["task"], r["family"])].append(i)
    swap = {}
    for a, b in pairs.values():
        assert pages[a] == pages[b] and queries[a] != queries[b]
        swap[a] = b
        swap[b] = a
    tiles = ink.array()
    chars = dict(enumerate(read(MODEL / "readback.json")))
    for c, i in ink.chars.items():
        chars.setdefault(i, c)
    counts = {}
    with gpu_owner(), torch.inference_mode():
        for arm in ("human", "predicted"):
            actor = VisualAnswer(arm, "cuda")
            conditions = [("ordinary", False, "both"), ("sibling", True, "both")]
            if arm == "predicted":
                conditions.extend(
                    [("fact_only", False, "fact"), ("region_only", False, "region")]
                )
            reference = unpack(WORK / ("fresh-" + arm + ".json.gz"))["original"]
            for name, moved, mode in conditions:
                tally = defaultdict(lambda: defaultdict(int))
                for j in range(0, len(rows), 12):
                    ix = list(range(j, min(j + 12, len(rows))))
                    spans = [facts[swap[i] if moved else i] for i in ix]
                    vals, _ = actor.answer(
                        tiles,
                        [queries[i] for i in ix],
                        [pages[i] for i in ix],
                        spans,
                        evidence=mode,
                    )
                    for i, (ids, end) in zip(ix, vals):
                        r = rows[i]
                        v = tally[r["task"]]
                        raw = tiles[np.asarray(ids, int)].tobytes()
                        matches = end and any(
                            raw == b"".join(ink.tiles[k] for k in ink.encode(s))
                            for s in r["answers"]
                        )
                        t = "catalogue" if r["task"] == "exam" else r["task"]
                        text = "".join(chars[k] for k in ids)
                        hit = end and contains_fact(text, r["facts"]["answer"], t)
                        other = end and contains_fact(
                            text, rows[swap[i]]["facts"]["answer"], t
                        )
                        different = (
                            cached[i]["fact_sha256"] != cached[swap[i]]["fact_sha256"]
                        )
                        if arm == "predicted":
                            different |= (
                                cached[i]["evidence_region_sha256"]
                                != cached[swap[i]]["evidence_region_sha256"]
                            )
                        v["n"] += 1
                        v["exact"] += matches
                        v["fact"] += hit
                        v["sibling_fact"] += other
                        v["different_records"] += different
                        v["exact_on_different"] += different and matches
                        v["requested_on_different"] += different and hit
                        v["sibling_on_different"] += different and other
                        same = (
                            sha256(raw).hexdigest() == reference[i]["sha256"]
                            and end == reference[i]["stop"]
                        )
                        v["unchanged_outputs"] += same
                        if name == "ordinary" and not same:
                            raise ValueError("Original output changed")
                counts.setdefault(arm, {})[name] = tally
    dump(PUB / "crossed-evidence.json", counts)
    print(counts)


if __name__ == "__main__":
    configure_cuda_cli()
    main()
