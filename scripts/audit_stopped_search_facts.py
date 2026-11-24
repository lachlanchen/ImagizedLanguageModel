"""Separate a wrong visible fact from a wrong highest-probability answer."""
from collections import Counter, defaultdict
import numpy as np
from scripts.prepare_sentence_context import ROOT, FOURTH, Ink, read, dump
from scripts.prepare_fact_channel import MODEL
from scripts.evaluate_visual_discourse import unpack


def main():
    folder = ROOT / "references/evidence/visual_sentence_search_v1"
    records = read(folder / "development.json")["records"]
    rows = read(FOURTH / "confirmation.json")
    saved = unpack(FOURTH / "whole.json.gz")["original"]
    ink = Ink(np.load(MODEL / "palette.npy"))
    counts = defaultdict(Counter)
    for i, (case, rec, out) in enumerate(zip(rows, records, saved, strict=True)):
        assert rec["index"] == i
        actual = bytes.fromhex(out["fact_hex"])
        reference = next(iter(case["facts"].values()))
        canonical = b"".join(ink.tiles[v] for v in ink.encode(reference))
        field = case["task"] + ("-correct-fact" if actual == canonical else "-other-fact")
        counts[field].update(n=1, certified=rec["certified"], accepted=rec["exact"], certified_unmatched=rec["certified"] and not rec["exact"], offered=rec["offered"])
    dump(folder / "fact_strata.json", counts)
    print(dict(counts))


if __name__ == "__main__":
    main()
