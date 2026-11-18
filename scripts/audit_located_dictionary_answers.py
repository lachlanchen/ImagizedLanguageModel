"""Check whether an incorrect stopped image answers a different question on its card."""
from collections import Counter, defaultdict
import json

import numpy as np

from scripts.audit_general_entry_writer import EVIDENCE, archived
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_general_entry_writer import AUTHORED, ENTRIES, WORK
from scripts.train_paired_passage import unpack_sequences


def main():
    scores = load(EVIDENCE / "selection.json")
    folder = ENTRIES / "development"
    meta = load(folder / "index.json")
    tiles = np.load(folder / "pixels.npy")
    targets = [tiles[s].tobytes() for s in unpack_sequences(folder / "answers.npz")]
    known = defaultdict(dict)
    for i, (m, im) in enumerate(zip(meta, targets, strict=True)):
        assert im not in known[m["card"]]
        known[m["card"]][im] = i
    authored = [json.loads(l)["row"] for l in AUTHORED.read_text().splitlines()]
    report = {}
    for mode, result in scores["modes"].items():
        path = WORK / "visible_selection" / (mode + ".npz")
        streams, stopped = archived(path, result["sha256"])
        counts = Counter()
        rows = authored if mode == "independent" else range(len(meta))
        for x, stop, r in zip(streams, stopped, rows, strict=True):
            m = meta[r]
            other = known[m["card"]].get(x)
            if not stop:
                counts["unstopped"] += 1
            elif other is None:
                counts["outside_card_responses"] += 1
            elif r == other:
                counts["exact"] += 1
            else:
                n = meta[other]
                changes = [k for k in ("head","field","stage") if m[k] != n[k]]
                counts["wrong_" + "_and_".join(changes)] += 1
        assert sum(counts.values()) == result["records"] and counts["exact"] == result["exact"]
        report[mode] = dict(counts)
    write_json(EVIDENCE / "located_errors.json", dict(modes=report, program_sha256=sha(__file__)))
    print(report)


if __name__ == "__main__":
    main()
