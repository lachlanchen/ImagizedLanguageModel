"""Measure recovery from superposed ink and agreement between saved reading and writing."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/pictured_idioms_v1"
ALIGNED = ROOT / "artifacts/pictured_idiom_alignment_v1"
EVIDENCE = ROOT / "references/evidence"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preparation = json.loads((EVIDENCE / "pictured_idioms_20261118/preparation.json").read_text())
    fit = json.loads((EVIDENCE / "pictured_idiom_alignment_20261118/result.json").read_text())
    bindings = {DATA / "pixels.npy": preparation["pixels_sha256"],
                DATA / "validation.npz": preparation["splits"]["validation"]["sha256"]}
    for name in ("diagnostic_choices.npz", "generated.npz"):
        path = ALIGNED / name
        bindings[path] = fit["source_hashes"][str(path.relative_to(ROOT))]
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f"Input changed: {path}")
    pixels = np.load(DATA / "pixels.npy")
    with np.load(DATA / "validation.npz") as archive:
        targets = archive["target"]
    with np.load(ALIGNED / "diagnostic_choices.npz") as archive:
        reading = (archive["supplied_inventory_selection"] == targets).all(1)
        local = archive["local_exact"]
    with np.load(ALIGNED / "generated.npz") as archive:
        writing = ((archive["packed"][:, :4] == pixels[targets]).all((1, 2))
                   & archive["stopped"] & (archive["length"] == 4))
    if (int(reading.sum()), int(local.sum()), int(writing.sum())) != (930, 1012, 134):
        raise ValueError("Physical decisions disagree with the published results")
    combinations = Counter(f"{int(r)}{int(w)}{int(l)}" for r, w, l in zip(reading, writing, local))
    words, frequencies = np.unique(targets, axis=0, return_counts=True)
    shapes = np.unpackbits(pixels, axis=1)
    sums = defaultdict(list)
    reversible = 0
    reversed_same_sum = 0
    for word, freq in zip(words, frequencies):
        planes = shapes[word]
        summed = planes.sum(axis=0, dtype=np.uint8)
        sums[summed.tobytes()].append(int(freq))
        reversed_same_sum += int(freq) * int(not np.array_equal(planes, planes[::-1]) and
                                             np.array_equal(summed, planes[::-1].sum(axis=0)))
        # Store the four position bits in a nibble at each physical coordinate.
        coded = (planes << np.arange(4, dtype=np.uint8)[:, None]).sum(axis=0, dtype=np.uint8)
        decoded = (coded >> np.arange(4, dtype=np.uint8)[:, None]) & 1
        reversible += int(freq) * int(np.array_equal(decoded, planes))
    result = {"validation": len(targets), "distinct_reference_images": len(words),
              "joint_correct_counts_order": ["supplied_reader", "free_writer", "local_count"],
              "joint_correct": dict(sorted(combinations.items())),
              "supplied_reader_correct_free_writer_misses": int((reading & ~writing).sum()),
              "free_writer_correct_supplied_reader_misses": int((writing & ~reading).sum()),
              "free_writer_correct_local_misses": int((writing & ~local).sum()),
              "distinct_pixel_sums": len(sums), "ambiguous_natural_sums": sum(len(v) > 1 for v in sums.values()),
              "unavoidable_errors_from_sums_on_observed_closed_corpus": len(targets) - sum(map(max, sums.values())),
              "changed_under_reversal_with_equal_sum": reversed_same_sum,
              "four_bit_ordered_plane_recovery": reversible,
              "sources": {str(p.relative_to(ROOT)): h for p,h in bindings.items()}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
