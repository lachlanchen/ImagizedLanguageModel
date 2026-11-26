"""Count possible causal targets in the passages already read by the two frozen writers."""

import numpy as np

from scripts.prepare_passage_selector import PUB, WORK, base


def main():
    frozen = base.read(PUB / "data.json")["files"]
    for name in ("train.npz", "forms.json"):
        if base.sha(WORK / name) != frozen[name]:
            raise ValueError(f"Changed input: {name}")
    words = base.read(WORK / "forms.json")
    printable = np.array(
        [word.isascii() and word.isalpha() and len(word) <= 23 for word in words]
    )
    with np.load(WORK / "train.npz") as inputs:
        source = inputs["source"]
        final = inputs["target"][:, 0]
    lengths = (source != 0).sum(1)
    first = source.shape[1] - lengths[:, None] + 16
    interior = int((printable[source] & (np.arange(source.shape[1]) >= first)).sum())
    endpoint = int(printable[final].sum())
    result = dict(
        windows=len(source),
        pictured_word_occurrences=int(lengths.sum()),
        minimum_history=16,
        potential_interior_targets=interior,
        existing_final_targets=endpoint,
        potential_targets_per_existing_target=(interior + endpoint) / endpoint,
        inputs={k: frozen[k] for k in ("train.npz", "forms.json")},
    )
    base.dump(PUB / "supervision_capacity.json", result)
    print(result)


if __name__ == "__main__":
    main()
