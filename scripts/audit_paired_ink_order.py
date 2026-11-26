"""Measure what image order contributes on previously frozen bilingual siblings."""

from collections import Counter
import numpy as np
import torch
from scripts.prepare_paired_ink import BASE, WORK, PUB, read, dump, sha
from scripts.train_paired_ink import material, scores_for_pairs
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore, stopped


@torch.inference_mode()
def main():
    if sha(WORK / "held.json") != read(PUB / "split.json")["files"]["held"]:
        raise ValueError("Holdout changed")
    held = read(WORK / "held.json")
    palette = np.load(BASE / "palette.npy")
    # Canonical ordering is by the 1,024 ink bits, independent of any transcription.
    rank = {
        i: k
        for k, i in enumerate(
            sorted(range(2, len(palette)), key=lambda i: bytes(palette[i]))
        )
    }
    torch.set_num_threads(3)
    f, inv, local = material("cpu")
    net = restore("paired-assignment", "cpu")
    result = Counter()
    rows = []
    for t in (0, 1):
        for j in range(0, len(held), 12):
            batch = [r["cells"] for r in held[j : j + 12]]
            modified = [
                [
                    [
                        sorted(seq, key=rank.__getitem__) if k == 1 - t else seq
                        for k, seq in enumerate(pair)
                    ]
                    for pair in family
                ]
                for family in batch
            ]
            margins = []
            for view in (batch, modified):
                good, bad, _ = scores_for_pairs(net, f, inv, local, view, t, "cpu")
                margins.append((good - bad).sum(dim=1).tolist())
            inp = [z[1 - t] for family in modified for z in family]
            ys = net.generate(f, pack(inp, "cpu"), inv[t], t)
            for k, fam in enumerate(modified):
                collision = fam[0][1 - t] == fam[1][1 - t]
                raw, bag = (values[k] for values in margins)
                if collision:
                    assert abs(bag) < 0.0001 and stopped(ys[2 * k]) == stopped(
                        ys[2 * k + 1]
                    )
                    assert fam[0][t] != fam[1][t]
                result["cases"] += 1
                result["collisions"] += int(collision)
                result["ordered_correct"] += int(raw > 0)
                result["sorted_correct"] += int(bag > 1e-4)
                result["ordered_correct_on_collisions"] += int(collision and raw > 0)
                rows.append(
                    dict(
                        id=held[j + k]["id"],
                        target=t,
                        source_collision=collision,
                        ordered_gap=raw,
                        bag_gap=bag,
                    )
                )
    dump(PUB / "order.json", dict(counts=result, rows=rows))
    print(result)


if __name__ == "__main__":
    main()
