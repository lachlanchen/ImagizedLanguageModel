"""Pair results by passage and preserve both question languages when estimating uncertainty."""

from collections import Counter
import json
import numpy as np
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.role_change_corpus import load
from scripts.train_causal_glyph_writer import record


def main():
    report = {}
    for group in ("duie", "fresh"):
        scenes = load()[group]
        records = {
            arm: json.loads((WORK / f"{group}_{arm}.json").read_text())
            for arm in ("aligned", "independent")
        }
        a, c = records.values()
        assert len(a) == len(c) == len(scenes) * 8 and [x["key"] for x in a] == [
            x["key"] for x in c
        ]
        differences = []
        by = {}
        for i, scene in enumerate(scenes):
            ix = slice(i * 8, i * 8 + 8)
            vals = []
            for arm, rows in records.items():
                flags = [x["ordered"] for x in rows[ix]]
                contrasts = sum(
                    all(flags[j] for j in js) for js in ((0, 1, 4, 5), (2, 3, 6, 7))
                )
                by.setdefault((scene["kind"], arm), Counter()).update(
                    requests=8, ordered=sum(flags), contrasts=2, complete=contrasts
                )
                vals.append(contrasts / 2)
            differences.append(vals[0] - vals[1])
        rng = np.random.default_rng(242110)
        delta = np.array(differences)
        dist = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
        report[group] = dict(
            paired_contrast_difference=float(delta.mean()),
            interval95=np.quantile(dist, [0.025, 0.975]).tolist(),
            resampling_unit="whole paired scene",
            cases=len(scenes),
            by_kind={k + "_" + arm: v for (k, arm), v in by.items()},
        )
    record(PUBLIC / "transfer.json", report)


if __name__ == "__main__":
    main()
