"""Compare answers on the same questions, sampling whole English Wikipedia articles."""

from collections import defaultdict
import gzip
import json
import numpy as np
from scripts.prepare_story_mentions import PUBLIC, dump
from scripts.check_mention_transfer import datasets
from scripts.train_story_cast import partitions


def main():
    external, _ = datasets()
    parts, _ = partitions()
    out = {}
    for name in ("gap", "clue"):
        for kind, rows in (("official", external[name]), ("fresh", parts[name][1])):
            tag = name + "-" + kind
            dec = json.loads(
                gzip.decompress(
                    (PUBLIC / ("cast-" + tag + "-decisions.json.gz")).read_bytes()
                )
            )

            def flags(arm):
                out = []
                for r, loc in zip(rows, dec[arm], strict=True):
                    val = None if loc is None else r["text"][slice(*loc)]
                    gold = (
                        None
                        if r["truth"] is None
                        else r["text"][slice(*r["cast"][r["truth"]])]
                    )
                    out.append(
                        val == gold
                        if name == "gap"
                        else (val == r["text"][slice(*r["cast"][0])])
                        == (gold is not None)
                    )
                return np.array(out)

            cand = flags("ink")
            summary = {}
            for arm in ("control", "nearest", "byte", "geometry", "always-null"):
                orig = flags(arm)
                delta = cand.astype(int) - orig
                stats = dict(
                    wins=int((delta == 1).sum()),
                    losses=int((delta == -1).sum()),
                    ties=int((delta == 0).sum()),
                )
                if name == "gap":
                    groups = defaultdict(list)
                    for r, d in zip(rows, delta):
                        groups[r["article"]].append(d)
                    pairs = np.array([[sum(x), len(x)] for x in groups.values()])
                    rng = np.random.default_rng(913)
                    boot = []
                    for _ in range(50):
                        sample = pairs[
                            rng.integers(len(pairs), size=(100, len(pairs)))
                        ].sum(1)
                        boot.extend(100 * sample[:, 0] / sample[:, 1])
                    stats.update(
                        articles=len(pairs),
                        difference_points=100 * delta.mean(),
                        article_bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
                    )
                summary[arm] = stats
            out[tag] = summary
    dump(PUBLIC / "paired-comparisons.json", out)


if __name__ == "__main__":
    main()
