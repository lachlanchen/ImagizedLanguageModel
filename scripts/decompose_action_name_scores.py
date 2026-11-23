"""Identify which variation dominates the writer's odds between two pictured names."""

import json
from collections import defaultdict, Counter
import numpy as np
from scripts.test_pictured_action_binding import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha


def main():
    out = {}
    for arm in ("aligned", "independent"):
        reports = json.loads((PUBLIC / f"single_{arm}.json").read_text())["groups"]
        out[arm] = {}
        for group, r in reports.items():
            path = WORK / f"{arm}_{group}.json"
            assert sha(path) == r["detail_sha"]
            tables = defaultdict(dict)
            for row in json.loads(path.read_text()):
                k, w, lang, a = row["key"].rsplit("/", 3)
                w = int(w)
                a = int(a)
                tables[k, lang][w, a] = row["log_margin"] * (-1) ** (w + a)
            values = []
            dominant = Counter()
            for table in tables.values():
                x, y, z, w = [table[j] for j in ((0, 0), (0, 1), (1, 0), (1, 1))]
                c, p, q, i = (
                    np.array(
                        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
                    )
                    @ np.array([x, y, z, w])
                ) / 4
                biases = (-(c + p + q), c + p - q, c - p + q, -c + p + q)
                assert np.isclose(min(x, -y, -z, w), i - max(biases))
                values.append((c, p, q, i, min(x, -y, -z, w)))
                absolute = np.abs([c, p, q, i])
                dominant[
                    ("name", "page", "question", "interaction")[int(absolute.argmax())]
                    if (absolute == absolute.max()).sum() == 1
                    else "tie"
                ] += 1
            arr = np.array(values)
            count = int((arr[:, -1] > 0).sum())
            assert count == sum(s["all_preferred"] for s in r["counts"].values())
            out[arm][group] = dict(
                crossings=len(tables),
                all_four=count,
                positive_interaction=int((arr[:, 3] > 0).sum()),
                largest_absolute_component=dominant,
                median_absolute_coefficients=dict(
                    zip(
                        ("name", "page", "question", "interaction"),
                        np.median(abs(arr[:, :4]), axis=0).tolist(),
                    )
                ),
                median_worst_margin=float(np.median(arr[:, -1])),
            )
    record(PUBLIC / "name_odds.json", out)


if __name__ == "__main__":
    main()
