"""Compare physical training exposure and estimate paired transfer uncertainty by scene."""

from collections import defaultdict, Counter
from hashlib import sha256
import json
import numpy as np
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_individual_clause_writer import PUBLIC, WORK, training


def exposure():
    signatures = []
    for arm in ("independent", "aligned"):
        inputs = sha256()
        by_group = defaultdict(Counter)
        local = defaultdict(Counter)
        counts = Counter()
        rows, _ = training(arm)
        for r in rows:
            for buf in (r.task.encode(), r.key.encode(), r.p, r.q):
                inputs.update(len(buf).to_bytes(4, "big"))
                inputs.update(buf)
            label = sha256(r.y).digest()
            if r.task.startswith("single_"):
                scene, w, lang, action, repeat = r.key.rsplit("/", 4)
                group = (scene, lang, action)
                local[group, w][label] += 1
            else:
                group = (r.task, r.key)
            by_group[group][label] += 1
            counts.update(rows=1, target_positions=1 + len(r.y) // 128)
        fit = json.loads((PUBLIC / f"{arm}_fit.json").read_text())
        assert (
            counts["rows"] == fit["rows"]
            and counts["target_positions"] == fit["exposures"]
        )
        expected = (2,) if arm == "aligned" else (1, 1)
        assert all(tuple(sorted(v.values())) == expected for v in local.values())
        signatures.append((inputs.hexdigest(), by_group, counts))
        del rows
    assert signatures[0] == signatures[1]
    record(
        PUBLIC / "exposure.json",
        dict(
            **counts,
            individual_inputs=len(local),
            ordered_input_sha=inputs.hexdigest(),
            equal_full_label_marginals=True,
            identical_replay=True,
        ),
    )


def paired():
    result = {}
    for group in ("credits", "old", "new", "books96", "books253"):
        stats = {}
        for arm in ("aligned", "independent", "parent"):
            path = WORK / f"{arm}_{group}_separate.json"
            expected = json.loads((PUBLIC / f"{arm}_evaluation.json").read_text())[
                "groups"
            ][group]["separate"]["detail_sha"]
            assert sha(path) == expected
            facts = defaultdict(list)
            for r in json.loads(path.read_text()):
                facts[r["scene"], r["lang"]].append(r["correct"])
            stats[arm] = {
                s: (int(all(facts[s, "zh"])) + int(all(facts[s, "en"]))) / 2
                for s, l in facts
            }
        result[group] = {}
        for ref in ("independent", "parent"):
            assert stats["aligned"].keys() == stats[ref].keys()
            d = np.array([a - stats[ref][s] for s, a in stats["aligned"].items()])
            rng = np.random.default_rng(241131)
            boot = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
            result[group][ref] = dict(
                whole_scenes=len(d),
                difference=float(d.mean()),
                interval=np.quantile(boot, [0.025, 0.975]).tolist(),
            )
    record(PUBLIC / "paired_transfer.json", result)


if __name__ == "__main__":
    exposure()
    paired()
