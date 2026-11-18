from scripts.prepare_contextual_phrase_realization import OUT, load, sha, write_new
from scripts.prepare_predicate_likelihood import read_rows, clauses

FIXED = [
    ("鲁人持斧伐柏", "鲁国人拿着斧头砍伐柏树", "鲁人举兵伐齐", "鲁国人出兵攻打齐国"),
    ("晋人入山伐松", "晋国人进山砍伐松树", "晋人出师伐郑", "晋国人出兵攻打郑国"),
    ("吴王遣人伐树", "吴王派人砍树", "吴王遣师伐越", "吴王派军队攻打越国"),
    ("齐师为舟伐竹", "齐军为了造船砍竹子", "齐师渡河伐卫", "齐军渡河攻打卫国"),
]


def admit():
    rows = load(OUT / "proposed.json")
    train = load(OUT / "train.json")
    kept = [
        dict(r, cohort="archive") for i, r in enumerate(rows) if i // 2 not in (2, 3)
    ]
    seen = {r[k] for r in train + rows for k in ("source", "target")}
    for r in read_rows("train") + read_rows("heldout"):
        for k in ("source", "old", "new"):
            seen.update(clauses(r[k]))
    for i, (s, t, u, v) in enumerate(FIXED):
        assert (
            len(s) == len(u)
            and s not in seen
            and u not in seen
            and s[:2] == u[:2]
            and s.index("伐") == u.index("伐")
        )
        for x, y, g in ((s, t, "砍伐"), (u, v, "攻打")):
            kept.append(
                dict(
                    source=x,
                    target=y,
                    gloss=g,
                    chapter=f"authored/{i}",
                    predicate="伐",
                    cohort="authored",
                )
            )
    write_new(OUT / "test.json", kept)
    write_new(
        OUT / "admission.json",
        dict(
            archive=len(kept) - 8,
            authored=8,
            removed_pairs={2: "unclear 滴滴 reading", 3: "unclear negative gloss"},
            train_sha=sha(OUT / "train.json"),
            test_sha=sha(OUT / "test.json"),
        ),
    )


if __name__ == "__main__":
    admit()
