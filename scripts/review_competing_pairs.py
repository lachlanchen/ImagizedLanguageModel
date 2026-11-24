"""Read all nonmatching strings on pages where the two fitted readers disagree."""

from collections import Counter
from hashlib import sha256
from scripts.prepare_competing_facts import PUB, dump
from scripts.evaluate_visual_discourse import unpack

# Read against the original question, visible passage and human answers with
# model labels omitted. Recoverable includes a right name with a broken edge.
GUIDE = {
    "clear": set(
        "4f5105338a 5d6320ecf2 cc038c9e6e f8730e17b5 4e69299dbe ce5da8106c d86d752de8 b44d61b9f4 4d53327301".split()
    ),
    "recoverable": set(
        "951261bac2 58d1f9d89d 18eda8fea7 70fab5e6c1 aa206cf0b7".split()
    ),
    "wrong_or_incomplete": set(
        "08d9828689 4654093b01 5f48947ac6 0b0d494bb6 aff662505b 8edf699b8c 4838f2249d 9ac7be9af4 65c64580cb d55fda8e86 2ceb042771 5bd6f86fe6 0c9e8d2177 c640daebd6 2c29ddad36 28e425265f 715018028f ed028abe7d".split()
    ),
}


def main():
    arms = ("parent", "contrast")
    states = {
        a: [r for r in unpack(PUB / (a + "-outputs.json.gz")) if r["test"] == "human"]
        for a in arms
    }
    looked = set()
    rows = []
    counts = {a: Counter() for a in arms}
    for i in range(0, len(states["parent"]), 2):
        pair = {a: states[a][i : i + 2] for a in arms}
        if all(r["correct"] for r in pair["parent"]) == all(
            r["correct"] for r in pair["contrast"]
        ):
            continue
        for a, items in pair.items():
            labels = []
            for r in items:
                status = "exact"
                if not r["correct"]:
                    tag = sha256((r["key"] + "|" + r["fact"]).encode()).hexdigest()[:10]
                    (status,) = [k for k, v in GUIDE.items() if tag in v]
                    looked.add(tag)
                labels.append(status)
                rows.append(
                    dict(arm=a, key=r["key"], sha256=r["physical"], status=status)
                )
            counts[a]["both_exact"] += all(l == "exact" for l in labels)
            counts[a]["both_clear"] += all(l in ("exact", "clear") for l in labels)
            counts[a]["both_identifiable"] += all(
                l != "wrong_or_incomplete" for l in labels
            )
    assert looked == set.union(*GUIDE.values())
    dump(
        PUB / "disagreement-readings.json",
        dict(
            pages=len(rows) // 4,
            unique_nonmatches=len(looked),
            counts=counts,
            rows=rows,
            scope="All exact-score flips only; neither representative accuracy nor independent human adjudication. Clear permits an unambiguous answer with natural extra context; identifiable additionally permits broken or underspecified surface wording.",
        ),
    )


if __name__ == "__main__":
    main()
