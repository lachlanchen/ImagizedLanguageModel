"""Extend the frozen fresh-page readings to the changed sentence images."""

from hashlib import sha256
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_competing_facts import ROOT, PUB, read, dump

# Unique changed transcripts were read in original page order, without arm names.
# Exact raw-pixel equality transfers old judgments. A changed output is scored anew.
VERDICTS = {
    "911b41027c0b": "incorrect traditions",
    "d1df0d270db8": "wrong cause",
    "828d299cd596": "incomplete predicate",
    "17910e978e6c": "correct",
    "74a1d66c1780": "broken syntax",
    "2adb8ebef638": "correct",
    "7c67b19b3e28": "broken syntax",
    "9835c85b355c": "circular",
    "bdadbf9968ab": "wrong title",
    "e0b570624d43": "missing show",
    "953628809d70": "missing show",
    "1089a075b7b7": "wrong directive",
    "ed0718257bf3": "missing three items",
}


def main():
    prior = read(ROOT / "references/evidence/visual_statements_v1/fresh-reading.json")[
        "rows"
    ]["trained"]
    report = {}
    used = set()
    for arm in ("parent", "likelihood", "contrast"):
        fresh = {
            r["key"]: r
            for r in unpack(PUB / (arm + "-outputs.json.gz"))
            if r["test"] == "fresh96"
        }
        rows = []
        for old in prior:
            r = fresh[old["id"]]
            if r["answer_ink"] == old["shape_sha256"]:
                verdict = old["judgement"]
            else:
                assert arm != "parent"
                tag = sha256(r["answer"].encode()).hexdigest()[:12]
                verdict = VERDICTS[tag]
                used.add(tag)
            rows.append(
                dict(
                    id=old["id"],
                    sha256=r["answer_ink"],
                    verdict=verdict,
                    correct=verdict == "correct" and r["answer_stop"],
                )
            )
        report[arm] = dict(
            correct=sum(r["correct"] for r in rows), n=len(rows), rows=rows
        )
    assert used == set(VERDICTS)
    dump(
        PUB / "statement-readings.json",
        dict(
            method="First 24 earlier hash-ordered fresh pages. Preserve physically identical earlier answers. One model-based project reviewer classified 13 distinct changed transcripts with arm names omitted, after seeing aggregate results. Exploratory.",
            arms=report,
        ),
    )


if __name__ == "__main__":
    main()
