"""Apply pre-fit source adjudication and check for embedded training clauses."""

from collections import Counter

from scripts.prepare_predicate_likelihood import (
    OUT,
    EVENT,
    clauses,
    load,
    read_rows,
    sha,
    write_new,
)

REJECTIONS = {
    2: "The stream and reference disagree on the place name 螳川/蝗螂川.",
    3: "The gloss substitutes an ultimate spiritual realm for 物 without the surrounding argument.",
    8: "The distance fragment omits both armies supplied by its gloss.",
    10: "The isolated 此去梁国 fragment admits a journey or distance reading.",
    11: "The reference expands and renames the mountain in the first clause.",
    12: "The mulberry reference substitutes 入 for 人 and ends in an incomplete 来 phrase.",
    17: "The military gloss adds unsupported joint action and the felling gloss adds habitual frequency.",
    22: "The unusual 残伐乱宋 clause has a disputed segmentation given its short context.",
}


def admit():
    meta = load(OUT / "preparation.json")
    proposed = load(OUT / "proposed.json")
    assert sha(OUT / "proposed.json") == meta["proposed_sha"]
    train = read_rows("train")
    previous = {c for r in train for k in ("source", "new") for c in clauses(r[k])}
    for r in load(EVENT / "train.json"):
        for k in ("source", "target"):
            previous.update(clauses(r[k]))
    kept = []
    excluded = {}
    for i in range(0, len(proposed), 2):
        group = proposed[i : i + 2]
        reason = REJECTIONS.get(i // 2)
        if any(r["source"] in previous or r["target"] in previous for r in group):
            reason = (
                reason + " " if reason else ""
            ) + "Exact clause on an earlier training side."
        if reason:
            excluded[i // 2] = reason
        else:
            kept.extend(group)
    assert set(r["gloss"] for r in kept) == set(meta["training_counts"])
    write_new(OUT / "test.json", kept)
    admission = dict(
        test_sha=sha(OUT / "test.json"),
        train_sha=meta["training_sha"],
        accepted=len(kept),
        chapters=len({r["chapter"] for r in kept}),
        excluded_pairs=excluded,
        counts=dict(Counter(r["gloss"] for r in kept)),
        previous_complete_training_clauses=len(previous),
    )
    write_new(OUT / "admission.json", admission)
    print(admission, flush=True)


if __name__ == "__main__":
    admit()
