"""Seal a second set of natural passage pairs for the likelihood comparison."""

from collections import defaultdict
from pathlib import Path
from scripts.prepare_region_writer import (
    ROOT,
    inventory,
    squad,
    statements,
    phrase_key,
    digest,
    read,
    dump,
    sha,
)
from scripts.prepare_phrase_confirmation import WORK as PREVIOUS, exam_rows

WORK = ROOT / "artifacts/phrase_marginal_v1"
PUB = ROOT / "references/evidence/phrase_marginal_v1"


def main():
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    file = WORK / "confirmation.json"
    receipt = PUB / "confirmation.json"
    if receipt.is_file() and file.is_file():
        if read(receipt)["sha256"] != sha(file):
            raise ValueError("Changed held data")
        return
    train, prior = inventory()
    used = train + prior + read(PREVIOUS / "confirmation.json")
    families = {r["family"] for r in used}
    questions = {phrase_key(r["q"]) for r in used}
    targets = {phrase_key(a) for r in used for a in r["answers"]}
    pages = {k: p for fold in (0, 1) for k, p, _, _ in squad(fold)}
    results = []
    counts = {}
    for source, task in (
        (statements("dev", pages), "statements"),
        (exam_rows(), "exam"),
    ):
        groups = defaultdict(list)
        for r in source:
            bs = r["blocks"]
            if (
                r["family"] not in families
                and phrase_key(r["q"]) not in questions
                and not any(phrase_key(a) in targets for a in r["answers"])
                and all(0 < len(a) < 256 for a in r["answers"])
                and 0 < len(r["q"]) <= 192
                and 0 < len(bs) <= 24
                and max(map(len, bs)) <= 256
                and sum(map(len, bs)) <= 1536
            ):
                groups[r["family"]].append(r)
        count = 0
        for f in sorted(groups, key=lambda f: digest("marginal-family" + f)):
            pool = sorted(groups[f], key=lambda r: digest("marginal-query" + r["q"]))
            first = pool[0]
            second = next(
                (
                    r
                    for r in pool[1:]
                    if phrase_key(r["q"]) != phrase_key(first["q"])
                    and phrase_key(r["facts"]["answer"])
                    != phrase_key(first["facts"]["answer"])
                    and r["blocks"] == first["blocks"]
                ),
                None,
            )
            if second is None:
                continue
            results.extend([first, second])
            count += 1
            if count == 96:
                break
        counts[task] = count
    assert counts == {"statements": 96, "exam": 96}
    dump(file, results)
    measured = dict(
        families=counts,
        sha256=sha(file),
        excluded_prior=sha(PREVIOUS / "confirmation.json"),
        source=sha(Path(__file__)),
    )
    if receipt.exists() and read(receipt) != measured:
        raise ValueError("Reservation differs")
    dump(receipt, measured)
    print(measured)


if __name__ == "__main__":
    main()
