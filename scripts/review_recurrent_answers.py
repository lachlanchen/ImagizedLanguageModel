"""Create a reproducible masked semantic audit and score its frozen judgements."""

import gzip
import hashlib
import json
import random
from scripts.prepare_recurrent_evidence import WORK, PUBLIC, read, dump
from scripts.evaluate_recurrent_evidence import lexical


def prepare():
    held = read(WORK / "held.json")
    excluded = {0, 1, 2, 3, 51, 104, 121, 128, 129, 130, 131, 138, 160, 200}
    arms = ("previous", "fixed", "dynamic", "sentence")
    predictions = {
        a: json.loads(
            gzip.decompress((WORK / (a + "-private-answers.json.gz")).read_bytes())
        )["answers"]
        for a in arms[:-1]
    }
    rng = random.Random(871004)
    pages = []
    key = []
    for lang in ("zh", "en"):
        rows = sorted(
            (
                i
                for i, r in enumerate(held)
                if r["lang"] == lang and r["subset"] == "absent" and i not in excluded
            ),
            key=lambda i: hashlib.sha256(
                ("meaning-audit-1 " + held[i]["id"]).encode()
            ).digest(),
        )[:8]
        for i in rows:
            r = held[i]
            choices = {}
            for a in arms:
                answer = lexical(r) if a == "sentence" else predictions[a][i]["output"]
                choices.setdefault(answer, []).append(a)
            unique = list(choices)
            rng.shuffle(unique)
            pages.append(
                dict(
                    id=r["id"],
                    lang=lang,
                    question=r["question"],
                    source=r["source"],
                    references=r["refs"],
                    candidates={chr(65 + j): v for j, v in enumerate(unique)},
                )
            )
            key.append({chr(65 + j): choices[v] for j, v in enumerate(unique)})
    dump(WORK / "blind.json", pages)
    dump(WORK / "unmask.json", key)


def score():
    pages = read(WORK / "blind.json")
    keys = read(WORK / "unmask.json")
    judgements = read(PUBLIC / "meaning-readings.json")
    totals = {}
    full = []
    for r, key, grade in zip(pages, keys, judgements, strict=True):
        if grade["id"] != r["id"] or set(grade["grades"]) != set(key):
            raise ValueError("Incomplete judgement")
        for c, arms in key.items():
            val = grade["grades"][c]
            if val not in ("correct", "partial", "wrong", "undetermined"):
                raise ValueError("Unknown grade")
            for a in arms:
                dst = totals.setdefault(r["lang"], {}).setdefault(
                    a, dict(correct=0, partial=0, wrong=0, undetermined=0)
                )
                dst[val] += 1
                if val == "correct":
                    full.append(dict(id=r["id"], arm=a, lang=r["lang"]))
    dump(
        PUBLIC / "meaning-audit.json",
        dict(
            counts=totals,
            correct=full,
            blind_sha256=hashlib.sha256((WORK / "blind.json").read_bytes()).hexdigest(),
            grades_sha256=hashlib.sha256(
                (PUBLIC / "meaning-readings.json").read_bytes()
            ).hexdigest(),
        ),
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["score"]:
        score()
    else:
        prepare()
