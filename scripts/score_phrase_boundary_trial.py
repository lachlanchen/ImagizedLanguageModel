"""Combine frozen complete-answer readings with the concealed method assignments."""

import csv

from scripts.prepare_phrase_boundary_trial import EVIDENCE, load, sha, write_new


def run():
    tests = load(EVIDENCE / "test.json")
    packet = load(EVIDENCE / "review.json")
    keys = load(EVIDENCE / "mapping.json")
    with (EVIDENCE / "reading.tsv").open() as f:
        grades = list(csv.DictReader(f, delimiter="\t"))
    by = {}
    for j, (g, r, k) in enumerate(zip(grades, packet, keys, strict=True)):
        assert int(g["review"]) == j and len(g["grades"]) == len(r["choices"])
        i = k["row"]
        for method, opt in k["options"].items():
            by.setdefault(method, {})[i] = g["grades"][opt]
    metrics = {}
    for method, group in by.items():
        rec = {}
        for cohort in ("archive_pair", "authored_pair", "archive_single", "broad"):
            order = sorted(
                i
                for i in group
                if (tests[i]["cohort"] if i < len(tests) else "broad") == cohort
            )
            if not order:
                continue
            answers = [group[i] for i in order]
            rec[cohort] = dict(n=len(answers), faithful=answers.count("F"))
            if cohort != "broad":
                rec[cohort]["sense"] = sum(v != "X" for v in answers)
            if cohort.endswith("_pair"):
                assert all(
                    tests[i]["pair"] == tests[j]["pair"]
                    for i, j in zip(order[::2], order[1::2])
                )
                rec[cohort]["both_faithful"] = sum(
                    x == y == "F" for x, y in zip(answers[::2], answers[1::2])
                )
        metrics[method] = rec
    write_new(
        EVIDENCE / "scores.json",
        dict(
            reading_sha=sha(EVIDENCE / "reading.tsv"),
            mapping_sha=sha(EVIDENCE / "mapping.json"),
            scores=metrics,
        ),
    )
    for m, r in metrics.items():
        print(m, r)


if __name__ == "__main__":
    run()
