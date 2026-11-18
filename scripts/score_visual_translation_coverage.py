"""Recover per-method scores only after all masked semantic decisions are frozen."""

import csv

from scripts.train_visual_translation_coverage import DEST, load, write_new, sha


def score():
    folder = DEST / "reading"
    assessment = load(DEST / "fresh_assessment.json")
    packet = load(folder / "review.json")
    mapping = load(folder / "mapping.json")
    with (folder / "credits.tsv").open() as f:
        grades = list(csv.DictReader(f, delimiter="\t"))
    natural = set(assessment["reading"])
    pairs = [(r["row"], 192 + i) for i, r in enumerate(assessment["number_pairs"])]
    selected = natural.union(*pairs)
    assert (
        len(packet) == len(grades) == len(mapping) == len(selected)
        and {b["row"] for b in mapping} == selected
    )
    results = {}
    for method, v in load(DEST / "generation.json")["methods"].items():
        whole, detail = set(), set()
        for j, (b, row, g) in enumerate(zip(mapping, packet, grades)):
            assert int(g["j"]) == j
            for name, passed in (("whole", whole), ("detail", detail)):
                ids = {int(i) for i in g[name].split(",") if i != "-"}
                assert ids <= set(range(len(row["choices"])))
                if b["options"][method] in ids:
                    passed.add(b["row"])
        results[method] = dict(
            natural_correct=sorted(whole & natural),
            both_meanings=[i for i, (a, b) in enumerate(pairs) if {a, b} <= whole],
            both_quantities=[i for i, (a, b) in enumerate(pairs) if {a, b} <= detail],
            archive_edits=sum(v["edits"][:192]),
            emitted=sum(n > 0 for n in v["lengths"]),
            stopped=sum(v["stopped"]) if "stopped" in v else None,
        )
    write_new(
        DEST / "scores.json",
        dict(
            scores=results,
            grades_sha=sha(folder / "credits.tsv"),
            packet_sha=sha(folder / "review.json"),
            mapping_sha=sha(folder / "mapping.json"),
        ),
    )


if __name__ == "__main__":
    score()
