"""Recount masked readings and compute paired complete-answer and semantic-detail scores."""

import csv
from scripts.train_screened_visual_targets import EVIDENCE, load, write_new, sha


def run():
    reg = load(EVIDENCE / "registration.json")
    ev = load(EVIDENCE / "evaluation.json")["records"]
    d = EVIDENCE / "reading"
    masked = load(d / "review.json")
    mapping = load(d / "mapping.json")
    with (d / "credits.tsv").open() as f:
        grades = list(csv.DictReader(f, delimiter="\t"))
    assert len(masked) == len(mapping) == len(grades) and sorted(
        v["row"] for v in mapping
    ) == sorted(reg["reader"])
    rows = reg["rows"]
    a = reg["archive"]
    scores = {}
    for model, v in ev.items():
        whole = set()
        detail = set()
        for j, (b, row, grade) in enumerate(zip(mapping, masked, grades)):
            assert j == int(grade["j"])
            for name, passed in (("whole", whole), ("detail", detail)):
                ids = {int(i) for i in grade[name].split(",") if i != "-"}
                assert ids <= set(range(len(row["choices"])))
                if b["options"][model] in ids:
                    passed.add(b["row"])
        paired = lambda approved: [
            i
            for i in range((len(rows) - a) // 2)
            if {a + 2 * i, a + 2 * i + 1} <= approved
        ]
        scores[model] = dict(
            archive_edits=sum(v["edits"][:a]),
            archive_whole=sorted(i for i in whole if i < a),
            contrast_whole=sorted(i for i in whole if i >= a),
            contrast_detail=sorted(detail),
            whole_pairs=paired(whole),
            detail_pairs=paired(detail),
            archive_capped=a - sum(v["stopped"][:a]) if "stopped" in v else None,
        )
    write_new(
        EVIDENCE / "scores.json",
        dict(
            scores=scores,
            readings_sha=sha(d / "credits.tsv"),
            mapping_sha=sha(d / "mapping.json"),
        ),
    )
    for k, v in scores.items():
        print(
            k,
            v["archive_edits"],
            len(v["archive_whole"]),
            len(v["contrast_whole"]),
            v["whole_pairs"],
            v["detail_pairs"],
            v["archive_capped"],
        )


if __name__ == "__main__":
    run()
