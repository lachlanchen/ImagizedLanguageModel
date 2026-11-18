"""Export complete native model output strips and the counts used by Figure 66."""

import csv
from scripts.evaluate_anchor_gap_writer import (
    DATA,
    OUT,
    ROSTER,
    EVIDENCE,
    load,
    read_streams,
    write_new,
)
from scripts.infer_visual_translation import page


def run():
    scores = load(EVIDENCE / "scores.json")
    finite = load(EVIDENCE / "finite.json")
    old = list(csv.DictReader((EVIDENCE / "readings.tsv").open(), delimiter="\t"))
    grades = {
        r["row"]: r["choice"]
        in [
            int(v)
            for v in old[r["packet"]]["credited_zero_based_choices"].split(",")
            if v != "-"
        ]
        for r in finite["reused"]
    }
    new = load(EVIDENCE / "finite_review.json")
    with (EVIDENCE / "finite_readings.tsv").open() as f:
        readings = list(csv.DictReader(f, delimiter="\t"))
    assert [int(r["row"]) for r in readings] == [r["row"] for r in new]
    for r in readings:
        assert r["credit"] in ("0", "1")
        grades[int(r["row"])] = bool(int(r["credit"]))
    assert set(grades) == set(finite["review_rows"])
    faithful = sorted(k for k, v in grades.items() if v)
    rows = load(ROSTER)["rows"]
    gen = load(EVIDENCE / "generation.json")
    finite_bands = [
        sum(v for v, r in zip(finite["edits"], rows) if r["length_band"] == band)
        for band in range(4)
    ]
    scores["scores"]["finite"] = dict(
        totals=dict(edits=sum(finite["edits"])),
        bands=finite_bands,
        exact=finite["edits"].count(0),
    )
    scores["faithful_counts"]["finite"] = len(faithful)
    write_new(
        EVIDENCE / "finite_scores.json",
        dict(
            faithful=faithful,
            edits=sum(finite["edits"]),
            edits_by_length=finite_bands,
            exact=finite["edits"].count(0),
            same_answers=len(finite["reused"]),
            new_readings=len(new),
        ),
    )
    table = [["Output", "Glyph edits", "Exact / 512", "Faithful / 24", "At length cap"]]
    for k, label in [
        ("source", "Copy input"),
        ("composer", "Phrase composer"),
        ("learned", "Learned positions"),
        ("all", "All positions"),
        ("supplied", "Supplied positions¹"),
        ("finite", "Finite gaps²"),
    ]:
        v = scores["scores"][k]
        table.append(
            [
                label,
                f'{v["totals"]["edits"]:,}',
                str(v["exact"]),
                str(scores["faithful_counts"][k]),
                str(512 - v["totals"]["stop"]) if "stop" in v["totals"] else "—",
            ]
        )
    streams = {
        "source": read_streams("source", DATA / "controls"),
        "learned": read_streams("learned", OUT),
        "finite": read_streams("finite", OUT),
    }
    widths = {}
    for i in (45, 184, 6, 105):
        for k, st in streams.items():
            key = f"{i}_{k}"
            assert 0 < len(st[i]) <= 32
            page(st[i]).save(EVIDENCE / (key + ".png"))
            widths[key] = len(st[i])
    series = [scores["scores"][k]["bands"] for k in ("learned", "composer", "finite")]
    write_new(EVIDENCE / "figure.json", dict(widths=widths, table=table, series=series))


if __name__ == "__main__":
    run()
