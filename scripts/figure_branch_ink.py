"""Arrange the released readers' physical answers under two question and fact exchanges."""

from scripts.figure_recurrent_evidence import Plate, MUTED, TEAL, RUST
from scripts.prepare_branch_ink import PUB, MODEL
from scripts.prepare_open_exam import read, dump


def historical(assets):
    p = Plate("Figure_161_questions_about_historical_ink", assets)
    for case, top, title in (
        ("paired", 0, "Two periods of the character for raising"),
        ("yan", 139, "Two periods of the character for speech"),
    ):
        p.label(5, top + 1, "a" if case == "paired" else "b", 12, True)
        p.label(23, top + 2, title, 10, True)
        for j in range(2):
            d = MODEL / "examples/history" / case / str(j)
            r = read(d / "receipt.json")
            if j == 0:
                p.strip(
                    case + "-page",
                    d / "passage.png",
                    r["page_cells"],
                    8,
                    top + 22,
                    517,
                    20,
                )
                for x, t in (
                    (8, "Pictured request"),
                    (221, "Specimen ID"),
                    (326, "Reference"),
                    (428, "Written reply"),
                ):
                    p.label(x, top + 48, t, 8, True, MUTED)
            y = top + 63 + 31 * j
            p.strip(
                case + str(j) + "-q",
                d / "question.png",
                r["question_cells"],
                8,
                y + 2,
                203,
                17,
            )
            p.label(221, y + 5, r["label"], 9, True)
            p.strip(case + str(j) + "-r", d / "reference.png", 1, 326, y, 60, 25)
            p.strip(case + str(j) + "-a", d / "answer.png", r["cells"], 428, y, 95, 25)
    p.rule(130)
    return p.render()


def main():
    assets = PUB / "plates"
    assets.mkdir(parents=True, exist_ok=True)
    plate = Plate("Figure_160_distinguishing_the_requested_fact", assets)
    rows = {
        (r["case"], r["version"], r["query"], r["arm"]): r
        for r in read(PUB / "authored.json")
    }

    def strip(case, version, query, field, x, y, width, height, arm="branch"):
        r = rows[case, version, query, arm]
        source = MODEL / "examples" / case / str(version) / str(query)
        key = {"question": "question", "passage": "page", "reference": "expected"}.get(
            field
        )
        count = len(r[key]) if key else r["cells"]
        plate.strip(
            f"{case}-{version}-{query}-{field}",
            source / (field + ".png"),
            count,
            x,
            y,
            width,
            height,
        )

    plate.label(5, 2, "a", 12, True)
    plate.label(23, 3, "Which cabinet holds each object?", 10, True)
    for v, name in enumerate(("Original page", "After exchange")):
        y = 26 + 25 * v
        plate.label(8, y + 4, name, 8.6, True, MUTED)
        strip("shelf", v, 0, "passage", 100, y, 430, 20)
    plate.rule(80)
    for x, name in ((195, "Original locations"), (370, "Exchanged locations")):
        plate.label(x, 86, name, 9, True)
    for x, t in (
        (8, "Question image"),
        (195, "Control"),
        (263, "New writer"),
        (370, "Control"),
        (439, "New writer"),
    ):
        plate.label(x, 103, t, 8, True, TEAL if t == "New writer" else MUTED)
    for q in range(2):
        y = 120 + q * 29
        strip("shelf", 0, q, "question", 8, y, 176, 19)
        for v, col in enumerate((195, 370)):
            for arm, x in (("matched", col), ("branch", col + 68)):
                strip("shelf", v, q, arm, x, y, 58, 21, arm)
    plate.rule(179)
    plate.label(5, 187, "b", 12, True)
    plate.label(23, 188, "The same writer confuses two travellers", 10, True)
    strip("depart", 0, 1, "passage", 8, 205, 527, 16)
    for x, t in (
        (8, "Question"),
        (185, "Reference"),
        (300, "Control"),
        (423, "New writer"),
    ):
        plate.label(x, 229, t, 8, True, RUST if x == 423 else MUTED)
    for x, field, arm in (
        (8, "question", "branch"),
        (185, "reference", "branch"),
        (300, "matched", "matched"),
        (423, "branch", "branch"),
    ):
        strip("depart", 0, 1, field, x, 243, 150 if x == 8 else 90, 20, arm)
    dump(PUB / "plates.json", [plate.render(), historical(assets)])


if __name__ == "__main__":
    main()
