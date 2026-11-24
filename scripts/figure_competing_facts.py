"""Print-size visual evidence for the model, its answer choices and the local benchmark."""

from scripts.figure_recurrent_evidence import Plate, TEAL, MUTED
from scripts.prepare_competing_facts import MODEL, PUB, ROOT, dump, read

ASSETS = PUB / "plates"


def strip(f, tag, path, image, n, x, y, w, h, lo=0, hi=None):
    f.strip(tag, path / (image + ".png"), n, x, y, w, h, lo, hi)


def mechanism():
    f = Plate("Figure_150_optical_reading_and_writing", ASSETS)
    root = ROOT / "models/visual_statements/examples/intervention"
    f.label(5, 0, "a", 12, True)
    f.label(23, 1, "Two learned stages connect a page to an answer", 10.6, True)
    for x, title, body in (
        (6, "Pictured prompt", "Question + passage"),
        (187, "Optical reader", "Locate and emit a fact"),
        (375, "Sentence writer", "Read fact and question"),
    ):
        f.band(x, 25, 161, 42)
        f.label(x + 6, 30, title, 9.4, True, TEAL)
        f.label(x + 6, 47, body, 8.2, False, MUTED)
    f.label(171, 38, "›", 13, True, MUTED)
    f.label(354, 38, "›", 13, True, MUTED)
    f.label(7, 78, "Who is credited on this page?", 8.7, True, MUTED)
    p = root / "locomotive-original"
    r = read(p / "readout.json")
    strip(f, "rail-q", p, "question", r["question_cells"], 193, 71, 335, 20)
    f.label(7, 103, "Name on page", 8.5, True, MUTED)
    f.label(184, 103, "Fact read", 8.5, True, MUTED)
    f.label(308, 103, "Generated sentence", 8.5, True, TEAL)
    fresh = read(ROOT / "artifacts/visual_statements_v1/fresh.json")[12]["source"]
    pos = fresh.index("Matthew Murray")
    for j, (k, who) in enumerate(
        (("original", "Matthew Murray"), ("edited", "Daniel Harper"))
    ):
        p = root / ("locomotive-" + k)
        r = read(p / "readout.json")
        y = 121 + 23 * j
        strip(
            f,
            "name-" + k,
            p,
            "passage",
            r["passage_cells"],
            7,
            y,
            165,
            17,
            pos,
            pos + len(who),
        )
        strip(f, "fact-" + k, p, "fact", r["fact_cells"], 184, y, 116, 17)
        strip(f, "answer-" + k, p, "answer", r["written"], 308, y, 227, 19)
    f.rule(172)
    f.label(5, 183, "b", 12, True)
    f.label(
        23, 184, "A date can survive reading and disappear during writing", 10.5, True
    )
    p = root / "school-original"
    r = read(p / "readout.json")
    strip(f, "school-q", p, "question", r["question_cells"], 7, 208, 320, 18)
    f.label(7, 232, "Fact read", 8.2, True, MUTED)
    f.label(163, 232, "Generated sentence", 8.2, True, TEAL)
    strip(f, "school-f", p, "fact", r["fact_cells"], 7, 247, 140, 19)
    strip(f, "school-out", p, "answer", r["written"], 163, 246, 372, 20)
    return f.render()


def reading():
    f = Plate("Figure_149_competing_facts", ASSETS)
    p = MODEL / "examples/squad/gain/1"
    r = read(p / "readout.json")
    f.label(5, 1, "a", 12, True)
    f.label(23, 2, "A question about a research network", 10.5, True)
    f.label(7, 25, "Question", 8.2, True, MUTED)
    strip(f, "nsf-q", p, "question", r["query_cells"], 113, 20, 420, 17)
    for y, a, label in (
        (47, "parent", "Original reader"),
        (71, "contrast", "Contrast-trained reader"),
    ):
        f.label(7, y + 2, label, 8.0, True, MUTED)
        strip(f, "nsf-" + a, p, a, r["outputs"][a]["cells"], 133, y - 2, 400, 18)
    f.label(7, 95, "Generated sentence", 8.1, True, TEAL)
    strip(
        f,
        "nsf-sentence",
        p,
        "contrast-sentence",
        r["outputs"]["contrast"]["answer_cells"],
        133,
        88,
        400,
        20,
    )
    f.rule(117)
    for j, (why, title) in enumerate(
        (
            ("gain", "A name needs its ordinal"),
            ("loss", "A period is replaced by a year"),
        )
    ):
        p = MODEL / "examples/drcd" / why / "0"
        r = read(p / "readout.json")
        y = 126 + j * 74
        f.label(5, y, "bc"[j], 12, True)
        f.label(23, y + 1, title, 10.5, True)
        strip(f, why + "-q", p, "question", r["query_cells"], 7, y + 19, 523, 18)
        for x, a, title in (
            (7, "parent", "Original"),
            (230, "contrast", "Contrast-trained"),
        ):
            f.label(x, y + 47, title, 8.2, True, MUTED)
            strip(
                f,
                why + "-" + a,
                p,
                a,
                r["outputs"][a]["cells"],
                x + (56 if a == "parent" else 96),
                y + 40,
                145,
                20,
            )
    return f.render()


def boundaries():
    f = Plate("Figure_152_endpoint_interaction", ASSETS)
    examples = ROOT / "models/endpoint_interaction/examples"
    for tag, y, panel, title in (
        ("tower", 0, "a", "An answer loses one of two film categories"),
        ("title", 132, "b", "A title gains its closing mark"),
    ):
        path = examples / tag
        r = read(path / "readout.json")
        f.label(5, y + 2, panel, 12, True)
        f.label(23, y + 3, title, 10.7, True)
        f.band(5, y + 24, 534, 27)
        strip(f, tag + "-q", path, "question", r["question_cells"], 9, y + 28, 524, 20)
        for j, (arm, label) in enumerate(
            (("additive", "Separate endpoints"), ("product", "Interacting endpoints"))
        ):
            f.label(7, y + 62 + 27 * j, label, 8.5, True, MUTED)
            strip(
                f,
                tag + "-" + arm,
                path,
                arm,
                r["outputs"][arm]["cells"],
                143,
                y + 55 + 27 * j,
                388,
                20,
            )
    f.rule(118)
    return f.render()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    dump(PUB / "plates.json", [mechanism(), reading(), boundaries()])


if __name__ == "__main__":
    main()
