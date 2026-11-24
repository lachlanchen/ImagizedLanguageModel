"""Typeset original answer ink and the measured distinction between pairing and writing."""

from scripts.figure_recurrent_evidence import Plate, MUTED, TEAL, RUST
from scripts.prepare_complete_fact_sets import MODEL, PUB
from scripts.prepare_multiple_ink_runs import read, dump
from scripts.prepare_fact_binding import PUB as BIND

ASSETS = PUB / "plates"


def output_plate():
    f = Plate("Figure_155_complete_answer_ink", ASSETS)

    def strip(case, arm, file, count, y, x, width, height, lo=0, hi=None):
        d = MODEL / "examples" / case
        f.strip(
            f"{case}-{arm}-{file}-{lo}",
            (d / arm if arm else d) / (file + ".png"),
            count,
            x,
            y,
            width,
            height,
            lo,
            hi,
        )

    poem = MODEL / "examples/poem"
    r = read(poem / "contrast/result.json")
    f.label(5, 2, "a", 12, True)
    f.label(23, 3, "Which lines from the Li Sao does the reader select?", 10, True)
    f.label(6, 27, "Question", 8, True, MUTED)
    strip("poem", "", "question", r["question_cells"], 40, 6, 140, 17)
    f.label(215, 27, "Likelihood control  ·  entire answer", 8, True, MUTED)
    strip(
        "poem",
        "control",
        "answer",
        read(poem / "control/result.json")["answer_cells"],
        40,
        215,
        210,
        17,
    )
    f.band(5, 65, 534, 69)
    f.label(
        10, 70, "Contrast-trained reader  ·  first four of sixteen parts", 8, True, TEAL
    )
    ends = sum(map(len, r["parts"][:4])) + 9
    strip("poem", "contrast", "answer", r["answer_cells"], 83, 10, 508, 18, 0, ends)
    f.label(10, 111, "Final part", 8, True, RUST)
    strip(
        "poem",
        "contrast",
        "answer",
        r["answer_cells"],
        106,
        90,
        425,
        18,
        r["answer_cells"] - len(r["parts"][-1]),
    )
    f.rule(144)
    f.label(5, 153, "b", 12, True)
    f.label(
        23,
        154,
        "Two correct composers, followed by an error in the written statement",
        10,
        True,
    )
    m = read(MODEL / "examples/music/contrast/result.json")
    f.label(6, 178, "Question", 8, True, MUTED)
    strip("music", "", "question", m["question_cells"], 190, 6, 326, 14)
    f.label(348, 178, "Entire list", 8, True, TEAL)
    strip("music", "contrast", "answer", m["answer_cells"], 190, 348, 188, 14)
    f.rule(213)
    f.label(6, 224, "Complete stopped sentence", 8, True, MUTED)
    strip("music", "contrast", "sentence", m["sentence_cells"], 239, 6, 529, 17)
    return f.render()


def measures():
    f = Plate("Figure_156_binding_and_free_writing", ASSETS)
    d = read(BIND / "scores.json")
    arms = ("parent", "linear", "mixture")
    cols = (
        ("#94A7A2", "Frozen reader"),
        ("#B46C49", "Additive head"),
        (TEAL, "Two readings"),
    )
    f.label(5, 3, "a", 12, True)
    f.label(
        23, 4, "When the two answer lists and their exchanges are supplied", 10, True
    )
    for i, (col, name) in enumerate(cols):
        f.items.append(dict(type="rect", x=8 + 166 * i, y=28, w=8, h=8, color=col))
        f.label(22 + 166 * i, 27, name, 8.4)
    for row, (suffix, txt) in enumerate(
        (("", "Original question"), ("-exchanged", "Unrelated question"))
    ):
        y = 57 + row * 51
        f.label(6, y + 11, txt, 8.7, True)
        for i, (arm, (col, _)) in enumerate(zip(arms, cols)):
            n = d[arm]["witness" + suffix]["witness"]["both"]
            yy = y + i * 11
            if n:
                f.items.append(dict(type="rect", x=135, y=yy, w=7 * n, h=7, color=col))
            f.label(140 + 7 * n, yy - 1, str(n), 7.1, False, col)
    f.label(326, 69, "Both original lists score above their", 8.6)
    f.label(326, 82, "respective exchanges on the same page.", 8.6)
    f.label(326, 110, "64 held Chinese questions per condition.", 8, False, MUTED)
    f.rule(153)
    f.label(5, 162, "b", 12, True)
    f.label(23, 163, "When the model must find and write every part itself", 10, True)
    names = ("New Chinese pages", "English plural", "Chinese plural", "Chinese single")
    for i, (label, den) in enumerate(zip(names, (64, 64, 96, 64))):
        f.label(116 + 108 * i, 187, label, 7.1, True, MUTED)
        f.label(116 + 108 * i, 198, f"n = {den}", 7, False, MUTED)
    for j, (arm, (col, label)) in enumerate(zip(arms, cols)):
        y = 215 + 16 * j
        f.label(7, y, label, 8.3, True, col)
        vals = [d[arm]["witness"]["witness"]["exact"]] + [
            d[arm]["confirmation"][group]["exact"]
            for group in ("en-multi", "zh-multi", "zh-single")
        ]
        for i, v in enumerate(vals):
            f.label(116 + 108 * i, y, str(v), 8.7, True)
    return f.render()


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    a = output_plate()
    measures()
    dump(PUB / "plates.json", [a])
