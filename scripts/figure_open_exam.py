"""Print the observed optical answers, question controls and crossed historical cards."""

from scripts.figure_recurrent_evidence import Plate, MUTED, TEAL, RUST
from scripts.prepare_open_exam import MODEL, PUB, read, dump
from scripts.prepare_page_competition import MODEL as LISTS

ASSETS = PUB / "plates"


def competition():
    f = Plate("Figure_157_whole_page_competition", ASSETS)
    case = LISTS / "examples/186530"
    f.label(5, 2, "a", 12, True)
    f.label(
        23, 3, "Which compounds illustrate the readings of this character?", 10, True
    )
    f.label(7, 29, "Question", 8, True, MUTED)
    f.strip("question", case / "question.png", 7, 7, 43, 140, 18)
    f.label(165, 29, "Original reader  ·  beginning of the list", 8, True, MUTED)
    f.strip("parent", case / "parent/answer.png", 68, 165, 43, 363, 18, 0, 25)
    f.band(5, 73, 534, 54)
    for i, (arm, lab) in enumerate(
        [("control", "Likelihood control"), ("competition", "Whole-page competition")]
    ):
        x = 13 + 266 * i
        f.label(x, 80, lab + "  ·  entire answer", 8.5, True, TEAL)
        f.strip(arm, case / arm / "answer.png", 2, x, 96, 85, 23)
    f.label(
        7,
        138,
        "Both readers recovered one compound from a valid seventeen-compound list.",
        8.8,
    )
    f.rule(157)
    f.label(5, 167, "b", 12, True)
    f.label(23, 168, "What changed on held Chinese pages?", 10, True)
    for x, txt in ((270, "Likelihood"), (382, "Competition")):
        f.label(x, 190, txt, 8.5, True, MUTED)
    for y, (name, a, b) in enumerate(
        [
            ("Median winning score minus reference", "14.013", "4.481"),
            ("Correct parts against first annotation", "27 / 729", "50 / 729"),
            ("Complete multi-part answers", "0 / 112", "0 / 112"),
        ]
    ):
        yy = 209 + y * 17
        f.label(7, yy, name, 8.8)
        f.label(270, yy, a, 9, True)
        f.label(382, yy, b, 9, True, TEAL)
    return f.render()


def writing():
    f = Plate("Figure_158_question_conditioned_writing", ASSETS)
    f.label(5, 2, "a", 12, True)
    f.label(23, 3, "Changing a destination changes the written answer", 10, True)
    for i, ver in enumerate(("original", "changed")):
        d = MODEL / "examples/travel" / ver
        x = 7 + 268 * i
        f.label(x, 27, "Page fragment", 8, True, MUTED)
        f.strip(ver + "-p", d / "passage.png", 26, x, 41, 205, 17, 0, 8)
        f.label(x, 67, "Same question", 8, True, MUTED)
        f.strip(ver + "-q", d / "question.png", 7, x, 81, 128, 16)
        f.label(x + 157, 67, "Written reply", 8, True, TEAL)
        f.strip(ver + "-a", d / "matched/answer.png", 2, x + 157, 79, 87, 21)
    f.rule(110)
    f.label(5, 118, "b", 12, True)
    f.label(23, 119, "One passage, two meanings", 10, True)
    d = MODEL / "examples/speech/original"
    f.strip("speech-p", d / "passage.png", 22, 8, 140, 347, 17)
    f.label(7, 166, "Question", 8, True, MUTED)
    f.strip("speech-q", d / "question.png", 7, 7, 178, 167, 18)
    f.label(181, 166, "Written reply", 8, True, RUST)
    f.strip("speech-a", d / "matched/answer.png", 2, 181, 178, 75, 18)
    f.label(276, 170, "The reply gives the gloss for water", 8.4)
    f.label(276, 183, "when asked about speech.", 8.4)
    f.band(5, 211, 534, 53)
    vals = read(PUB / "results.json")["matched"]
    for i, (name, cond) in enumerate(
        [
            ("Original inputs", "intact"),
            ("Question exchanged", "other_question"),
            ("Page exchanged", "other_page"),
        ]
    ):
        x = 15 + i * 177
        f.label(x, 217, name, 8, True, MUTED)
        f.label(x, 233, f'{vals[cond]["free"]} / 384', 14, True, TEAL)
    return f.render()


def history():
    f = Plate("Figure_159_where_the_mark_appears", ASSETS)
    f.label(5, 2, "a", 12, True)
    f.label(23, 3, "Two period names on the same dictionary card", 10, True)
    base = MODEL / "examples/first-history"
    r = read(base / "1/receipt.json")
    f.strip("card", base / "1/passage.png", r["page_cells"], 7, 25, 520, 21)
    for x, t in (
        (7, "Question"),
        (245, "Attested form"),
        (380, "Entire written reply"),
    ):
        f.label(x, 62, t, 8.3, True, MUTED)
    for j in range(2):
        d = base / str(j + 1)
        r = read(d / "receipt.json")
        y = 77 + j * 35
        f.strip(
            "q" + str(j), d / "question.png", r["question_cells"], 7, y + 3, 226, 16
        )
        f.strip("ref" + str(j), d / "reference.png", 1, 245, y, 53, 25)
        f.strip("output" + str(j), d / "answer.png", r["cells"], 380, y, 150, 25)
    f.rule(151)
    f.label(5, 160, "b", 12, True)
    f.label(23, 161, "Separate exchanges of labels and specimens", 10, True)
    m = read(PUB / "stage-order.json")["summary"]
    for i, t in enumerate(
        ("Requested mark", "First position", "Second position", "All four arrangements")
    ):
        f.label((7, 161, 287, 411)[i], 188, t, 7.7, True, MUTED)
    for i, (lang, name) in enumerate(
        (("zh", "Chinese question"), ("en", "English question"))
    ):
        y = 208 + i * 20
        a, b = m[lang]["correct_by_position"]
        for x, t in zip(
            (7, 161, 287, 411),
            (name, f"{a} / 256", f"{b} / 256", f'{m[lang]["all_four"]} / 128'),
        ):
            f.label(x, y, t, 8.8, True, TEAL if i == 0 else MUTED)
    f.label(
        7,
        253,
        "Each question is evaluated on every arrangement. A correct reply must stop after the mark.",
        8,
    )
    return f.render()


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    dump(PUB / "plates.json", [competition(), writing(), history()])
