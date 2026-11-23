"""Prepare literal figure values and original reply strips from verified outputs."""

import json
import shutil

from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.train_bilingual_novel_writer import PUBLIC, WORK, opposite
from scripts.train_causal_glyph_writer import sha
from scripts.train_typeface_writer import Face, FACES


def prepare():
    result = json.loads((PUBLIC / "result.json").read_text())
    scores = result["scores"]
    arms = ("parent", "chinese_repeat", "english_exchanged", "bilingual")
    columns = ("book_zh", "book_en", "xquad_regular_zh", "xquad_regular_en")
    series = [
        [round(100 * scores[a][k]["exact"] / scores[a][k]["n"], 1) for k in columns]
        for a in arms
    ]
    table = [
        [
            "Complete passages",
            "Passages",
            "Parent",
            "Repeat ZH",
            "Swapped EN",
            "Aligned EN",
        ]
    ]
    for label, key in (
        ("BiPaR, four replies", "book_four"),
        ("XQuAD, four replies", "xquad_regular_four"),
        ("CMRC, two replies", "trial_regular"),
        ("Historical, two orders", "archive_familiar"),
    ):
        table.append(
            [label, str(scores["parent"][key]["boards"])]
            + [str(scores[a][key]["complete"]) for a in arms]
        )
    strips = []

    def strip(tag, lang, index, kind, label):
        name = f"{tag}_{lang}_{index}_{kind}"
        shutil.copyfile(PUBLIC / "examples" / (name + ".png"), PUBLIC / (name + ".png"))
        strips.append([name, label])

    if "book_success" in result["examples"]:
        strip("book_success", "en", 0, "question", "English question")
        strip("book_success", "en", 0, "answer", "Chinese reply")
        strip("book_success", "zh", 1, "question", "Second question")
        strip("book_success", "zh", 1, "answer", "Chinese reply")
    # The first incomplete board can still contain some exact answers.
    # Select its first actual error in English-then-Chinese, source order.
    cases = []
    for line in (WORK / "valid.jsonl").read_text().splitlines():
        qs = json.loads(line)["questions"]
        cases.extend(
            next(
                (a, b) for j, a in enumerate(qs) for b in qs[j + 1 :] if opposite(a, b)
            )
        )
    face = Face(FACES[0])
    ids = result["examples"]["book_miss"]
    found = False
    for lang in ("en", "zh"):
        name = "book_" + lang
        archive = WORK / "evaluation/bilingual" / (name + ".npz")
        assert sha(archive) == scores["bilingual"][name]["sha256"]
        ys, ends = load_streams(archive)
        for i, q in enumerate(cases):
            if q["id"] not in ids or (
                ends[i] and ys[i] in [face(v) for v in q["refs"]]
            ):
                continue
            k = ids.index(q["id"])
            strip("book_miss", lang, k, "question", "Missed question")
            strip("book_miss", lang, k, "answer", "Generated reply")
            strip("book_miss", lang, k, "reference", "Accepted answer")
            found = True
            break
        if found:
            break
    assert found
    (PUBLIC / "figure.json").write_text(
        json.dumps(dict(table=table, series=series, strips=strips), indent=2) + "\n"
    )


if __name__ == "__main__":
    prepare()
