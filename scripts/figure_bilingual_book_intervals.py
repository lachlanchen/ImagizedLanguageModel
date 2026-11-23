"""Assemble Figure 107 from saved answers and the two question experiments."""

import json
from scripts.audit_interval_teacher import save_image
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.prepare_bilingual_book_intervals import PUBLIC, WORK
from scripts.reserve_bilingual_interval_books import PAGES, packets
from scripts.train_pictured_question_alignment import PUBLIC as QUESTIONS
from scripts.train_causal_glyph_writer import record
from scripts.train_typeface_writer import Face, FACES


def prepare():
    result = json.loads((PUBLIC / "result.json").read_text())["scores"]
    domain = json.loads((PUBLIC / "domains.json").read_text())
    scores = json.loads((QUESTIONS / "result.json").read_text())["scores"]
    arms = ["parent", "exchanged", "aligned"]
    series = []
    for a in arms:
        series.append(
            [
                round(100 * x["en"]["exact"] / x["en"]["n"], 1)
                for x in (
                    domain[a]["book"],
                    domain[a]["xquad_regular"],
                    result[a]["classical"],
                    result[a]["modern"],
                )
            ]
        )
    table = [["Question assignment", "Passages", "Length", "Exchanged", "Aligned"]]
    for name, key in (
        ("BiPaR novels", "book"),
        ("XQuAD", "xquad_regular"),
        ("Classical works", "classical"),
        ("Modern essays", "modern"),
    ):
        table.append(
            [
                name,
                str(scores["aligned"][key]["pairs"]),
                *[
                    str(scores[a][key]["wins"])
                    for a in ("length", "exchanged", "aligned")
                ],
            ]
        )
    pages = json.loads(PAGES.read_text())
    groups = packets()
    strips = []
    examples = []
    face = Face(FACES[0])
    for lang, good in (("zh", True), ("en", False)):
        for reg in ("classical", "modern"):
            name = reg + "_" + lang + ".npz"
            quoted, _ = load_streams(WORK / "evaluation" / "aligned" / name)
            outputs, ended = load_streams(WORK / "evaluation" / "writer_aligned" / name)
            candidates = groups[reg]["fresh_" + lang]
            index = next(
                (
                    i
                    for i, c in enumerate(candidates)
                    if bool(c.grade(outputs[i], ended[i])[2]) == good and outputs[i]
                ),
                None,
            )
            if index is None:
                continue
            c = candidates[index]
            k, j = c.question_id.split("/")
            row = pages[k]
            q = row["questions"][int(j)]
            for tag, raw, label in (
                ("question", c.q, lang.upper() + " question"),
                ("selected", quoted[index], "Selected excerpt"),
                ("reply", outputs[index], "Stopped writer reply"),
                ("ref", face(q["refs"][0]), "Accepted answer"),
            ):
                if good and tag == "ref":
                    continue
                file = lang + "_" + tag
                save_image(PUBLIC / (file + ".png"), raw)
                strips.append([file, label])
            examples.append(
                dict(
                    id=c.question_id,
                    lang=lang,
                    source=row["title"],
                    question=q[lang],
                    refs=q["refs"],
                    exact=good,
                )
            )
            break
    record(
        PUBLIC / "figure.json",
        dict(table=table, series=series, strips=strips, examples=examples),
    )


if __name__ == "__main__":
    prepare()
