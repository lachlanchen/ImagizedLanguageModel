"""Assemble the measured evidence scores and the first complete classical example."""

import json
from scripts.audit_interval_teacher import save_image
from scripts.reserve_mlqa_pictures import PUBLIC, WORK
from scripts.reserve_bilingual_interval_books import packets, PAGES
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_page_evidence_writer import tensor
from scripts.mine_stopped_book_replies import PARENT
from scripts.train_causal_glyph_writer import record
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
import torch


def prepare():
    r = json.loads((PUBLIC / "result.json").read_text())["scores"]
    writer = json.loads((PUBLIC / "writer.json").read_text())
    arms = ("parent", "exchanged", "matched")
    series = [
        [
            round(100 * r[a][d][l]["exact"] / r[a][d][l]["n"], 1)
            for d, l in (("mlqa", "en"), ("mlqa", "zh"), ("book", "en"), ("book", "zh"))
        ]
        for a in arms
    ]
    table = [
        ["MLQA input to writer", "Chinese / 412", "English / 412", "All four / 32"]
    ]
    for a, label in (
        ("full", "Original page"),
        ("parent", "Parent excerpt"),
        ("exchanged", "Exchanged excerpt"),
        ("matched", "Matched excerpt"),
    ):
        w = writer[a]
        table.append(
            [
                label,
                str(w["zh"]["exact"]),
                str(w["en"]["exact"]),
                str(w["four"]["exact"]),
            ]
        )
    groups = packets()["classical"]
    inputs = {l: groups["fresh_" + l] for l in ("en", "zh")}
    images = {l: load_streams(WORK / f"classical_{l}_matched.npz")[0] for l in inputs}
    start = next(
        i
        for i in range(0, len(inputs["zh"]), 2)
        if all(
            inputs[l][j].grade(images[l][j], True)[2]
            for l in inputs
            for j in (i, i + 1)
        )
    )
    pages = json.loads(PAGES.read_text())
    key = inputs["zh"][start].board
    rows = []
    strips = []
    torch.set_num_threads(2)
    net = PreparedGlyphResponder(PARENT, "cpu")
    directory = PUBLIC / "examples"
    directory.mkdir(exist_ok=True)
    save_image(directory / "page.png", inputs["en"][start].p)
    for i in (start, start + 1):
        for lang in ("zh", "en"):
            c = inputs[lang][i]
            y = images[lang][i]
            result, stop = net.answer_many([(tensor(y), tensor(c.q))])[0]
            raw = result.numpy().tobytes()
            tag = f"{lang}{i-start+1}"
            save_image(directory / (tag + "_question.png"), c.q)
            save_image(directory / (tag + "_excerpt.png"), y)
            save_image(directory / (tag + "_written.png"), raw)
            if lang == "en":
                for suffix, label, stream in (
                    ("question", "English question", c.q),
                    ("excerpt", "Selected excerpt", y),
                    ("written", "Writer reply", raw),
                ):
                    name = tag + "_" + suffix
                    save_image(PUBLIC / (name + ".png"), stream)
                    strips.append([name, label])
            recordrow = pages[key]["questions"][i - start]
            rows.append(
                dict(
                    id=c.question_id,
                    language=lang,
                    question=recordrow[lang],
                    accepted=recordrow["refs"],
                    exact_excerpt=True,
                    writer_exact=bool(c.grade(raw, stop)[2]),
                    stopped=bool(stop),
                )
            )
    record(
        PUBLIC / "figure.json",
        dict(
            series=series,
            table=table,
            strips=strips,
            source=pages[key]["title"],
            example=rows,
        ),
    )


if __name__ == "__main__":
    prepare()
