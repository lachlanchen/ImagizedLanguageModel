"""Illustrate two opposing requests on the first card and the first cue-placement failure."""

import numpy as np

from scripts.audit_pictured_entry_answers import sequences
from scripts.check_entry_question_order import build_reversal
from scripts.evaluate_paired_student import read_streams
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import write_json
from scripts.train_entry_question_reader import (
    ARMS,
    DEST,
    MODELS,
    OUT,
    corpus,
    load,
    sha,
)


def run():
    source = corpus("development")
    d = DEST / "development"
    ids = np.load(d / "target_row.npy")
    questions = sequences(d, "questions", np.load(d / "query_pixels.npy"))
    reversal = build_reversal(questions, ids)
    report = load(OUT / "evaluation.json")["conditions"]
    supplemental = load(OUT / "recount.json")["results"]["before_label"]
    known = read_streams("questions_ordered", OUT / "outputs")
    changed = read_streams("cue_before_label", OUT / "outputs")
    success = set(supplemental["rows"])
    failure = next(
        (i for i in report["questions"]["ordered"]["rows"] if i not in success), 0
    )
    examples = MODELS / "examples"
    examples.mkdir()
    strips = {}
    for i, tag, question, output in [
        (0, "meaning", questions[0], known[0]),
        (3, "sound", questions[3], known[3]),
        (failure, "cue_first", reversal[failure], changed[failure]),
    ]:
        raw = output.tobytes()
        for part, pixels in (("question", question), ("answer", raw)):
            name = tag + "_" + part
            img = page(pixels, width=len(pixels) // 128)
            img.save(OUT / (name + ".png"))
            img.save(examples / (name + ".png"))
            strips[name] = dict(
                cells=len(pixels) // 128, row=i, sha=sha(OUT / (name + ".png"))
            )
        card = source["cards"][ids[i] // 8]
        page(card, width=24).save(examples / (tag + "_card.png"))
        strips[tag + "_question"]["card_cells"] = len(card) // 128
        strips[tag + "_answer"]["exact"] = raw == source["answers"][ids[i]]
    table = [
        [
            "Question wording",
            "Ordered exact",
            "Sorted exact",
            "Ordered all four",
            "Sorted all four",
        ]
    ]
    for view, title in (
        ("questions", "Training frames"),
        ("unseen_wording", "Reserved wording"),
    ):
        arm = report[view]
        table.append(
            [
                title,
                *[
                    f"{arm[a][key]:,}"
                    for key in ("exact", "complete_groups")
                    for a in ("ordered", "canonical")
                ],
            ]
        )
    table.append(
        [
            "Cue before label",
            f"{supplemental['exact']:,}",
            "—",
            f"{supplemental['complete_groups']:,}",
            "—",
        ]
    )
    series = [
        [round(x, 5) for x in load(OUT / (a + "_fit.json"))["losses"]] for a in ARMS
    ]
    write_json(OUT / "figure.json", dict(table=table, series=series, strips=strips))


if __name__ == "__main__":
    run()
