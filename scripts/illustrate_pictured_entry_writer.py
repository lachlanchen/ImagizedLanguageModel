"""Collect the actual first held-out card, its answers, and the training example for 言."""

import numpy as np
from scripts.evaluate_paired_student import read_streams
from scripts.infer_pictured_entry import page
from scripts.train_pictured_entry_writer import (
    DEFAULT,
    MODELS,
    OUT,
    assets,
    corpus,
    load,
    restored,
    sha,
)
from scripts.prepare_pictured_entry_answers import write_json
from ilm.visual_lm.entry_span_writer import alternatives, parse_card


def run():
    sets = {s: corpus(s) for s in ("train", "development")}
    roster = load(DEFAULT / "train/index.json")
    yan = next(
        i
        for i, r in enumerate(roster)
        if r["head"] == "言" and r["stage"] == "seal" and r["field"] == "本义"
    )
    study = load(OUT / "evaluation.json")
    samples = MODELS / "examples"
    samples.mkdir()
    normal = read_streams("ordinary_matched", OUT / "outputs")
    alias = read_streams("alias_matched", OUT / "outputs")
    keys, literals = assets()
    model = restored("matched", "cpu")
    examples = {}
    strips = {}
    for split, index, tag in (
        ("development", 0, "ear_meaning"),
        ("development", 3, "ear_sound"),
        ("train", yan, "yan"),
    ):
        data = sets[split]
        card = data["cards"][index // 8]
        question = data["questions"][index]
        [answer], [stop] = model.write(
            [alternatives(parse_card(card, keys), question, literals)]
        )
        if split == "development":
            assert normal[index].tobytes() == answer
        assert answer == data["answers"][index] and stop
        for name, raw in (("card", card), ("question", question), ("answer", answer)):
            if name == "card" and tag == "ear_sound":
                continue
            path = samples / (tag + "_" + name + ".png")
            page(raw, width=(len(raw) // 128 if name != "card" else 24)).save(path)
        examples[tag] = dict(
            split=split,
            row=index,
            card_cells=len(card) // 128,
            question_cells=len(question) // 128,
            expected_sha=__import__("hashlib").sha256(answer).hexdigest(),
            stop=stop,
        )
        if split == "development":
            for kind, raw in (("question", question), ("answer", answer)):
                path = OUT / (tag + "_" + kind + ".png")
                page(raw, width=len(raw) // 128).save(path)
                strips[path.stem] = dict(
                    cells=len(raw) // 128, row=index, sha=sha(path)
                )
    # Compare the same paraphrase and its field synonym on the first mismatched case.
    ordinary = study["conditions"]["paraphrase"]["matched"]["exact_rows"]
    alias_hits = set(study["conditions"]["alias"]["matched"]["exact_rows"])
    first = next(i for i in ordinary if i not in alias_hits)
    r = load(DEFAULT / "development/index.json")[first]
    from scripts.prepare_pictured_entry_answers import STAGES, Rasterizer

    renderer = Rasterizer()
    word = "原来的意思" if r["field"] == "本义" else "发音"
    q = (
        "查"
        + r["head"]
        + "：它的"
        + word
        + "是什么？并列出"
        + STAGES[r["stage"]][0]
        + "。"
    )
    for kind, raw in (
        ("question", b"".join(renderer.tile(c) for c in q)),
        ("answer", alias[first].tobytes()),
    ):
        path = OUT / ("alias_" + kind + ".png")
        page(raw, width=len(raw) // 128).save(path)
        strips[path.stem] = dict(cells=len(raw) // 128, row=first, sha=sha(path))
    write_json(samples / "examples.json", examples)
    cond = study["conditions"]
    table = [
        [
            "Input or control",
            "Complete / 1,136",
            "Fact / 1,136",
            "Form / 1,136",
            "All eight / 142",
        ]
    ]
    for title, c, m in (
        ("Original wording", "ordinary", "matched"),
        ("New phrasing", "paraphrase", "matched"),
        ("Payloads exchanged", "relocated", "matched"),
        ("Field synonyms", "alias", "matched"),
        ("Deranged supervision", "ordinary", "deranged"),
        ("Authored parser", "ordinary", "parser"),
    ):
        v = cond[c][m]
        table.append(
            [title, *[f"{v[k]:,}" for k in ("exact", "facts", "forms", "all_eight")]]
        )
    series = [
        [round(x, 4) for x in load(OUT / (m + "_fit.json"))["losses"]]
        for m in ("deranged", "matched")
    ]
    write_json(
        OUT / "figure.json",
        dict(table=table, series=series, strips=strips, alias_example=r),
    )


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    run()
