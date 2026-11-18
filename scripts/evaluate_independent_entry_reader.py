"""Blind deployment on branch choices, independently written questions and retention sets."""

import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.conditioned_entry_reader import ConditionedEntryReader
from ilm.visual_lm.entry_question_reader import QuestionReader
from ilm.visual_lm.entry_span_writer import parse_card
from scripts.check_entry_question_order import build_reversal, produce
from scripts.audit_pictured_entry_answers import sequences
from scripts.evaluate_paired_student import save_streams
from scripts.prepare_entry_request_order import FIELDS, STAGES, DEST as ORDER_DATA
from scripts.prepare_pictured_entry_answers import DEFAULT, Rasterizer, write_json
from scripts.train_independent_entry_reader import (
    MODELS,
    OUT,
    PARENT,
    ARMS,
    assets,
    binding,
    corpus,
    load,
    sha,
)


def views():
    src = corpus("development")
    d = ORDER_DATA / "development"
    ids = np.load(d / "target_row.npy")
    queries = sequences(d, "questions", np.load(d / "query_pixels.npy"))
    meta = load(DEFAULT / "development/index.json")
    render = Rasterizer()

    def raster(text):
        return b"".join(render.tile(c) for c in text)

    prompts = [
        json.loads(line) for line in (OUT / "questions.jsonl").read_text().splitlines()
    ]
    yield (
        "independent",
        [raster(r["text"]) for r in prompts],
        np.asarray([r["row"] for r in prompts]),
        4,
    )
    corrections, references = [], []
    for group in range(len(ids) // 4):
        targets = ids[group * 4 : group * 4 + 4]
        fields = list(FIELDS)
        stages = [STAGES[meta[t]["stage"]][0] for t in targets[:2]]
        if group % 4 & 1:
            fields.reverse()
        if group % 4 & 2:
            stages.reverse()
        for t in targets:
            row = meta[t]
            h = row["head"]
            field = row["field"]
            stage = STAGES[row["stage"]][0]
            a, b = fields.index(field), stages.index(stage)
            corrections.append(
                raster(
                    f"我说的不是{h}的{fields[1-a]}，是它的{field}；图也请换一下，把{stages[1-b]}换成{stage}。"
                )
            )
            references.append(
                raster(
                    f"查{h}。{fields[0]}和{fields[1]}这两项，我要知道的是{('前者','后者')[a]}；{stages[0]}和{stages[1]}两种写法，给我看{('前者','后者')[b]}。"
                )
            )
    yield "correction", corrections, ids, 4
    yield "reference", references, ids, 4
    yield "before", build_reversal(queries, ids), ids, 4
    yield "after", queries, ids, 4
    yield "simple", src["questions"], np.arange(len(src["questions"])), 8


def evaluate():
    assert binding() == load(OUT / "registration.json")["sources"]
    source = corpus("development")
    marks, lit = assets()
    records = [parse_card(r, marks) for r in source["cards"]]
    folder = OUT / "outputs"
    folder.mkdir()
    result = {}
    root = PARENT.parents[1]
    parent_scores = load(
        root / "references/evidence/entry_question_reader_20261119/evaluation.json"
    )["conditions"]
    previous = load(
        root / "references/evidence/entry_question_reader_20261119/recount.json"
    )["results"]
    for view, questions, ids, group in views():
        section = result[view] = {}
        for arm in ("parent", *ARMS):
            if arm == "parent" and view in ("after", "before", "simple"):
                old = (
                    previous["before_label"]
                    if view == "before"
                    else parent_scores["ordinary" if view == "simple" else "questions"][
                        "ordered"
                    ]
                )
                section[arm] = dict(
                    exact=old["exact"],
                    all_correct=old["complete_groups"],
                    n=len(ids),
                    archived=True,
                )
                continue
            if arm == "parent":
                m = QuestionReader()
                path = PARENT / "ordered.safetensors"
            else:
                m = ConditionedEntryReader(arm == "candidate")
                path = MODELS / (arm + ".safetensors")
                assert sha(path) == load(OUT / (arm + "_fit.json"))["weight_sha"]
            m.load_state_dict(load_file(str(path)))
            m.eval()
            outputs, stops = produce(m, questions, records, ids // 8, lit)
            good = np.asarray(
                [
                    done and raw == source["answers"][t]
                    for raw, done, t in zip(outputs, stops, ids, strict=True)
                ]
            )
            name = view + "_" + arm
            save_streams(
                name,
                [np.frombuffer(x, np.uint8).reshape(-1, 128) for x in outputs],
                folder,
            )
            stats = dict(
                exact=int(good.sum()),
                all_correct=int(good.reshape(-1, group).all(1).sum()),
                n=len(ids),
                rows=np.flatnonzero(good).tolist(),
                stops=stops,
                stream_sha=sha(folder / (name + ".npz")),
            )
            if view == "independent":
                excluded = load(OUT / "review.json")[
                    "same_two_requested_items_clear_except"
                ]
                stats["clear_exact"] = int(np.delete(good, excluded).sum())
                stats["direct_exact"] = int(good.reshape(24, 4)[::2].sum())
                stats["distractor_exact"] = int(good.reshape(24, 4)[1::2].sum())
            section[arm] = stats
            print(view, arm, stats["exact"], stats["all_correct"], flush=True)
    s = result["independent"]

    def crosses(arm):
        for view in ("independent", "correction", "reference"):
            r = result[view][arm]
            if r["exact"] < 0.8 * r["n"] or r["all_correct"] < 0.65 * r["n"] / 4:
                return False
        return result["simple"][arm]["exact"] >= 0.95 * result["simple"][arm]["n"]

    write_json(
        OUT / "evaluation.json",
        dict(
            conditions=result,
            cross_construction={arm: crosses(arm) for arm in ("parent", *ARMS)},
            architecture_criterion=s["candidate"]["exact"] - s["shared"]["exact"] >= 10
            and s["candidate"]["all_correct"] >= s["shared"]["all_correct"],
            evaluator_sha=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    evaluate()
