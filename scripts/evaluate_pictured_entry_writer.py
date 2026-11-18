"""Run complete answers, source interventions and the authored parser on frozen weights."""

import fcntl
from time import monotonic
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.entry_span_writer import alternatives, parse_card
from scripts.evaluate_paired_student import save_streams
from scripts.prepare_pictured_entry_answers import STAGES, write_json
from scripts.train_pictured_entry_writer import (
    DEFAULT,
    MODELS,
    OUT,
    ROOT,
    Rasterizer,
    assets,
    corpus,
    load,
    restored,
    sha,
)


def relocate(records):
    (a, fa), (b, fb) = records
    av, bv = dict(fa), dict(fb)
    assert av.keys() == bv.keys()
    return [(a, [(k, bv[k]) for k, _ in fa]), (b, [(k, av[k]) for k, _ in fb])]


def authored(records, question, text):
    for head, fields in records:
        h = b"".join(head)
        for f, value in fields:
            field = b"".join(f)
            for k, ink in fields:
                stage = b"".join(k)
                if question in (
                    text("请写出")
                    + h
                    + text("的")
                    + field
                    + text("和")
                    + stage
                    + text("字形。"),
                    text("查")
                    + h
                    + text("：它的")
                    + field
                    + text("是什么？并列出")
                    + stage
                    + text("。"),
                ):
                    return (
                        h
                        + text("的")
                        + field
                        + text("是")
                        + b"".join(value)
                        + text("，")
                        + stage
                        + text("作")
                        + b"".join(ink)
                        + text("。")
                    )
    return b""


def grade(outputs, stops, gold, facts, labels, text):
    full = np.array(
        [x == y and done for x, y, done in zip(outputs, gold, stops)]
    ).reshape(-1, 8)
    good_fact, good_form = 0, 0
    comma, period = text("，"), text("。")
    for x, y, f, label in zip(outputs, gold, facts, labels):
        mark = text("的" + label + "是")
        field = x.partition(mark)[2].partition(comma)
        good_fact += int(bool(field[1]) and field[0] == text(f))
        good_form += int(len(x) >= 256 and x[-256:] == y[-256:] and x[-128:] == period)
    return dict(
        exact=int(full.sum()),
        all_eight=int(full.all(1).sum()),
        facts=good_fact,
        forms=good_form,
        stops=sum(stops),
        exact_rows=np.flatnonzero(full.ravel()).tolist(),
        pairs={
            name: int((full & full[:, np.arange(8) ^ mask]).sum() // 2)
            for name, mask in (("stage", 1), ("fact", 2), ("head", 4))
        },
    )


def run():
    d = corpus("development")
    metadata = load(DEFAULT / "development/index.json")
    marks, literals = assets()
    for arm in ("matched", "deranged"):
        assert (
            sha(MODELS / "appearance.npz")
            == load(OUT / (arm + "_fit.json"))["assets_sha"]
        )
    base = [parse_card(raw, marks) for raw in d["cards"]]
    moved = [relocate(c) for c in base]
    render = Rasterizer()

    def text(s):
        return b"".join(render.tile(c) for c in s)

    gold = d["answers"]
    n = len(gold)
    moved_gold = []
    synonyms = []
    for i, r in enumerate(metadata):
        # Opposite entry on this card has the same requested field and stage.
        other = metadata[i ^ 4]
        assert (
            r["card"] == other["card"]
            and r["field"] == other["field"]
            and r["stage"] == other["stage"]
        )
        assert gold[i ^ 4][:128] == text(other["head"])
        moved_gold.append(text(r["head"]) + gold[i ^ 4][128:])
        alias = "原来的意思" if r["field"] == "本义" else "发音"
        synonyms.append(
            text(
                "查"
                + r["head"]
                + "：它的"
                + alias
                + "是什么？并列出"
                + STAGES[r["stage"]][0]
                + "。"
            )
        )
    conditions = dict(
        ordinary=(d["questions"], base, gold),
        paraphrase=(d["question_paraphrases"], base, gold),
        relocated=(d["questions"], moved, moved_gold),
        alias=(synonyms, base, gold),
        no_question=([b"\0" * len(q) for q in d["questions"]], base, gold),
        no_card=(d["questions"], [[] for _ in base], gold),
    )
    stream_dir = OUT / "outputs"
    stream_dir.mkdir()
    results = {}
    for condition, (qs, records, refs) in conditions.items():
        options = [alternatives(records[i // 8], q, literals) for i, q in enumerate(qs)]
        results[condition] = {}
        for method in ("deranged", "matched", "parser"):
            if method == "deranged" and condition not in ("ordinary", "paraphrase"):
                continue
            start = monotonic()
            if method == "parser":
                output = [authored(records[i // 8], q, text) for i, q in enumerate(qs)]
                done = [bool(o) for o in output]
            else:
                model = restored(method, "cuda")
                output, done = [], []
                for i in range(0, n, 64):
                    images, stops = model.write(options[i : i + 64])
                    output.extend(images)
                    done.extend(stops)
            seconds = monotonic() - start
            name = condition + "_" + method
            save_streams(
                name,
                [np.frombuffer(x, np.uint8).reshape(-1, 128) for x in output],
                stream_dir,
            )
            facts = [
                metadata[i ^ 4 if condition == "relocated" else i]["value"]
                for i in range(n)
            ]
            stats = grade(
                output, done, refs, facts, [r["field"] for r in metadata], text
            )
            stats.update(
                seconds=seconds,
                stopped=done,
                stream_sha=sha(stream_dir / (name + ".npz")),
            )
            results[condition][method] = stats
            print(
                condition,
                method,
                stats["exact"],
                stats["all_eight"],
                stats["facts"],
                stats["forms"],
                flush=True,
            )
    assert all(
        results[k]["parser"]["exact"] == n
        for k in ("ordinary", "paraphrase", "relocated")
    )
    gate = (
        all(
            results[k]["matched"]["exact"] >= 0.8 * n
            and results[k]["matched"]["all_eight"] >= 0.65 * len(base)
            and results[k]["matched"]["exact"] - results[k]["deranged"]["exact"]
            >= 0.5 * n
            for k in ("ordinary", "paraphrase")
        )
        and results["relocated"]["matched"]["exact"] >= 0.8 * n
    )
    write_json(
        OUT / "evaluation.json",
        dict(
            conditions=results,
            questions=n,
            cards=len(base),
            gate=gate,
            registration_sha=sha(OUT / "registration.json"),
            evaluator_sha=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
