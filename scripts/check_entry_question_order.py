"""Check complete archived answers and test unchanged weights with cues before their fields."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.entry_question_reader import QuestionReader, anchors, offsets, retina
from ilm.visual_lm.entry_span_writer import alternatives, cells, parse_card
from scripts.audit_pictured_entry_answers import sequences
from scripts.evaluate_paired_student import read_streams, save_streams
from scripts.prepare_entry_request_order import FRAMES, FIELDS, STAGES
from scripts.prepare_pictured_entry_answers import Rasterizer, write_json
from scripts.train_entry_question_reader import (
    ARMS,
    DEST,
    MODELS,
    OUT,
    assets,
    corpus,
    load,
    sha,
)
from scripts.train_pictured_entry_writer import DEFAULT


def reordered_question(row, field_pair, stage_pair, frame, outer):
    known, ask, seen, show = FRAMES[frame]
    fields = "，".join((ask if f == row["field"] else known) + f for f in field_pair)
    forms = "，".join(
        (show if s == STAGES[row["stage"]][0] else seen) + s for s in stage_pair
    )
    return (
        "关于" + row["head"] + "：" + fields + "；" + forms + "。"
        if outer == 0
        else row["head"] + "这个字，" + forms + "；" + fields + "。"
    )


def build_reversal(data, ids):
    directory = DEST / "development"
    meta = load(DEFAULT / "development/index.json")
    layout = load(directory / "layouts.json")
    raster = Rasterizer()
    questions = []
    for i, t in enumerate(ids):
        frame, outer, perm = layout[i // 4]
        field_pair = FIELDS[::-1] if perm & 1 else FIELDS
        start = ids[(i // 4) * 4]
        stage_pair = [STAGES[meta[j]["stage"]][0] for j in (start, start + 1)]
        if perm & 2:
            stage_pair.reverse()
        s = reordered_question(meta[t], field_pair, stage_pair, frame, outer)
        q = b"".join(raster.tile(c) for c in s)
        assert Counter(cells(q)) == Counter(cells(data[i]))
        questions.append(q)
    return questions


@torch.inference_mode()
def produce(reader, questions, records, card_ids, literal):
    options = [alternatives(entry, b"", literal) for entry in records]
    labels = [anchors(entry, len(literal)) for entry in records]
    out, stopped = [], []
    for begin in range(0, len(questions), 64):
        group = questions[begin : begin + 64]
        cids = card_ids[begin : begin + 64]
        opts = [options[c] for c in cids]
        loc = np.stack(
            [offsets(q, labels[c]) for q, c in zip(group, cids, strict=True)]
        )
        length = max(map(len, group)) // 128 + 4
        pixels = np.zeros((len(group), length, 64), np.float32)
        mask = np.zeros((len(group), length), bool)
        for i, q in enumerate(group):
            spans = cells(q)
            pixels[i, : len(spans)] = np.stack([retina(g) for g in spans])
            mask[i, : len(spans)] = True
        keys = reader.read(
            torch.from_numpy(np.stack([o.features for o in opts])),
            torch.from_numpy(loc),
            torch.from_numpy(pixels),
            torch.from_numpy(mask),
        )
        o, s = reader.emit(keys, opts)
        out.extend(o)
        stopped.extend(s)
    return out, stopped


def check():
    directory = DEST / "development"
    tiles = np.load(directory / "query_pixels.npy")
    ids = np.load(directory / "target_row.npy")
    original = corpus("development")
    questions = sequences(directory, "questions", tiles)
    views = dict(
        questions=(questions, ids),
        unseen_wording=(sequences(directory, "unseen_wording", tiles), ids),
        ordinary=(original["questions"], np.arange(len(original["questions"]))),
    )
    marks, literal = assets()
    records = [parse_card(x, marks) for x in original["cards"]]
    reader = QuestionReader().eval()
    reader.load_state_dict(load_file(str(MODELS / "ordered.safetensors")))
    assert (
        sha(MODELS / "ordered.safetensors")
        == load(OUT / "ordered_fit.json")["weight_sha"]
    )
    measured = load(OUT / "evaluation.json")["conditions"]
    report = {}
    for view, (qs, target) in views.items():
        out, stop = produce(reader, qs, records, target // 8, literal)
        report[view] = {}
        for arm in ARMS:
            stats = measured[view][arm]
            saved = [
                x.tobytes() for x in read_streams(view + "_" + arm, OUT / "outputs")
            ]
            rows = [
                i
                for i, (raw, s, t) in enumerate(
                    zip(saved, stats["stopped"], target, strict=True)
                )
                if raw == original["answers"][t] and s
            ]
            assert rows == stats["rows"]
            report[view][arm] = {"exact": len(rows), "streams": len(saved)}
            if arm == "ordered":
                diff = sum(
                    a != b or x != y
                    for a, b, x, y in zip(
                        out, saved, stop, stats["stopped"], strict=True
                    )
                )
                report[view][arm]["cpu_replay_differences"] = diff
                report[view][arm]["cpu_exact"] = sum(
                    done and x == original["answers"][t]
                    for done, x, t in zip(stop, out, target, strict=True)
                )
    flipped = build_reversal(questions, ids)
    out, stop = produce(reader, flipped, records, ids // 8, literal)
    score = np.asarray(
        [
            done and raw == original["answers"][t]
            for raw, done, t in zip(out, stop, ids, strict=True)
        ]
    )
    save_streams(
        "cue_before_label",
        [np.frombuffer(x, np.uint8).reshape(-1, 128) for x in out],
        OUT / "outputs",
    )
    report["before_label"] = dict(
        exact=int(score.sum()),
        complete_groups=int(score.reshape(-1, 4).all(1).sum()),
        stops=stop,
        rows=np.flatnonzero(score).tolist(),
        n=len(out),
        stream_sha=sha(OUT / "outputs/cue_before_label.npz"),
    )
    write_json(
        OUT / "recount.json", dict(results=report, evaluator_sha=sha(Path(__file__)))
    )
    print(
        {
            k: (v["ordered"] if "ordered" in v else (v["exact"], v["complete_groups"]))
            for k, v in report.items()
        }
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    check()
