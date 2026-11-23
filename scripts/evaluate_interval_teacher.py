"""Compare complete image replies and factual retention after matched writer training."""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from time import monotonic

import torch
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import read_options
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_paragraph_evidence import inputs
from scripts.evaluate_pointing_causal_writer import store_streams, load_streams
from scripts.interval_teacher_corpus import WORK, PARENT, PUBLIC
from scripts.paragraph_evidence_corpus import WORK as ROUTER
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_paragraph_evidence import records
from scripts.train_span_choice import gpu_owner
from scripts.train_visual_facts import READER


@dataclass
class Case:
    board: str
    p: bytes
    q: bytes
    refs: tuple


def trials():
    font = Face((FONTS[0], 2))
    face = lru_cache(maxsize=16000)(font)
    groups = defaultdict(list)
    old_path = PARENT / "evaluation/extra.json"
    parent_results = json.loads((PARENT / "evaluation/mixed.json").read_text())
    from scripts.train_mixed_visual_answers import PUBLIC as OLD_PUBLIC

    assert (
        sha(old_path)
        == json.loads((OLD_PUBLIC / "evaluation.json").read_text())["extra_sha256"]
    )
    fresh_path = WORK / "pairs_dev.jsonl"
    assert (
        sha(fresh_path)
        == json.loads((WORK / "corpus.json").read_text())["fresh"]["sha256"]
    )
    fresh = [json.loads(v) for v in fresh_path.read_text().splitlines()]
    for name, rows in [("prior", json.loads(old_path.read_text())), ("fresh", fresh)]:
        for v in rows:
            p, q, refs = face(v["p"]), face(v["q"]), tuple(face(r) for r in v["refs"])
            groups[name].append(Case(v["board"], p, q, refs))
            groups[name + "_blind"].append(Case(v["board"], p, bytes(128), refs))
    for v in split("fresh"):
        groups[v["group"]].append(
            Case(
                v["key"], face(v["p"]), face(v["q"]), tuple(face(r) for r in v["refs"])
            )
        )
    for r in plates("development", font):
        groups[r.task].append(Case(r.key.split("/")[0], r.p, r.q, (r.y,)))
    choices = ROUTER / "evaluation/routing.json"
    assert (
        sha(choices)
        == json.loads((ROUTER / "evaluation/result.json").read_text())[
            "source_choices_sha256"
        ]
    )
    for v, ch in zip(
        records(ROUTER, "held"), json.loads(choices.read_text()), strict=True
    ):
        if v["type"] in ("cmrc", "drcd"):
            continue
        for arm in ("whole", "learned"):
            p, q = inputs(v, None if arm == "whole" else ch[arm])
            groups[v["type"] + "_" + arm].append(
                Case(v["board"], face(p), face(q), tuple(face(r) for r in v["refs"]))
            )
    assert len(groups["prior"]) == parent_results["extra_cmrc"]["n"]
    return groups


def grade(cases, streams):
    boards = defaultdict(list)
    exact = stops = 0
    for case, (y, s) in zip(cases, streams, strict=True):
        good = s and y in case.refs
        boards[case.board].append(good)
        exact += good
        stops += s
    return dict(
        n=len(cases),
        correct=exact,
        stops=stops,
        boards=len(boards),
        complete=sum(all(v) for v in boards.values()),
    )


def previous(arm, group):
    key = {"prior": "extra_cmrc", "prior_blind": "blind_cmrc"}.get(group, group)
    name = "mixed" if arm == "parent" else "extractor"
    root = PARENT / "evaluation"
    info = json.loads((root / (name + ".json")).read_text())
    if key not in info:
        return None
    path = root / (name + "_" + key + ".npz")
    if sha(path) != info[key]["sha256"]:
        raise ValueError("Changed original reply pixels")
    return list(zip(*load_streams(path), strict=True))


@torch.inference_mode()
def evaluate(arms):
    torch.set_num_threads(2)
    groups = trials()
    for arm in arms:
        root = WORK / "evaluation" / arm
        root.mkdir(parents=True, exist_ok=False)
        datasets = (
            groups
            if arm != "teacher"
            else {k: groups[k] for k in ("prior", "fresh", "cmrc", "drcd")}
        )
        start = monotonic()
        outputs = {}
        for name, cases in datasets.items():
            cached = previous(arm, name) if arm in {"parent", "teacher"} else None
            if cached is not None:
                outputs.update(
                    ((v.p, v.q), y) for v, y in zip(cases, cached, strict=True)
                )
        if arm == "teacher":
            model = FontInvariantReader.restore(READER, "cuda")
        else:
            source = PARENT if arm == "parent" else WORK / arm
            expected = json.loads((source / "page_fit.json").read_text())["weights"]
            assert all(sha(source / "page" / name) == h for name, h in expected.items())
            model = PreparedGlyphResponder(source / "page", "cuda")
        needed = sorted(
            {(v.p, v.q) for ds in datasets.values() for v in ds} - outputs.keys(),
            key=lambda x: (len(x[0]) + 2 * len(x[1]), x),
        )
        torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(needed), 16):
            batch = needed[i : i + 16]
            if arm == "teacher":
                _, spans = read_options(model, batch)
                ys = [(s, True) for s in spans]
            else:
                ys = [
                    (pic.numpy().tobytes(), bool(stop))
                    for pic, stop in model.answer_many(
                        [(tensor(p), tensor(q)) for p, q in batch]
                    )
                ]
            outputs.update(zip(batch, ys, strict=True))
            if i and i % 1200 == 0:
                print(arm, i, len(needed), flush=True)
        scores = {}
        for name, cases in datasets.items():
            streams = [outputs[v.p, v.q] for v in cases]
            path = root / (name + ".npz")
            store_streams(path, [y for y, _ in streams], [s for _, s in streams])
            scores[name] = dict(**grade(cases, streams), sha256=sha(path))
        record(
            root / "scores.json",
            dict(
                scores=scores,
                seconds=monotonic() - start,
                new_inputs=len(needed),
                peak_bytes=torch.cuda.max_memory_allocated(),
            ),
        )
        print(
            arm,
            {g: (v["correct"], v["complete"]) for g, v in scores.items()},
            flush=True,
        )
        del model


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "arms",
        nargs="+",
        choices=["parent", "teacher", "replay", "aligned", "distilled"],
    )
    with gpu_owner():
        evaluate(ap.parse_args().arms)
