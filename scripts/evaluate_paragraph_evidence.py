"""Separate evidence selection from exact, stopped writing on the same pictured inputs."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
from time import monotonic

import torch
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.paragraph_evidence import ParagraphRanker, lexical_score
from ilm.visual_lm.pictured_answer_selection import read_options
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor, contains
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.grounded_proposition_corpus import noun
from scripts.paragraph_evidence_corpus import WORK
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_page_writer_requests import VARIED, NEW, load_records
from scripts.train_paragraph_evidence import records
from scripts.train_span_choice import gpu_owner
from scripts.train_visual_facts import READER


def saved(folder, name, expected):
    path = folder / (name + ".npz")
    if sha(path) != expected:
        raise ValueError(f"Changed physical streams in {path}")
    return list(zip(*load_streams(path), strict=True))


def prior_outputs():
    cache = {}
    prev = VARIED / "evaluation"
    result = json.loads((prev / "results.json").read_text())
    fresh = load_records(VARIED, "fresh")
    for k, meta in enumerate(result["varied"]["wordings"]):
        ys = saved(prev, f"varied_{k}", meta["sha256"])
        for v, y in zip(fresh, ys, strict=True):
            cache[v["p"], NEW[k].format(noun(v["s"], v["r"]))] = y
    transfer = VARIED / "transfer"
    receipt = json.loads((transfer / "results.json").read_text())["page"]
    ys = saved(transfer, "page", receipt["streams"])
    humans = split("fresh")
    if len(ys) != sum(x["questions"] for x in receipt["scores"].values()) or len(
        humans
    ) != sum(receipt["scores"][g]["questions"] for g in ("drcd", "cmrc")):
        raise ValueError("Transfer stream inventory changed")
    for v, y in zip(humans, ys[: len(humans)], strict=True):
        cache[v["p"], v["q"]] = y
    return cache


def routes(v, face, ranker):
    q = face(v["q"])
    ps = [face(s) for s in v["parts"]]
    scores = ranker.scores(q, ps)
    literal = [lexical_score(q, p) for p in ps]
    return dict(
        learned=max(range(2), key=scores.__getitem__),
        literal=max(range(2), key=literal.__getitem__),
        oracle=v["right"],
        rival=1 - v["right"],
    )


def inputs(v, route):
    parts = v["parts"]
    if route is None:
        separator = "" if v["type"] in ("cmrc", "drcd") else "　"
        return separator.join(parts), v["q"]
    return parts[route], v["q"]


@torch.inference_mode()
def predictions(keys, face, call, cache=None):
    outputs = {} if cache is None else cache
    needed = sorted(
        set(keys) - outputs.keys(), key=lambda z: (len(z[0]) + 2 * len(z[1]), z)
    )
    for start in range(0, len(needed), 16):
        batch = needed[start : start + 16]
        predicted = call([(face(p), face(q)) for p, q in batch])
        outputs.update(zip(batch, predicted, strict=True))
        if start and start % 800 == 0:
            print("new model inputs", start, "of", len(needed), flush=True)
    return outputs


def report(rows, ys, face, selected=None):
    groups = defaultdict(Counter)
    complete = defaultdict(list)
    for i, (v, (y, stop)) in enumerate(zip(rows, ys, strict=True)):
        group = v["type"]
        ok = stop and y in [face(r) for r in v["refs"]]
        c = groups[group]
        c["questions"] += 1
        c["correct"] += int(ok)
        c["stops"] += int(stop)
        if selected is not None:
            c["right_paragraph"] += int(selected[i] == v["right"])
        if v["year"]:
            c["right_year"] += int(
                stop
                and contains(y, face(v["year"]))
                and not contains(y, face(v["rival"]))
            )
        complete[group, v["board"]].append(ok)
    for (group, _), passed in complete.items():
        groups[group]["boards"] += 1
        groups[group]["complete"] += int(all(passed))
    return dict(groups)


def evaluate(folder):
    torch.set_num_threads(2)
    start = monotonic()
    out = folder / "evaluation"
    out.mkdir(exist_ok=False)
    fit = json.loads((folder / "fit.json").read_text())
    if sha(folder / "paragraph.safetensors") != fit["sha256"]:
        raise ValueError("Selector weights changed")
    if any(
        sha(VARIED / "page" / n) != h
        for n, h in json.loads((VARIED / "page_fit.json").read_text())[
            "weights"
        ].items()
    ):
        raise ValueError("Writer weights changed")
    rows = records(folder, "held")
    face = lru_cache(maxsize=24000)(Face((FONTS[0], 2)))
    ranker = ParagraphRanker.restore(folder / "paragraph.safetensors", "cuda")
    choices = [routes(x, face, ranker) for x in rows]
    record(out / "routing.json", choices)
    keys = {
        arm: [
            inputs(x, None if arm == "whole" else ch[arm])
            for x, ch in zip(rows, choices, strict=True)
        ]
        for arm in ("whole", "oracle", "rival", "literal", "learned")
    }
    writer = PreparedGlyphResponder(VARIED / "page", "cuda")

    def call(pairs):
        return [
            (v.numpy().tobytes(), bool(s))
            for v, s in writer.answer_many([(tensor(p), tensor(q)) for p, q in pairs])
        ]

    ys = predictions([k for v in keys.values() for k in v], face, call, prior_outputs())
    results = {}
    for arm, items in keys.items():
        outputs = [ys[k] for k in items]
        path = out / (arm + ".npz")
        store_streams(path, [y for y, _ in outputs], [s for _, s in outputs])
        results[arm] = dict(
            scores=report(
                rows,
                outputs,
                face,
                None if arm == "whole" else [v[arm] for v in choices],
            ),
            streams=sha(path),
        )
    del writer
    human = [i for i, x in enumerate(rows) if x["type"] in ("cmrc", "drcd")]
    reader = FontInvariantReader.restore(READER, "cuda")

    def read(pairs):
        _, quotes = read_options(reader, pairs)
        return [(y, True) for y in quotes]

    spans = predictions([v[i] for v in keys.values() for i in human], face, read)
    readers = {}
    for arm, v in keys.items():
        answers = [spans[v[i]] for i in human]
        path = out / ("extract_" + arm + ".npz")
        store_streams(path, [y for y, _ in answers], [True] * len(answers))
        readers[arm] = dict(
            scores=report([rows[i] for i in human], answers, face), streams=sha(path)
        )
    record(
        out / "result.json",
        dict(
            writer=results,
            extractor=readers,
            seconds=monotonic() - start,
            source_choices_sha256=sha(out / "routing.json"),
        ),
    )
    for arm, x in results.items():
        print(
            arm,
            {
                g: (m["correct"], m["complete"], m["boards"])
                for g, m in x["scores"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=WORK)
    with gpu_owner():
        evaluate(ap.parse_args().run_dir)
