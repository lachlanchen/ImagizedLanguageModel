"""Measure batch-one response time and strict accuracy on identical Chinese requests."""

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
from time import monotonic

import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.interval_teacher_corpus import WORK, PUBLIC
from scripts.local_translation_teacher import local_teacher
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import sha, record
from scripts.train_span_choice import gpu_owner

MODEL = "qwen3:8b-q8_0"
SYSTEM = "只从材料中摘录能完整回答问题的最短原文。只写答案，不要解释或添加标点。"
DIGEST = "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130"


def summarize(rows, answers):
    boards = defaultdict(list)
    for v, good in zip(rows, answers, strict=True):
        boards[v["board"]].append(bool(good))
    assert all(len(v) == 2 for v in boards.values())
    return dict(
        n=len(answers),
        correct=sum(answers),
        boards=len(boards),
        complete=sum(all(ds) for ds in boards.values()),
    )


def compare():
    path = WORK / "pairs_dev.jsonl"
    assert (
        sha(path) == json.loads((WORK / "corpus.json").read_text())["fresh"]["sha256"]
    )
    rows = list(map(json.loads, path.read_text().splitlines()))
    dest = WORK / "latency"
    dest.mkdir(exist_ok=True)
    face = Face((FONTS[0], 2))
    before = monotonic()
    pairs = [(tensor(face(v["p"])), tensor(face(v["q"]))) for v in rows]
    render = monotonic() - before
    warm = ("小河穿过山谷。", "什么穿过山谷？")
    results = {}
    for arm in ("replay", "aligned", "distilled"):
        receipt = dest / (arm + ".json")
        if receipt.exists():
            results[arm] = json.loads(receipt.read_text())
            continue
        root = WORK / arm
        report = json.loads((root / "page_fit.json").read_text())
        assert all(sha(root / "page" / k) == h for k, h in report["weights"].items())
        net = PreparedGlyphResponder(root / "page", "cuda")
        net.answer_many([(tensor(face(warm[0])), tensor(face(warm[1])))])
        torch.cuda.synchronize()
        t = monotonic()
        ys = []
        for pair in pairs:
            ((im, stop),) = net.answer_many([pair])
            ys.append((im.numpy().tobytes(), bool(stop)))
        torch.cuda.synchronize()
        elapsed = monotonic() - t
        ok = [
            s and any(y == face(ref) for ref in v["refs"])
            for (y, s), v in zip(ys, rows, strict=True)
        ]
        raw = dest / (arm + ".npz")
        store_streams(raw, [p for p, s in ys], [s for p, s in ys])
        results[arm] = dict(
            **summarize(rows, ok),
            seconds=elapsed,
            seconds_per_question=elapsed / len(rows),
            render_seconds=render,
            stream_hash=sha(raw),
            checkpoint_bytes=sum(f.stat().st_size for f in (root / "page").iterdir()),
        )
        record(receipt, results[arm])
        del net
        print(arm, results[arm], flush=True)
    torch.cuda.empty_cache()
    ledger = dest / "qwen.jsonl"
    answers = (
        [json.loads(s) for s in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    assert all(
        v["index"] == i and v["id"] == rows[i]["id"] and v["split"] == sha(path)
        for i, v in enumerate(answers)
    )
    with local_teacher(MODEL, dest) as (ask, digest, size), ledger.open("a") as log:
        if digest != DIGEST:
            raise ValueError("Qwen model manifest changed")

        def request(p, q):
            return ask(
                SYSTEM,
                dict(role="user", content=f"材料：{p}\n问题：{q}"),
                think=False,
                options=dict(
                    seed=241179,
                    temperature=0,
                    num_predict=80,
                    num_ctx=2048,
                    num_thread=2,
                ),
            )

        request(*warm)
        for i, v in enumerate(rows[len(answers) :], start=len(answers)):
            t = monotonic()
            a = request(v["p"], v["q"])
            seconds = monotonic() - t
            a.update(index=i, id=v["id"], seconds=seconds, split=sha(path))
            log.write(json.dumps(a, ensure_ascii=False) + "\n")
            log.flush()
            answers.append(a)
            if (i + 1) % 64 == 0:
                print("qwen", i + 1, flush=True)
    ok = [
        a["stop"] == "stop" and a["text"] in v["refs"]
        for a, v in zip(answers, rows, strict=True)
    ]
    seconds = sum(a["seconds"] for a in answers)
    results["qwen"] = dict(
        **summarize(rows, ok),
        seconds=seconds,
        seconds_per_question=seconds / len(rows),
        manifest=digest,
        bytes=size,
        model=MODEL,
        output_tokens=sum(a["tokens"] for a in answers),
        length_stops=sum(a["stop"] == "length" for a in answers),
        ledger=sha(ledger),
    )
    PUBLIC.mkdir(parents=True, exist_ok=True)
    record(PUBLIC / "latency.json", results)
    print(results["qwen"], flush=True)


if __name__ == "__main__":
    with gpu_owner():
        compare()
