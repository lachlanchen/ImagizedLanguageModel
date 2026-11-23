"""Measure natural question switches, historical ink and factual retention from actual image replies."""

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random
import shutil

import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import read_options
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_paragraph_evidence import inputs, report
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.paragraph_evidence_corpus import WORK as ROUTER
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import ROOT, Face, FONTS
from scripts.span_reservation import normalized, reservation, development_path
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import WORK, PUBLIC
from scripts.train_page_writer_requests import VARIED, load_records
from scripts.train_paragraph_evidence import records
from scripts.train_span_choice import gpu_owner
from scripts.train_visual_facts import READER


def more_pairs(face):
    blocked = set()
    for name in ("train", "validation", "fresh"):
        blocked.update(normalized(v["p"]) for v in split(name))
    for name in ("training", "held", "fresh"):
        blocked.update(normalized(v["p"]) for v in load_records(VARIED, name))
    for name in ("train", "held"):
        blocked.update(normalized(p) for v in records(ROUTER, name) for p in v["parts"])
    for folder in ("span_choice_v1", "span_list_v1"):
        blocked.update(
            normalized(v["p"])
            for v in json.loads(
                (ROOT / "artifacts" / folder / "fresh.json").read_text()
            )
        )
    rows = []
    guard = reservation()
    rng = random.Random(241155)
    for group, path in [
        ("cmrc", ROOT / "data/raw/cmrc2018/cmrc2018_dev.json"),
        ("drcd", development_path()),
    ]:
        data = json.loads(path.read_text())["data"]
        passages = (
            list(guard.filter_development(data))
            if group == "drcd"
            else [p for a in data for p in a["paragraphs"]]
        )
        rng.shuffle(passages)
        count = 0
        for par in passages:
            p = par["context"][:416]
            norm = normalized(p)
            if (
                norm in blocked
                or guard.overlaps(p)
                or any(not c.isspace() and ord(c) not in face.coverage for c in p)
            ):
                continue
            chosen = []
            for q in par["qas"]:
                refs = list(
                    dict.fromkeys(
                        a["text"]
                        for a in q["answers"]
                        if 0 < len(a["text"]) <= 48 and a["text"] in p
                    )
                )
                question = q["question"]
                if (
                    not refs
                    or not question
                    or len(p) + len(question) > 509
                    or any(
                        not c.isspace() and ord(c) not in face.coverage
                        for c in question
                    )
                ):
                    continue
                if any(
                    question == v["q"]
                    or any(s in t or t in s for s in refs for t in v["refs"])
                    for v in chosen
                ):
                    continue
                chosen.append(
                    dict(
                        group=group,
                        board=str(count),
                        p=p,
                        q=question,
                        refs=refs,
                        id=q["id"],
                    )
                )
                if len(chosen) == 2:
                    break
            if len(chosen) == 2:
                rows.extend(chosen)
                count += 1
                blocked.add(norm)
                if count == 128:
                    break
    return rows


@torch.inference_mode()
def evaluate(folder=WORK, evidence=PUBLIC):
    out = folder / "evaluation"
    if (evidence / "evaluation.json").exists():
        raise FileExistsError(evidence / "evaluation.json")
    out.mkdir(exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    font = Face((FONTS[0], 2))
    face = lru_cache(maxsize=20000)(font)
    reg = records(ROUTER, "held")
    rpath = ROUTER / "evaluation/routing.json"
    assert (
        sha(rpath)
        == json.loads((ROUTER / "evaluation/result.json").read_text())[
            "source_choices_sha256"
        ]
    )
    ch = json.loads(rpath.read_text())
    extra = more_pairs(font)
    record(out / "extra.json", extra)
    tasks = defaultdict(list)

    def put(group, board, p, q, refs):
        tasks[group].append((board, (p, q), refs))

    for v in extra:
        put(
            "extra_" + v["group"],
            v["board"],
            face(v["p"]),
            face(v["q"]),
            [face(s) for s in v["refs"]],
        )
    for v in extra:
        put(
            "blind_" + v["group"],
            v["board"],
            face(v["p"]),
            bytes(128),
            [face(s) for s in v["refs"]],
        )
    for v in split("fresh"):
        put(
            v["group"],
            v["key"],
            face(v["p"]),
            face(v["q"]),
            [face(s) for s in v["refs"]],
        )
    for r in plates("development", font):
        put(r.task, r.key.split("/")[0], r.p, r.q, [r.y])
    for v, c in zip(reg, ch, strict=True):
        for arm in ("whole", "learned", "oracle"):
            if arm == "oracle" and v["type"] not in ("cmrc", "drcd"):
                continue
            p, q = inputs(v, None if arm == "whole" else c[arm])
            put(
                v["type"] + "_" + arm,
                v["board"],
                face(p),
                face(q),
                [face(s) for s in v["refs"]],
            )
    results = {}
    exemplars = []
    for arm, checkpoint in (
        ("mixed", folder / "page"),
        ("parent", VARIED / "page"),
        ("extractor", READER),
    ):
        use = list(tasks) if arm == "mixed" else ["extra_cmrc", "extra_drcd"]
        material = [key for name in use for _, key, _ in tasks[name]]
        unique = sorted(set(material), key=lambda k: (len(k[0]) + 2 * len(k[1]), k))
        if arm == "extractor":
            model = FontInvariantReader.restore(checkpoint, "cuda")
        else:
            expected = json.loads((checkpoint.parent / "page_fit.json").read_text())[
                "weights"
            ]
            assert all(sha(checkpoint / n) == h for n, h in expected.items())
            model = PreparedGlyphResponder(checkpoint, "cuda")
        emitted = {}
        for start in range(0, len(unique), 16):
            batch = unique[start : start + 16]
            if arm == "extractor":
                _, quotes = read_options(model, batch)
                answers = [(y, True) for y in quotes]
            else:
                answers = [
                    (im.numpy().tobytes(), bool(s))
                    for im, s in model.answer_many(
                        [(tensor(p), tensor(q)) for p, q in batch]
                    )
                ]
            emitted.update(zip(batch, answers, strict=True))
            if start and start % 1200 == 0:
                print(arm, start, len(unique), flush=True)
        del model
        output = {}
        for name in use:
            ok = []
            stopped = []
            pixels = []
            boards = defaultdict(list)
            for j, (board, k, refs) in enumerate(tasks[name]):
                y, stop = emitted[k]
                exact = stop and y in refs
                ok.append(exact)
                stopped.append(stop)
                pixels.append(y)
                boards[board].append((y, exact))
                if arm == "mixed" and name == "extra_cmrc" and sum(ok) == 1 and exact:
                    for short, raw in (
                        ("question", k[1]),
                        ("answer", y),
                        ("page", k[0]),
                    ):
                        write_strip(
                            tensor(raw).numpy(), evidence / ("human_" + short + ".png")
                        )
                    exemplars.append(
                        dict(
                            group=name,
                            index=j,
                            question=next(
                                x["q"] for x in extra if face(x["q"]) == k[1]
                            ),
                            length=len(y) // 128,
                        )
                    )
            path = out / (arm + "_" + name + ".npz")
            store_streams(path, pixels, stopped)
            output[name] = dict(
                n=len(ok),
                exact=sum(ok),
                stopped=sum(stopped),
                boards=len(boards),
                complete=sum(all(z for _, z in vs) for vs in boards.values()),
                sha256=sha(path),
            )
            if name.startswith("extra_"):
                output[name]["identical_on_switch"] = sum(
                    vs[0][0] == vs[1][0] for vs in boards.values()
                )
        results[arm] = output
        print(arm, {k: v["exact"] for k, v in output.items()}, flush=True)
        record(out / (arm + ".json"), output)
    record(
        evidence / "evaluation.json",
        dict(results=results, extra_sha256=sha(out / "extra.json"), example=exemplars),
    )
    for n in ("corpus.json", "page_fit.json"):
        shutil.copyfile(folder / n, evidence / n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=WORK)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    evidence = args.evidence_dir or (
        PUBLIC
        if args.run_dir.resolve() == WORK.resolve()
        else args.run_dir / "evidence"
    )
    with gpu_owner():
        evaluate(args.run_dir, evidence)
