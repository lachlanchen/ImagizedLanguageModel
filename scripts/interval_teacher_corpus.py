"""Build disjoint human question pairs and cache image-reader training scores."""

from collections import Counter
import json
from pathlib import Path
import random
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from scripts.pointing_glyph_corpus import Row, digest
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import ROOT, Face, FONTS
from scripts.span_reservation import normalized, reservation
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import WORK as PARENT, materialize
from scripts.train_page_writer_requests import VARIED, load_records
from scripts.train_span_choice import gpu_owner
from scripts.train_visual_facts import READER

WORK = ROOT / "artifacts/interval_teacher_v1"
PUBLIC = ROOT / "references/evidence/interval_teacher_20261124"
NATURAL = {"cmrc", "drcd", "cmrc_part", "drcd_part", "cmrc_pair", "drcd_pair"}
SEED = 241179


def write_rows(path, rows):
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in rows))
    return dict(
        sha256=sha(path), rows=len(rows), groups=dict(Counter(v["group"] for v in rows))
    )


def fingerprints(p):
    text = normalized(p)
    return {text[:64], text[-64:]} if len(text) >= 64 else {text}


def choose(phase, face, blocked, limit):
    guard = reservation()
    rng = random.Random(SEED + int(phase == "dev"))
    result = []
    names = {
        "cmrc": f"cmrc2018/cmrc2018_{phase}.json",
        "drcd": "drcd/DRCD_" + ("dev" if phase == "dev" else "training") + ".json",
    }
    for group, file in names.items():
        articles = json.loads((ROOT / "data/raw" / file).read_text())["data"]
        pages = [
            p
            for a in articles
            if group != "drcd" or normalized(a["title"]) not in guard.titles
            for p in a["paragraphs"]
        ]
        rng.shuffle(pages)
        count = 0
        for par in pages:
            full = par["context"]
            p = full[:416]
            sig = fingerprints(p)
            if sig & blocked or guard.overlaps(full):
                continue
            if group == "drcd" and any(
                q["id"] in guard.by_question for q in par["qas"]
            ):
                continue
            if any(not c.isspace() and ord(c) not in face.coverage for c in p):
                continue
            qs = []
            for q in par["qas"]:
                question = q["question"]
                refs = list(
                    dict.fromkeys(
                        a["text"]
                        for a in q["answers"]
                        if 0 < len(a["text"]) <= 48 and a["text"] in p
                    )
                )
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
                    v["q"] == question
                    or any(s in t or t in s for s in refs for t in v["refs"])
                    for v in qs
                ):
                    continue
                qs.append(
                    dict(
                        group=group,
                        board=f"{group}/{phase}/{count}",
                        q=question,
                        p=p,
                        refs=refs,
                        id=q["id"],
                    )
                )
                if len(qs) == 2:
                    break
            if len(qs) == 2:
                result.extend(qs)
                blocked.update(sig)
                count += 1
                if count == limit:
                    break
    return result


def load_rows():
    receipt = json.loads((WORK / "corpus.json").read_text())
    path = WORK / "pairs_train.jsonl"
    if sha(path) != receipt["additional_train"]["sha256"]:
        raise ValueError("Training split changed")
    face = Face((FONTS[0], 2))
    rows = materialize(face)
    for line in path.read_text().splitlines():
        v = json.loads(line)
        rows.append(
            Row(
                v["group"] + "_pair",
                v["id"],
                face(v["p"]),
                face(v["q"]),
                face(v["refs"][0]),
                b"",
            )
        )
    if digest(rows) != receipt["image_digest"]:
        raise ValueError("Physical training records changed")
    return rows


def prepare():
    WORK.mkdir(parents=True, exist_ok=False)
    face = Face((FONTS[0], 2))
    blocked = set()
    for source in ("train", "validation", "fresh"):
        for v in split(source):
            blocked.update(fingerprints(v["p"]))
    old = json.loads((PARENT / "evaluation/extra.json").read_text())
    for v in old:
        blocked.update(fingerprints(v["p"]))
    for name in ("training", "held", "fresh"):
        for v in load_records(VARIED, name):
            for p in v["p"].split("　"):
                blocked.update(fingerprints(p))
    for study in ("span_choice", "span_list"):
        for v in json.loads((ROOT / f"artifacts/{study}_v1/fresh.json").read_text()):
            blocked.update(fingerprints(v["p"]))
    fresh = choose("dev", face, blocked, 128)
    added = choose("train", face, blocked, 2048)
    rows = materialize(face)
    assert (
        digest(rows) == json.loads((PARENT / "corpus.json").read_text())["image_digest"]
    )
    evaluation = {face(v["p"]) for v in fresh + old}
    assert not any(r.p in evaluation for r in rows)
    for v in added:
        p = face(v["p"])
        if p in evaluation:
            raise ValueError("Paired training page leaked")
        rows.append(
            Row(v["group"] + "_pair", v["id"], p, face(v["q"]), face(v["refs"][0]), b"")
        )
    receipt = dict(
        fresh=write_rows(WORK / "pairs_dev.jsonl", fresh),
        additional_train=write_rows(WORK / "pairs_train.jsonl", added),
        image_digest=digest(rows),
        rows=len(rows),
        groups=dict(Counter(r.task for r in rows)),
        seed=SEED,
    )
    record(WORK / "corpus.json", receipt)
    print(receipt, flush=True)
    reader = FontInvariantReader.restore(READER, "cuda")
    logits = []
    offsets = [0]
    indices = []
    t = monotonic()
    order = sorted(
        [i for i, r in enumerate(rows) if r.task in NATURAL],
        key=lambda i: len(rows[i].p) + len(rows[i].q),
    )
    torch.cuda.reset_peak_memory_stats()
    for i in range(0, len(order), 16):
        ids = order[i : i + 16]
        batch = contextualize(reader, [(rows[j].p, rows[j].q) for j in ids])
        for j, (_, state) in zip(ids, batch, strict=True):
            x = state[:, 768:770].numpy().copy()
            if len(x) * 128 != len(rows[j].p) or not np.isfinite(x).all():
                raise ValueError("Invalid teacher scores")
            logits.append(x)
            indices.append(j)
            offsets.append(offsets[-1] + len(x))
        if i % 3200 == 0:
            print("scored", i, "of", len(order), flush=True)
    np.savez_compressed(
        WORK / "teacher.npz",
        logits=np.concatenate(logits),
        indices=indices,
        offsets=offsets,
    )
    record(
        WORK / "teacher.json",
        dict(
            rows=len(order),
            cells=offsets[-1],
            seconds=monotonic() - t,
            peak_bytes=torch.cuda.max_memory_allocated(),
            sha256=sha(WORK / "teacher.npz"),
            reader={p.name: sha(p) for p in READER.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        prepare()
