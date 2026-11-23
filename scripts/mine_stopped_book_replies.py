"""Collect the released writer's own stopped completions on supervised book pages."""

from collections import Counter
import hashlib
import json
from time import monotonic

import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.train_bilingual_novel_writer import (
    PUBLIC as BOOK_PUBLIC,
    WORK as BOOK,
    match_page,
)
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import ROOT, FACES, Face

WORK = ROOT / "artifacts/stopped_book_preference_v1"
PUBLIC = ROOT / "references/evidence/stopped_book_preference_20261124"
PARENT = BOOK / "bilingual/page"
SHARD = 512


def training_cases():
    source = BOOK / "train.jsonl"
    assert (
        sha(source)
        == json.loads((BOOK_PUBLIC / "corpus.json").read_text())["splits"]["train"][
            "sha256"
        ]
    )
    items = {}
    assignments = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        for q, rival in match_page(row["questions"]):
            assert q["id"].startswith("TRAIN_")
            assignments.append((q["id"], rival["id"]))
            for lang in ("zh", "en"):
                key = (q["id"], lang)
                if key in items:
                    raise ValueError("Repeated training ID")
                items[key] = (row, q)
    binding = json.loads((BOOK / "bilingual/assignments.json").read_text())
    assert len(items) == binding["presentations"]
    assert (
        hashlib.sha256(json.dumps(assignments, ensure_ascii=False).encode()).hexdigest()
        == binding["assignments_sha256"]
    )
    return items


@torch.inference_mode()
def mine():
    items = training_cases()
    face = Face(FACES[0])
    weights = json.loads((BOOK_PUBLIC / "bilingual_fit.json").read_text())["weights"]
    assert all(sha(PARENT / k) == h for k, h in weights.items())
    model = PreparedGlyphResponder(PARENT, "cuda")
    folder = WORK / "mining"
    folder.mkdir(parents=True, exist_ok=True)
    keys = sorted(
        items, key=lambda k: (len(items[k][0]["page"]) + 2 * len(items[k][1][k[1]]), k)
    )
    counts = Counter()
    files = {}
    elapsed = 0.0
    for begin in range(0, len(keys), SHARD):
        path = folder / f"replies_{begin:05}.npz"
        receipt = path.with_suffix(".json")
        ids = keys[begin : begin + SHARD]
        if not receipt.exists():
            ys = []
            ends = []
            start = monotonic()
            for i in range(0, len(ids), 24):
                pairs = [
                    (tensor(face(items[k][0]["page"])), tensor(face(items[k][1][k[1]])))
                    for k in ids[i : i + 24]
                ]
                for im, end in model.answer_many(pairs):
                    ys.append(im.numpy().tobytes())
                    ends.append(bool(end))
            temp = path.with_suffix(".partial.npz")
            store_streams(temp, ys, ends)
            temp.replace(path)
            record(
                receipt,
                dict(
                    ids=ids,
                    weights=weights,
                    sha256=sha(path),
                    seconds=monotonic() - start,
                ),
            )
        r = json.loads(receipt.read_text())
        assert (
            r["weights"] == weights
            and r["ids"] == list(map(list, ids))
            and sha(path) == r["sha256"]
        )
        elapsed += r["seconds"]
        files[receipt.name] = sha(receipt)
        ys, ends = load_streams(path)
        for k, y, end in zip(ids, ys, ends, strict=True):
            q = items[k][1]
            status = (
                "cap"
                if not end
                else "exact"
                if y in [face(t) for t in q["refs"]]
                else "wrong_exact"
            )
            counts[k[1] + "/" + status] += 1
        print(begin + len(ids), dict(counts), flush=True)
    record(
        PUBLIC / "mining.json",
        dict(
            source=sha(BOOK / "train.jsonl"),
            weights=weights,
            files=files,
            seconds=elapsed,
            peak_bytes=torch.cuda.max_memory_allocated(),
            counts=dict(counts),
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        mine()
