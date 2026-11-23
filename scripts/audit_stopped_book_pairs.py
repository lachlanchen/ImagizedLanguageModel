"""Measure preference learning on the actual accepted training pairs by reply length."""

from collections import defaultdict
import json

import numpy as np
import torch
from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.audit_book_sequence_probability import score_batch
from scripts.mine_stopped_book_replies import PARENT, PUBLIC, WORK, training_cases
from scripts.pointing_glyph_corpus import Row
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face


@torch.inference_mode()
def run():
    items = training_cases()
    judged = json.loads((WORK / "accepted.json").read_text())
    audit = json.loads((PUBLIC / "label_review.json").read_text())
    assert audit["accepted_sha"] == sha(WORK / "accepted.json")
    rejected = {tuple(x) for x in audit["exclude"]}
    face = Face(FACES[0])
    rows = []
    labels = []
    keys = []
    for v in judged:
        k = (v["id"], v["lang"])
        if k in rejected:
            continue
        page, q = items[k]
        a, b = v["refs"][0], v["wrong"]
        labels.append(
            "shorter" if len(b) < len(a) else "longer" if len(b) > len(a) else "equal"
        )
        keys.append(k)
        rows.extend(
            Row("pair", v["id"], face(page["page"]), face(q[k[1]]), face(s), b"")
            for s in (a, b)
        )
    out = {}
    for arm in ("ordinary", "replay", "preference"):
        folder = PARENT if arm == "ordinary" else WORK / arm / "page"
        weights = json.loads((folder.parent / "page_fit.json").read_text())["weights"]
        assert all(sha(folder / k) == h for k, h in weights.items())
        model = PointingCausalWriter.restore(folder, "cuda")
        read, alphabet, pairs, ys = inputs(model.writer, rows)
        scores = []
        for i in range(0, len(rows), 12):
            batch, _ = score_batch(
                model, read, alphabet, pairs[i : i + 12], ys[i : i + 12]
            )
            scores.extend(batch)
        gaps = np.subtract(scores[::2], scores[1::2])
        groups = defaultdict(list)
        for gap, group in zip(gaps.tolist(), labels, strict=True):
            groups[group].append(gap)
            groups["all"].append(gap)
        path = WORK / (arm + "_pair_scores.json")
        record(
            path,
            [
                dict(id=k[0], lang=k[1], gold=a, rival=b)
                for k, a, b in zip(keys, scores[::2], scores[1::2], strict=True)
            ],
        )
        out[arm] = dict(
            groups={
                k: dict(
                    n=len(v),
                    preferred=sum(t > 1e-5 for t in v),
                    mean_gap=float(np.mean(v)),
                    median_gap=float(np.median(v)),
                )
                for k, v in groups.items()
            },
            sha256=sha(path),
        )
        del model, read, alphabet
    record(PUBLIC / "training_gaps.json", out)


if __name__ == "__main__":
    with gpu_owner():
        run()
