"""Test cross-script question identity with the passage, its answers and OCR withheld."""

from collections import defaultdict
import json
import math
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.paired_question_encoder import PicturedQuestionEncoder
from scripts.evaluate_bilingual_novel_writer import corpus
from scripts.reserve_bilingual_interval_books import packets
from scripts.train_pictured_question_alignment import WORK, PUBLIC
from scripts.train_span_choice import gpu_owner
from scripts.train_causal_glyph_writer import record, sha


def question_vectors(model, streams):
    # Equal-length batches give the same convolution boundary for every request.
    lengths = defaultdict(list)
    for i, s in enumerate(streams):
        lengths[len(s)].append(i)
    all_ = torch.empty((len(streams), 128))
    for ids in lengths.values():
        all_[ids] = model.encode([streams[i] for i in ids]).cpu()
    return all_


def score(matrix, zh, en):
    n = len(zh)
    # Raw corpus retrieval. Identical pictured English queries may have several
    # accepted Chinese translations. Physically identical targets get equal credit.
    aliases = defaultdict(set)
    for a, b in zip(zh, en, strict=True):
        aliases[b].add(a)
    top = matrix.argsort(axis=1, kind="stable")[:, ::-1][:, :5]
    strict = []
    ties = 0
    for i in range(0, n, 2):
        m = matrix[i, i] + matrix[i + 1, i + 1] - matrix[i, i + 1] - matrix[i + 1, i]
        strict.append(float(m))
        ties += int(abs(m) < 1e-6)
    positive = sum(v >= 1e-6 for v in strict)
    count = n // 2
    p = positive / count
    z = 1.95996398454
    half = (
        z
        * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
        / (1 + z * z / count)
    )
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    return dict(
        pairs=count,
        wins=positive,
        ties=ties,
        losses=count - positive - ties,
        wilson95=[center - half, center + half],
        margins=strict,
        queries=n,
        top1=sum(zh[row[0]] in aliases[q] for row, q in zip(top, en, strict=True)),
        top5=sum(
            any(zh[i] in aliases[q] for i in row)
            for row, q in zip(top, en, strict=True)
        ),
    )


@torch.inference_mode()
def run():
    previous = corpus()
    sets = {k: (r["fresh_zh"], r["fresh_en"]) for k, r in packets().items()}
    for k in ("book", "xquad_regular"):
        sets[k] = (previous[k + "_zh"], previous[k + "_en"])
    results = {}
    for arm in ("length", "aligned", "exchanged"):
        if arm != "length":
            weights = WORK / (arm + ".safetensors")
            assert (
                sha(weights)
                == json.loads((PUBLIC / (arm + "_fit.json")).read_text())[
                    "weights_sha256"
                ]
            )
            model = PicturedQuestionEncoder().cuda().eval()
            model.load_state_dict(load_file(str(weights)))
        groups = {}
        for name, (z, e) in sets.items():
            assert len(z) % 2 == 0 and all(
                a.p == b.p and a.question_id == b.question_id
                for a, b in zip(z, e, strict=True)
            )
            assert all(z[i].p == z[i + 1].p for i in range(0, len(z), 2))
            zh, en = [v.q for v in z], [v.q for v in e]
            start = monotonic()
            if arm == "length":
                # Similar lengths prefer similar lengths. Centering or a global
                # scale factor cannot change this baseline's paired assignment.
                a = np.log(np.array(list(map(len, en)))[:, None])
                b = np.log(np.array(list(map(len, zh)))[None, :])
                similarities = -((a - b) ** 2)
            else:
                similarities = (
                    (question_vectors(model, en) @ question_vectors(model, zh).T)
                    .numpy()
                    .astype(float)
                )
            groups[name] = score(similarities, zh, en)
            groups[name]["seconds"] = monotonic() - start
        results[arm] = groups
    a = results["aligned"]["book"]
    passed = a["wilson95"][0] > 0.5 and all(
        a["wins"] > results[k]["book"]["wins"] for k in ("length", "exchanged")
    )
    record(
        PUBLIC / "result.json", dict(scores=results, matched_direction_passed=passed)
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
