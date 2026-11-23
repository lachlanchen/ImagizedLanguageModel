"""Score both stopped answers to both pictured questions on the same reserved page."""

from collections import defaultdict
import json

import torch
from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.audit_book_sequence_probability import score_batch
from scripts.evaluate_stopped_book_preference import ARMS
from scripts.mine_stopped_book_replies import WORK, PUBLIC, PARENT
from scripts.pointing_glyph_corpus import Row
from scripts.reserve_preference_books import literary_cases
from scripts.train_causal_glyph_writer import sha, record
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner

TOL = 1e-5


@torch.inference_mode()
def measure():
    cases = literary_cases()
    rows, keys = [], []
    for lang in ("zh", "en"):
        requests = cases["fresh_" + lang]
        for first, second in zip(requests[::2], requests[1::2], strict=True):
            assert first.board == second.board and first.p == second.p
            assert not set(first.source) & set(second.source)
            for q in (first, second):
                for answer in (first.source[0], second.source[0]):
                    rows.append(Row("comparison", q.question_id, q.p, q.q, answer, b""))
            keys.append((first.board, lang))
    result = {}
    for arm in ARMS:
        folder = PARENT if arm == "ordinary" else WORK / arm / "page"
        weights = json.loads((folder.parent / "page_fit.json").read_text())["weights"]
        assert all(sha(folder / k) == h for k, h in weights.items())
        writer = PointingCausalWriter.restore(folder, "cuda")
        read, alphabet, pairs, targets = inputs(writer.writer, rows)
        values = []
        for i in range(0, len(rows), 8):
            scores, _ = score_batch(
                writer, read, alphabet, pairs[i : i + 8], targets[i : i + 8]
            )
            values.extend(scores)
        summary = defaultdict(list)
        for i, (key, lang) in enumerate(keys):
            aa, ab, ba, bb = values[4 * i : 4 * i + 4]
            gap = (aa - ab) + (bb - ba)
            summary[lang].append(
                dict(
                    board=key,
                    logp=[aa, ab, ba, bb],
                    positive=gap > TOL,
                    tie=abs(gap) <= TOL,
                    both=(aa - ab > TOL and bb - ba > TOL),
                )
            )
        result[arm] = {
            lang: dict(
                n=len(v),
                positive=sum(r["positive"] for r in v),
                tied=sum(r["tie"] for r in v),
                both=sum(r["both"] for r in v),
                pages=v,
            )
            for lang, v in summary.items()
        }
        del writer, read, alphabet
    record(PUBLIC / "question_interaction.json", result)


if __name__ == "__main__":
    with gpu_owner():
        measure()
