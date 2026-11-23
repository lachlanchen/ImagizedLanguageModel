"""Evaluate the released image writer with predicted evidence from the MLQA reader."""

import json
from time import monotonic
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.evaluate_question_evidence_head import mlqa
from scripts.mine_stopped_book_replies import PARENT
from scripts.reserve_mlqa_pictures import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def run():
    sets, pairs = mlqa()
    first = json.loads((PUBLIC / "result.json").read_text())
    net = PreparedGlyphResponder(PARENT, "cuda")
    results = {}
    for arm in ("full", "parent", "exchanged", "matched"):
        stats = {}
        correct = {}
        for lang, rows in sets.items():
            if arm == "full":
                inputs = [c.p for c in rows]
            else:
                path = WORK / f"mlqa_{lang}_{arm}.npz"
                assert sha(path) == first["scores"][arm]["mlqa"][lang]["sha256"]
                inputs = load_streams(path)[0]
            batch = [
                (tensor(p), tensor(r.q)) for p, r in zip(inputs, rows, strict=True)
            ]
            start = monotonic()
            outputs = [
                reply
                for i in range(0, len(batch), 8)
                for reply in net.answer_many(batch[i : i + 8])
            ]
            elapsed = monotonic() - start
            ys = [y.numpy().tobytes() for y, _ in outputs]
            ends = [bool(s) for _, s in outputs]
            flags = [
                bool(r.grade(y, s)[2]) for r, y, s in zip(rows, ys, ends, strict=True)
            ]
            dest = WORK / f"mlqa_{lang}_writer_{arm}.npz"
            store_streams(dest, ys, ends)
            stats[lang] = dict(
                exact=sum(flags),
                n=len(rows),
                seconds=elapsed,
                stopped=sum(ends),
                sha256=sha(dest),
            )
            if arm != "full":
                selected = [
                    bool(c.grade(p, True)[2]) for c, p in zip(rows, inputs, strict=True)
                ]
                stats[lang]["lost"] = sum(
                    a and not b for a, b in zip(selected, flags, strict=True)
                )
                stats[lang]["repaired"] = sum(
                    b and not a for a, b in zip(selected, flags, strict=True)
                )
            correct[lang] = dict(zip([r.question_id for r in rows], flags, strict=True))
        stats["four"] = dict(
            n=len(pairs),
            exact=sum(all(correct[l][i] for i in pair for l in sets) for pair in pairs),
        )
        results[arm] = stats
    record(PUBLIC / "writer.json", results)


if __name__ == "__main__":
    with gpu_owner():
        run()
