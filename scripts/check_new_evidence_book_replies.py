"""Recount stopped replies on every request in the five-work literary packet."""

import json
from collections import defaultdict
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.mine_stopped_book_replies import PARENT
from scripts.reserve_bilingual_interval_books import packets
from scripts.reserve_mlqa_pictures import WORK, PUBLIC
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def run():
    net = PreparedGlyphResponder(PARENT, "cuda")
    results = {}
    source = json.loads((PUBLIC / "result.json").read_text())["scores"]["matched"]
    for group, groups in packets().items():
        stats = {}
        boards = defaultdict(list)
        for lang in ("zh", "en"):
            cases = groups["fresh_" + lang]
            path = WORK / f"{group}_{lang}_matched.npz"
            assert sha(path) == source[group][lang]["sha256"]
            excerpts = load_streams(path)[0]
            pairs = [
                (tensor(x), tensor(c.q)) for x, c in zip(excerpts, cases, strict=True)
            ]
            replies = [
                r
                for i in range(0, len(pairs), 8)
                for r in net.answer_many(pairs[i : i + 8])
            ]
            ys = [x.numpy().tobytes() for x, _ in replies]
            stops = [bool(e) for _, e in replies]
            correct = [
                bool(c.grade(y, e)[2]) for c, y, e in zip(cases, ys, stops, strict=True)
            ]
            for c, v in zip(cases, correct, strict=True):
                boards[c.board].append(v)
            dest = WORK / f"{group}_{lang}_writer_matched.npz"
            store_streams(dest, ys, stops)
            stats[lang] = dict(
                exact=sum(correct), n=len(cases), stopped=sum(stops), sha256=sha(dest)
            )
        assert all(len(g) == 4 for g in boards.values())
        stats["four"] = dict(exact=sum(all(g) for g in boards.values()), n=len(boards))
        results[group] = stats
    record(PUBLIC / "book_writer.json", results)


if __name__ == "__main__":
    with gpu_owner():
        run()
