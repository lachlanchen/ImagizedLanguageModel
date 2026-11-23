"""Test an existing visual span reader on the writer's independently reserved books."""

from collections import defaultdict
from time import monotonic

import torch
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import read_options
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.mine_stopped_book_replies import WORK, PUBLIC
from scripts.reserve_first_glyph_books import reserved
from scripts.reserve_preference_books import literary_cases
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_visual_facts import READER


@torch.inference_mode()
def evaluate():
    model = FontInvariantReader.restore(READER, "cuda")
    folder = WORK / "span_baseline"
    folder.mkdir(exist_ok=True)
    result = {}
    for split, groups in (("earlier", reserved()), ("new", literary_cases())):
        boards = defaultdict(list)
        sets = {}
        for lang in ("zh", "en", "erased"):
            cases = groups["fresh_" + lang]
            start = monotonic()
            _, outputs = read_options(model, [(c.p, c.q) for c in cases])
            flags = [
                bool(c.grade(y, True)[2]) for c, y in zip(cases, outputs, strict=True)
            ]
            for c, y, ok in zip(cases, outputs, flags, strict=True):
                assert y in c.p and 1 <= len(y) // 128 <= 48
                if lang != "erased":
                    boards[c.board].append(ok)
            path = folder / (split + "_" + lang + ".npz")
            store_streams(path, outputs, [True] * len(cases))
            sets[lang] = dict(
                n=len(cases),
                exact=sum(flags),
                seconds=monotonic() - start,
                sha256=sha(path),
            )
        assert all(len(x) == 4 for x in boards.values())
        result[split] = dict(
            sets=sets, four=sum(all(v) for v in boards.values()), n=len(boards)
        )
    record(
        PUBLIC / "span_baseline.json",
        dict(
            scores=result,
            parameters=sum(p.numel() for p in model.parameters()),
            weights={
                p.name: sha(p)
                for p in READER.iterdir()
                if p.suffix in (".safetensors", ".json")
            },
            decoder="single contiguous span with predicted endpoints, maximum 48 glyphs",
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
