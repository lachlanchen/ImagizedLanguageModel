"""Check transfer to the previously used BiPaR and XQuAD development sources."""

import torch
from scripts.evaluate_bilingual_novel_writer import corpus
from scripts.evaluate_bilingual_book_intervals import READERS, save, four
from scripts.prepare_bilingual_book_intervals import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import read_options
from time import monotonic


@torch.inference_mode()
def run():
    groups = corpus()
    results = {}
    for arm, weights in READERS.items():
        model = FontInvariantReader.restore(weights, "cuda")
        dest = WORK / "evaluation" / arm
        scores = {}
        for domain in ("book", "xquad_regular"):
            stats = {}
            for lang in ("en", "zh"):
                key = domain + "_" + lang
                cases = groups[key]
                ys = [None] * len(cases)
                short = [
                    i
                    for i, c in enumerate(cases)
                    if (len(c.p) + len(c.q)) // 128 + 3 <= 512
                ]
                start = monotonic()
                for j in range(0, len(short), 48):
                    ids = short[j : j + 48]
                    _, texts = read_options(
                        model, [(cases[i].p, cases[i].q) for i in ids]
                    )
                    for i, y in zip(ids, texts, strict=True):
                        ys[i] = y
                for i, c in enumerate(cases):
                    if ys[i] is None:
                        ys[i] = model.answer(c.p, c.q, max_answer=48)[0]
                stats[lang] = save(
                    dest, key, cases, ys, [True] * len(cases), monotonic() - start
                )
                stats[lang]["multi_window"] = len(cases) - len(short)
            stats["four"] = four(stats)
            scores[domain] = stats
        results[arm] = scores
    record(PUBLIC / "domains.json", results)


if __name__ == "__main__":
    with gpu_owner():
        run()
