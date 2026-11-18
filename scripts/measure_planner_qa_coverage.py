"""Measure length and canonical renderer coverage on the existing held-out passage QA images."""
from collections import Counter
import time

import numpy as np
import torch

from ilm.visual_lm.glyph_planner import GlyphPlanner
from scripts.train_glyph_planner import OUT, ROOT, load, require, sha, write_new
from scripts.train_visual_extractive_reader import load_cases


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    weights = OUT/"aligned/weights.pt"
    require(sha(weights) == load(OUT/"aligned/fit.json")["weights_sha256"], "Changed glyph planner")
    state = torch.load(weights, weights_only=True, map_location="cpu")
    net = GlyphPlanner(state["landmarks"]).eval()
    net.load_state_dict(state)
    train, test = load_cases(time.monotonic())
    report = {}
    for split, cases in (("train", train), ("heldout", test)):
        answers = [case.answer.reshape(-1,128) for case in cases]
        lengths = np.array([len(a) for a in answers])
        unique, inverse = np.unique(np.concatenate(answers), axis=0, return_inverse=True)
        unique = torch.from_numpy(np.unpackbits(unique, axis=1)).bool()
        matches = torch.cat([(net.motor(u[:, net.landmarks]) > 0).eq(u).all(-1) for u in unique.split(128)]).numpy()
        cells = matches[inverse.reshape(-1)]
        whole = np.array([v.all() for v in np.split(cells, lengths.cumsum()[:-1])])
        bounded = (lengths <= 8)
        report[split] = dict(cases=len(answers), groups=len({c.group for c in cases}),
                             length_counts={str(k):v for k,v in sorted(Counter(map(int,lengths)).items())},
                             total_glyphs=len(cells), canonical_exact_glyphs=int(cells.sum()),
                             distinct_tiles=len(matches), canonical_exact_types=int(matches.sum()),
                             within_eight=int(bounded.sum()), full_canonical_exact=int(whole.sum()),
                             canonical_and_within_eight=int((whole & bounded).sum()))
    write_new(OUT/"qa_coverage.json",dict(splits=report,weights_sha256=sha(weights),
              program_sha256=sha(__file__),scope="Supplied reference landmarks on the full QA splits. No passage inference."))
    print(report, flush=True)


if __name__ == "__main__":
    main()
