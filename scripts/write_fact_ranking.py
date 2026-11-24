"""Use a learned complete-span score and the released image-native sentence writer."""

import torch
from ilm.visual_lm.fact_ranking import candidates
from scripts.train_fact_channel import pack, join_regions, split_span
from scripts.train_fact_ranking import restore
from scripts.write_fact_channel import VisualAnswer


class RankedAnswer(VisualAnswer):
    def __init__(self, mode, device, capture=False):
        super().__init__("human", device)
        self.ranker = None if mode == "frozen" else restore(mode, device)
        self.capture = capture
        self.lists = []

    def select(self, features, qs, pages):
        page = pack([join_regions(p) for p in pages], self.device)
        features, base, starts, lengths = candidates(
            self.reader, features, pack(qs, self.device), page
        )
        scores = base if self.ranker is None else self.ranker(features, base, lengths)
        choices = scores.argmax(1).tolist()
        if self.capture:
            for a, b, c, s in zip(
                starts.tolist(),
                lengths.tolist(),
                base.tolist(),
                scores.tolist(),
                strict=True,
            ):
                self.lists.append(
                    [
                        (i, n, f, v)
                        for i, n, f, v in zip(a, b, c, s, strict=True)
                        if f != -float("inf")
                    ]
                )
        return [
            split_span(p, int(starts[i, k]), int(lengths[i, k]))
            for i, (p, k) in enumerate(zip(pages, choices, strict=True))
        ]
