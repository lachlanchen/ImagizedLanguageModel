"""Deduplicate physically identical word images for visual pointer decoding."""

import numpy as np
import torch
from scripts.answer_optical_passage_qa import visual_words


def trims(word):
    return word, word[:-128], word[128:], word[128:-128]


class WordImages:
    def __init__(self, original, reader, optical):
        self.device = reader.sketch.device
        # A word is identified only by the packed rows of its ink, including in the supervision.
        forms = {b""}
        for word in original:
            forms.update(trims(word))
        self.raw = sorted(forms, key=lambda x: (len(x), x))
        self.ids = {word: i for i, word in enumerate(self.raw)}
        self.features = torch.zeros(len(forms), 115, device=self.device)
        for i in range(1, len(forms), 256):
            chunk = [
                np.frombuffer(w, np.uint8).reshape(-1, 128)
                for w in self.raw[i : i + 256]
            ]
            self.features[i : i + len(chunk)] = visual_words(reader, optical, chunk)
        edits = np.zeros((len(forms), 4), np.int32)
        for w in original:
            edits[self.ids[w]] = [self.ids[v] for v in trims(w)]
        self.edits = torch.tensor(edits, device=self.device)
