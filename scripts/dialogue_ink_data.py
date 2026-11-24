"""Deduplicate rasterized word images and pack variable-length dialogue batches."""

from pathlib import Path
import json
import numpy as np
import torch

from scripts.prepare_dialogue_ink import PUB, WORK, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.dialogue_ink_images import WordImages


def read_data():
    f = WORK / "corpus.json"
    if sha(f) != json.loads((PUB / "split.json").read_text())["sha256"]:
        raise ValueError("Dialogue corpus has changed")
    return json.loads(f.read_text())


def pack(lines, device):
    width = max(1, max(map(len, lines)))
    arr = np.zeros((len(lines), width), np.int32)
    for i, r in enumerate(lines):
        arr[i, : len(r)] = r
    return torch.tensor(arr, device=device), torch.tensor(
        list(map(len, lines)), device=device
    )


class Corpus:
    def __init__(self, data, group, net, optical):
        self.rows = [r for r in data["rows"] if r[0] == group]
        self.device = net.sketch.device
        keys = sorted({r[1] for r in self.rows})
        strings = [data["pages"][k] for k in keys]
        question = [r[3] for r in self.rows]
        history = [r[4] for r in self.rows]
        targets = [r[5] for r in self.rows]
        sources = strings + question + history
        if group == "train":
            sources += targets
        words = {w for s in sources for w in s.split()}
        face = Face(FACES[0])
        self.raw = {w: face(w) for w in words}
        self.bank = WordImages(self.raw.values(), net, optical)
        self.index = {w: self.bank.ids[v] for w, v in self.raw.items()}

        def encoded(lines):
            return pack(
                [[self.index[w] for w in s.split()] for s in lines], self.device
            )

        self.page, self.pl = encoded(strings)
        self.q, self.ql = encoded(question)
        self.hist, self.hl = encoded(history)
        if group == "train":
            self.ans, self.al = encoded(targets)
        mapping = {k: i for i, k in enumerate(keys)}
        self.pindex = torch.tensor(
            [mapping[r[1]] for r in self.rows], device=self.device
        )
        self.ev = torch.tensor([r[6] for r in self.rows], device=self.device)

    def glyph_targets(self):
        # The zero image marks the spelling boundary and is never a nonblank output letter.
        glyphs = sorted(
            {
                bytes(w[i : i + 128])
                for r in self.rows
                for t in r[5].split()
                for w in [self.raw[t]]
                for i in range(0, len(w), 128)
            }
        )
        palette = [bytes(128)] + glyphs
        ids = {v: i for i, v in enumerate(palette)}
        labels = np.zeros((len(self.bank.raw), 25), np.int16)
        for r in self.rows:
            for word in r[5].split():
                w = self.raw[word]
                labels[self.index[word], : len(word)] = [
                    ids[w[i : i + 128]] for i in range(0, len(w), 128)
                ]
        return torch.tensor(
            np.frombuffer(b"".join(palette), np.uint8).copy().reshape(-1, 128)
        ), torch.tensor(labels, device=self.device)

    def inputs(self, idx, qidx=None, pidx=None):
        qidx = idx if qidx is None else qidx
        p = self.pindex[idx if pidx is None else pidx]
        pi, qi, hi = self.page[p], self.q[qidx], self.hist[qidx]
        f = self.bank.features
        return (
            f[pi],
            self.pl[p],
            f[qi],
            self.ql[qidx],
            f[hi],
            self.hl[qidx],
        ), self.bank.edits[torch.cat((pi, qi, hi), 1).long()]
