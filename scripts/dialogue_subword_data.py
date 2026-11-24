"""Tokenize training-only BPE surfaces and pack complete source dialogues."""

import re
import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


def fit_tokenizer(data):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    rows = [r for r in data["rows"] if r[0] == "train"]
    pages = sorted({r[1] for r in rows})

    def lines():
        yield from (data["pages"][k] for k in pages)
        for r in rows:
            yield from r[3:6]

    tok.train_from_iterator(
        lines(),
        trainers.BpeTrainer(
            vocab_size=4096,
            show_progress=False,
            special_tokens=["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tok


class Turns:
    def __init__(self, data, group, tok):
        self.rows = [r for r in data["rows"] if r[0] == group]
        self.pages = {}
        self.spans = {}
        self.turns = []
        self.lengths = []
        for k in sorted({r[1] for r in self.rows}):
            text = data["pages"][k]
            enc = tok.encode(text)
            offsets = np.array(enc.offsets)
            words = list(re.finditer(r"\S+", text))
            starts = np.searchsorted(
                offsets[:, 1], [w.start() for w in words], side="right"
            )
            ends = (
                np.searchsorted(offsets[:, 0], [w.end() for w in words], side="left")
                - 1
            )
            self.pages[k] = enc.ids
            self.spans[k] = (starts, ends)
        for _, key, _, q, h, a, ev, _ in self.rows:
            q, h, a = (v.ids for v in tok.encode_batch([q, h, a]))
            self.turns.append((key, q, h, a, ev))
            self.lengths.append(len(self.pages[key]) + len(q) + len(h) + 3)

    def batch(self, ids, device, qids=None, pids=None):
        qids = ids if qids is None else qids
        pids = ids if pids is None else pids
        src = []
        segments = []
        labels = []
        evidence = []
        for i, j, k in zip(ids, qids, pids, strict=True):
            _, _, _, a, ev = self.turns[i]
            page = self.turns[k][0]
            _, q, h, _, _ = self.turns[j]
            p = self.pages[page]
            prefix = [1] + q + [3] + h + [3]
            src.append(prefix + p)
            segments.append([0] * (len(q) + 2) + [1] * (len(h) + 1) + [2] * len(p))
            labels.append([1] + a + [2])
            loc = self.spans[page]
            evidence.append(
                [int(loc[t][ev[t]]) + len(prefix) for t in (0, 1)]
                if i == k and ev[0] >= 0
                else [-1, -1]
            )

        def pad(rows):
            n = max(map(len, rows))
            array = np.zeros((len(rows), n), np.int32)
            for j, row in enumerate(rows):
                array[j, : len(row)] = row
            return torch.from_numpy(array).to(device, dtype=torch.long)

        return (
            pad(src),
            pad(segments),
            pad(labels),
            torch.tensor(evidence, device=device),
        )
