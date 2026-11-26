"""Causal visual page states shared across consecutive word predictions."""

from hashlib import sha256
import json
import math
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.passage_selector import PassageSelector, PassageWriter, Print, ROOT

MODEL = ROOT / "models/causal_book_words"


class CausalBookWords(PassageSelector):
    def encode_page(self, bank, src, marks):
        valid = src.ne(0)
        offsets = valid.long().cumsum(1).clamp_min(1) - 1
        x = self.read(bank.keys[src]) + self.gap(marks) + self.position[offsets]
        order = torch.arange(src.shape[1], device=src.device)
        visible = (order[:, None] >= order[None, :])[None, :, :] & valid[:, None, :]
        visible = visible | (
            torch.eye(len(order), device=src.device, dtype=torch.bool)[None]
            & ~valid[:, :, None]
        )
        for layer in self.layers:
            x, _ = layer(x, visible[:, None])
        return self.final(x)

    def read_at(self, bank, src, states, keys, rows, last, task):
        q = states[rows, last] + self.task(task)
        query = self.query(q)
        score = torch.bmm(keys[rows], query[:, :, None]).squeeze(2).float() / math.sqrt(
            q.shape[-1]
        )
        words = src[rows]
        valid = (
            torch.arange(src.shape[1], device=src.device) <= last[:, None]
        ) & bank.lengths[words].gt(0)
        score = score + (bank.idf[words] + 0.01).log()
        a = score.masked_fill(~valid, -1e20).softmax(-1) * valid
        gate = self.copy_gate(q).float().sigmoid().squeeze(1) * valid.any(1)
        return q, a, gate

    def observe(self, bank, src, marks, task):
        states = self.encode_page(bank, src, marks)
        rows = torch.arange(len(src), device=src.device)
        return self.read_at(
            bank,
            src,
            states,
            self.key(states),
            rows,
            torch.full_like(rows, src.shape[1] - 1),
            task,
        )

    def objective(self, bank, src, states, keys, rows, last, task, y, end):
        q, a, g = self.read_at(bank, src, states, keys, rows, last, task)
        match = (src[rows] == y[:, None]) & (
            torch.arange(src.shape[1], device=src.device) <= last[:, None]
        )
        present = match.any(1)
        copy = (a * match).sum(1) * g
        logspell = self.spell_logprob(q, bank.codes[y], bank.lengths[y])
        mixture = torch.logaddexp(
            copy.clamp_min(1e-30).log(), torch.log1p(-g.clamp_max(1 - 1e-7)) + logspell
        )
        nll = -mixture

        def mean_subset(loss, mask):
            return (loss * mask).sum() / mask.sum().clamp_min(1)

        categories = present.any().float() + (~present).any().float()
        nll = (mean_subset(nll, present) + mean_subset(nll, ~present)) / categories
        stop = F.binary_cross_entropy_with_logits(
            self.end_logits(q, bank.keys[y]), end.float(), reduction="none"
        )
        return nll + mean_subset(stop, task.eq(0)), mixture.detach(), present


class CausalBookWriter(PassageWriter):
    def __init__(self, arm="dense", device="cpu"):
        if arm not in ("dense", "endpoint"):
            raise ValueError(arm)
        self.device = device
        self.face = Print()
        self.net = CausalBookWords(self.face.coarse).to(device).eval()
        path = MODEL / arm
        weights = path / "weights.safetensors"
        receipt = json.loads((path / "fit.json").read_text())
        if sha256(weights.read_bytes()).hexdigest() != receipt["weights"]:
            raise ValueError("Changed checkpoint")
        self.net.load_state_dict(load_file(str(weights)))
