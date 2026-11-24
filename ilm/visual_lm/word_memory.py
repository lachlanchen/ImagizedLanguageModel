"""Add a fitting-image distribution to a frozen contextual word-image writer."""

import math
from functools import cached_property

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.word_canvas import WordCanvas


class WordMemory(WordCanvas):
    def __init__(self, palette, mode, images, lengths, features):
        super().__init__(palette, mode)
        self.register_buffer("prototype_cells", images)
        self.register_buffer("prototype_lengths", lengths)
        self.register_buffer("prototype_features", features)
        self.retrieve = nn.Linear(self.width, 96)
        self.keys = nn.Parameter(torch.empty(len(lengths), 96))
        self.prior = nn.Parameter(torch.zeros(len(lengths)))
        self.gate = nn.Linear(self.width, 1)
        nn.init.normal_(self.keys, std=0.1)
        nn.init.constant_(self.gate.bias, -2.0)

    @cached_property
    def prototype_words(self):
        glyphs = [bytes(c) for c in self.palette.detach().cpu().numpy()]
        return tuple(
            b"".join(glyphs[j] for j in row[:n])
            for row, n in zip(
                self.prototype_cells.tolist(),
                self.prototype_lengths.tolist(),
                strict=True,
            )
        )

    @cached_property
    def prototype_ids(self):
        return {w: i for i, w in enumerate(self.prototype_words)}

    def lookup(self, bank):
        cached = getattr(bank, "_word_memory_lookup", None)
        if cached is None or cached[0] is not self:
            idx = [self.prototype_ids.get(w, -1) for w in bank.raw]
            bank._word_memory_lookup = self, torch.tensor(idx, device=bank.device)
        return bank._word_memory_lookup[1]

    def retrieve_logits(self, latent):
        return F.linear(self.retrieve(latent), self.keys, self.prior) / math.sqrt(
            self.keys.shape[1]
        )

    def choose_images(
        self, bank, route, pointer, latent, groups, image_table, valid, pending
    ):
        # Within a page, merge every physical alias before taking the maximum.
        # The spelling distribution is searched at its greedy completion only.
        lookup = self.lookup(bank)
        k = len(self.prototype_words)
        suffix = torch.arange(image_table.size(1), device=valid.device)[None]
        union = torch.where(lookup[image_table] >= 0, lookup[image_table], k + suffix)
        p = torch.sigmoid(self.gate(latent))
        if getattr(self, "disable_retrieval", False):
            p = torch.zeros_like(p)
        memory_mass = p * self.retrieve_logits(latent).softmax(-1)
        existing = (1 - p) * route.exp()
        copy = pointer.flatten(2).masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        target = union.gather(1, groups)
        mass = F.pad(memory_mass, (0, image_table.size(1))).scatter_add_(
            2, target[:, None].expand_as(copy), copy * existing[:, :, :1]
        )
        best, chosen = mass.max(-1)
        source_id = image_table.gather(1, (chosen - k).clamp_min(0))
        word = torch.where(chosen < k, -1 - chosen, source_id)
        appearance = torch.where(
            (chosen < k)[:, :, None],
            self.prototype_features[chosen.clamp_max(k - 1), :50],
            bank.features[source_id, :50],
        )
        letters = word.new_zeros((*word.shape, self.max_letters))
        ended = torch.ones_like(pending)
        if pending.any():
            b, t = pending.nonzero(as_tuple=True)
            glyphs, stopped, score, feedback = self.draw_letters(latent[b, t])
            palette = [bytes(c) for c in self.palette.detach().cpu().numpy()]
            raw = [b"".join(palette[c] for c in row if c) for row in glyphs.tolist()]
            mid = word.new_tensor([self.prototype_ids.get(w, -1) for w in raw])
            sid = word.new_tensor([bank.ids.get(w, -1) for w in raw])
            eq = (sid[:, None] == image_table[b]) & (sid[:, None] > 0)
            merged = torch.where(mid >= 0, mid, eq.long().argmax(1) + k)
            overlap = mass[b, t, merged].masked_fill((mid < 0) & ~eq.any(1), 0)
            joint = overlap + existing[b, t, 1] * score.exp()
            if getattr(self, "collect_certificates", False):
                # The most mass any omitted spelling can add is the entire unused
                # spelling budget. Include an extra bin for a new greedy spelling.
                lower = F.pad(mass[b, t], (0, 1))
                bin_id = torch.where((mid >= 0) | eq.any(1), merged, lower.size(1) - 1)
                greedy_mass = existing[b, t, 1] * score.exp() * stopped
                lower.scatter_add_(1, bin_id[:, None], greedy_mass[:, None])
                top = lower.topk(2, dim=-1).values
                residual = existing[b, t, 1] - greedy_mass
                certified = top[:, 0] - top[:, 1] > residual + 1e-6
                self.certificate_count += int(certified.sum())
                self.certificate_total += len(certified)
            better = stopped & (joint > best[b, t])
            bi, ti = b[better], t[better]
            word[bi, ti] = 0
            letters[bi, ti] = glyphs[better]
            best[bi, ti] = joint[better]
            appearance[bi, ti] = feedback[better]
        return word, letters, ended, best.log(), appearance


def fit_loss(net, ds, indices, labels):
    with torch.no_grad():
        inputs, variants = ds.inputs(indices)
        mem, _ = net.memory(*inputs)
        lengths = ds.al[indices]
        targets = ds.ans[indices, : int(lengths.max())].long()
        active = (
            torch.arange(targets.size(1), device=targets.device)[None]
            < lengths[:, None]
        )
        rate = torch.rand(len(indices), 1, device=targets.device)
        rate = torch.where(torch.rand_like(rate) < 0.5, 1.0, rate)
        hidden = (torch.rand_like(targets, dtype=torch.float) < rate) & active
        force = (torch.rand_like(rate[:, 0]) * lengths).long()
        hidden[torch.arange(len(indices), device=targets.device), force] |= ~hidden.any(
            1
        )
        route, pointer, latent = net.paint(
            mem, ds.bank.features[targets, :50], active & ~hidden, lengths
        )
        b, t = hidden.nonzero(as_tuple=True)
        truth, z = targets[b, t], latent[b, t]
        ids = variants.flatten(1)[b]
        ptr = pointer[b, t].flatten(1).masked_fill(ids == 0, -torch.inf).log_softmax(-1)
        copy = ptr.masked_fill(ids != truth[:, None], -torch.inf).logsumexp(-1)
        letters = labels[truth].long()
        size = (letters != 0).sum(1)
        letters = letters[:, : int(size.max()) + 1]
        prefix = F.pad(letters[:, :-1], (1, 0))
        h, _ = net.spell(net.letter_inputs()[prefix], z[None].contiguous())
        logits = net.letter_out(h)
        logits[:, 0, 0] = -torch.inf
        spell = logits.log_softmax(-1).gather(-1, letters[:, :, None]).squeeze(-1)
        spell = spell.masked_fill(
            torch.arange(letters.size(1), device=z.device)[None] > size[:, None], 0
        ).sum(1)
        route = route[b, t].log_softmax(-1)
        base = torch.logaddexp(route[:, 0] + copy, route[:, 1] + spell)
        which = net.lookup(ds.bank)[truth]
    gate = net.gate(z)[:, 0]
    logits = net.retrieve_logits(z).log_softmax(-1)
    mem = logits.gather(1, which.clamp_min(0)[:, None])[:, 0].masked_fill(
        which < 0, -torch.inf
    )
    return -torch.logaddexp(F.logsigmoid(-gate) + base, F.logsigmoid(gate) + mem).mean()
