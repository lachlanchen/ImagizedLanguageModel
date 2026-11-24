"""A translator that can emit a source bitmap outside its learned language palette."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.ordered_translation import OrderedTranslator


class SpecimenTranslator(OrderedTranslator):
    def __init__(self, width=160):
        super().__init__(4, width)
        self.address = nn.Linear(2 * width, width)
        self.request = nn.Linear(width, width, bias=False)
        self.route = nn.Linear(width, 1)
        nn.init.constant_(self.route.bias, -4.0)

    def address_keys(self, context, bank, src):
        # Distinct pixel keys preserve individual addresses within one source block.
        v = context[-1][1].transpose(1, 2).reshape(len(src), -1, bank.size(-1))
        v = v.repeat_interleave(self.stride, 1)[:, : src.size(1)]
        return self.address(torch.cat((v, bank[src]), -1))

    def branches(self, h, words, addresses, src):
        scale = math.sqrt(h.size(-1))
        vocab = F.log_softmax((h @ words.T).float() / scale, -1)
        loc = (self.request(h) @ addresses.transpose(1, 2)).float() / scale
        loc = loc.masked_fill((src == 0)[:, None, :], -torch.inf)
        return vocab, loc.log_softmax(-1), self.route(h).float()

    def loss(self, features, src, tgt, inventory, local, direction):
        bank = self.optical(features)
        ctx, valid = self.encode(bank, src)
        x = torch.cat(
            (self.begin[direction].expand(len(src), 1, -1), bank[tgt[:, :-1]]), 1
        )
        h, _ = self.decode(x, ctx, valid)
        vocab, copy, gate = self.branches(
            h, self.prototypes(bank, inventory), self.address_keys(ctx, bank, src), src
        )
        v = local[tgt]
        match = (tgt[:, :, None] == src[:, None, :]) & (tgt != 0)[:, :, None]
        has = match.any(-1)
        if ((tgt != 0) & (v < 0) & (~has)).any():
            raise ValueError("Unrepresentable target")
        # Use a finite inactive branch before logsumexp so padding never has NaN derivatives.
        s = vocab.gather(-1, v.clamp_min(0)[:, :, None]).squeeze(-1) + F.logsigmoid(
            -gate.squeeze(-1)
        )
        s = s.masked_fill((v < 0) & (tgt != 0), -torch.inf)
        c = copy.masked_fill(~match, -1e8).logsumexp(-1) + F.logsigmoid(
            gate.squeeze(-1)
        )
        c = c.masked_fill(~has, -torch.inf)
        return -torch.logaddexp(s, c).masked_fill(tgt == 0, 0).sum() / (tgt != 0).sum()

    @torch.inference_mode()
    def generate(self, features, src, inventory, direction, limit=128, allow_copy=True):
        bank = self.optical(features)
        ctx, valid = self.encode(bank, src)
        keys = self.address_keys(ctx, bank, src)
        words = self.prototypes(bank, inventory)
        x = self.begin[direction].expand(len(src), 1, -1)
        past = None
        done = torch.zeros(len(src), dtype=torch.bool, device=src.device)
        sequence = []
        for i in range(limit + 1):
            h, past = self.decode(x, ctx, valid, past, i)
            vocab, copy, gate = self.branches(h, words, keys, src)
            if allow_copy:
                weight = gate[:, 0].sigmoid()
                prob = weight.new_zeros(len(src), len(features))
                prob[:, inventory] = vocab[:, 0].exp() * (1 - weight)
                prob.scatter_add_(1, src, copy[:, 0].exp() * weight)
                ids = prob.argmax(-1)
            else:
                ids = inventory[vocab[:, 0].argmax(-1)]
            sequence.append(ids.masked_fill(done, 0))
            done = done | (ids == 1)
            if done.all():
                break
            x = bank[ids][:, None]
        return torch.stack(sequence, 1)
