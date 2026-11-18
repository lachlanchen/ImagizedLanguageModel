"""Clause-conditioned visual generation with a source pointer and an open ink writer."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import InkMotor, visual_input, nibbles
from ilm.visual_lm.visual_edit_writer import InkPatterns

COPY, EXEMPLAR, DRAW, END = range(4)


class CausalGlyphTranslator(nn.Module):
    def __init__(self, landmarks, exemplars, width=192):
        super().__init__()
        self.register_buffer("landmarks", landmarks.long())
        self.register_buffer("exemplars", exemplars.bool())
        self.register_buffer("group_ids", torch.arange(12) * 16)
        self.ink = nn.Sequential(
            nn.Linear(112, width), nn.GELU(), nn.Linear(width, width)
        )
        self.groups = nn.Embedding(192, width)
        self.source = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.source_norm = nn.LayerNorm(width)
        self.initial = nn.Linear(width, width)
        self.start = nn.Parameter(torch.randn(width) * 0.02)
        self.history = nn.GRU(width, width, batch_first=True)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.stride = nn.Linear(width, 1)
        self.plan = nn.Sequential(
            nn.Linear(2 * width + 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.route = nn.Linear(width, 4)
        self.prototype = nn.Linear(width, width)
        self.prototype_key = nn.Linear(width, width)
        self.draw = InkPatterns(width)
        self.motor = InkMotor().requires_grad_(False)

    def embed(self, features):
        return self.ink(features) + self.groups(
            self.group_ids + nibbles(features[..., 64:])
        ).mean(-2)

    def read(self, features, lengths):
        x = self.embed(features)
        masked = torch.arange(x.size(1), device=x.device)[None] >= lengths[:, None]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        ctx, _ = self.source(packed)
        ctx, _ = nn.utils.rnn.pad_packed_sequence(
            ctx, batch_first=True, total_length=x.size(1)
        )
        memory = self.source_norm(ctx + x)
        summary = memory.masked_fill(masked[..., None], 0.0).sum(1) / lengths[:, None]
        return memory, self.key(memory), masked, self.initial(summary).tanh()[None]

    def plan_steps(self, q, memory, keys, masked, lengths, previous_center):
        # The decoder integrates nonnegative reading increments, without a supplied cursor.
        center = previous_center[:, None] + F.softplus(
            self.stride(q).squeeze(-1)
        ).float().cumsum(1)
        position = torch.arange(memory.size(1), device=q.device)
        score = (self.query(q) @ keys.transpose(1, 2)) * q.size(-1) ** -0.5
        score = score.float() - 0.32 * (center[:, :, None] - position).abs()
        attention = score.masked_fill(masked[:, None, :], float("-inf")).log_softmax(-1)
        focus = attention.exp().to(memory.dtype) @ memory
        left = lengths[:, None] - center
        progress = torch.stack((center / lengths[:, None], left.clamp(-8, 8) / 8), -1)
        state = q + self.plan(torch.cat((q, focus, progress.to(q.dtype)), dim=-1))
        return state, attention, center

    def image_keys(self):
        features = visual_input(self.exemplars, self.landmarks)
        return F.normalize(self.prototype_key(self.embed(features)), dim=-1)

    def prototype_scores(self, state, image_keys):
        return 16 * (F.normalize(self.prototype(state), dim=-1) @ image_keys.T)

    def loss(
        self,
        source,
        lens,
        target,
        target_lens,
        aligned,
        routes,
        exemplar_indices,
        history_features=None,
    ):
        memory, keys, masked, h = self.read(source, lens)
        target_features = self.embed(
            target if history_features is None else history_features
        )
        inputs = torch.cat(
            (self.start.expand(len(lens), 1, -1), target_features), dim=1
        )
        q, _ = self.history(inputs, h)
        state, attention, center = self.plan_steps(
            q,
            memory,
            keys,
            masked,
            lens,
            torch.full_like(lens, -0.65, dtype=torch.float),
        )
        pos = torch.arange(q.size(1), device=q.device)[None]
        valid = pos <= target_lens[:, None]
        route_loss = F.cross_entropy(self.route(state)[valid], routes[valid])
        live = pos[:, :-1] < target_lens[:, None]
        attn = attention[:, :-1][live]
        columns = aligned[live]
        location = -attn.gather(1, columns[:, None]).mean()
        monotone = F.smooth_l1_loss(center[:, :-1][live], columns.float())
        # Normalize route-specific likelihoods over all genuine output decisions.
        normalizer = valid.sum()
        real = state[:, :-1]
        memorized = (routes[:, :-1] == EXEMPLAR) & live
        painted = (routes[:, :-1] == DRAW) & live
        loss = route_loss + 0.65 * location + 0.08 * monotone
        if memorized.any():
            logits = self.prototype_scores(real[memorized], self.image_keys())
            loss += (
                F.cross_entropy(logits, exemplar_indices[memorized], reduction="sum")
                / normalizer
            )
        if painted.any():
            bits = target[..., 64:][painted]
            logits = self.draw.teacher(real[painted], bits)
            loss += (
                F.cross_entropy(logits.transpose(1, 2), nibbles(bits), reduction="sum")
                / normalizer
            )
        return loss

    def image_outcomes(self, pictures):
        # Physical equality is exact, including novel ink and repeated source cells.
        def packed(tensor):
            return np.packbits(tensor.cpu().numpy(), axis=-1).reshape(-1, 128)

        bank, inverse = np.unique(
            np.concatenate((packed(self.exemplars), packed(pictures))),
            axis=0,
            return_inverse=True,
        )
        device = pictures.device
        indices = torch.as_tensor(inverse, device=device)
        count = len(self.exemplars)
        return (
            torch.from_numpy(np.unpackbits(bank, axis=1)).to(device).bool(),
            indices[:count],
            indices[count:].reshape(pictures.shape[:2]),
        )

    def certified_outcome(self, state, attention, keys, outcomes):
        bank, exemplar_ids, source_ids = outcomes
        prob = self.route(state).double().softmax(-1)
        batch = len(state)
        mass = prob.new_zeros(batch, len(bank) + 1)
        mass.scatter_add_(
            1, source_ids, attention.double().softmax(-1) * prob[:, COPY, None]
        )
        exemplars = self.prototype_scores(state, keys).double().softmax(-1)
        mass.scatter_add_(
            1, exemplar_ids.expand(batch, -1), exemplars * prob[:, EXEMPLAR, None]
        )
        mass[:, -1] = prob[:, END]
        value, choice = mass.max(-1)
        upper = mass.clone()
        upper[:, :-1] += prob[:, DRAW, None]
        upper.scatter_(1, choice[:, None], -1)
        return choice, value > torch.maximum(upper.max(-1).values, prob[:, DRAW])

    @torch.inference_mode()
    def generate(self, pictures, lengths, limit=96, *, pool_routes=False):
        if (
            limit < 1
            or pictures.ndim != 3
            or pictures.shape[-1] != 1024
            or lengths.shape != (len(pictures),)
        ):
            raise ValueError("Expected a nonempty batch of 32-by-32 glyph streams")
        if (
            not len(lengths)
            or (lengths < 1).any()
            or (lengths > pictures.size(1)).any()
        ):
            raise ValueError("Invalid source lengths")
        memory, keys, mask, h = self.read(
            visual_input(pictures, self.landmarks), lengths
        )
        image_keys = self.image_keys()
        pred = self.start.expand(len(lengths), 1, -1)
        center = lengths.new_full(lengths.shape, -0.65, dtype=torch.float)
        lengths_out = torch.zeros_like(lengths)
        stopped = torch.zeros_like(lengths, dtype=torch.bool)
        stream = pictures.new_zeros((len(lengths), limit, 1024))
        modes = lengths.new_full((len(lengths), limit), END)
        outcomes = self.image_outcomes(pictures) if pool_routes else None
        for step in range(limit):
            q, h = self.history(pred, h)
            state, a, c = self.plan_steps(q, memory, keys, mask, lengths, center)
            center = c[:, 0]
            state = state[:, 0]
            mode = self.route(state).argmax(-1)
            if outcomes is not None:
                choice, certified = self.certified_outcome(
                    state, a[:, 0], image_keys, outcomes
                )
                is_end = choice == len(outcomes[0])
                # A certified ink decision overrides even a hierarchical END.
                stop_now = torch.where(certified, is_end, mode == END)
            else:
                stop_now = mode == END
            stopped = stopped | stop_now
            modes[:, step] = mode.masked_fill(stopped, END)
            if stopped.all():
                break
            pixels = pictures[
                torch.arange(len(lengths), device=pictures.device), a[:, 0].argmax(-1)
            ].clone()
            common = (mode == EXEMPLAR) & ~stopped
            if common.any():
                pixels[common] = self.exemplars[
                    self.prototype_scores(state[common], image_keys).argmax(-1)
                ]
            paint = (mode == DRAW) & ~stopped
            if paint.any():
                bits = self.draw.choose(state[paint])
                ink = self.motor(bits) > 0
                ink[:, self.landmarks] = bits.bool()
                pixels[paint] = ink
            if outcomes is not None:
                selected = certified & ~is_end & ~stopped
                pixels[selected] = outcomes[0][choice[selected]]
                # A pooled emission has no unique latent route.
                modes[selected, step] = 4
            stream[:, step] = pixels
            lengths_out += ~stopped
            pred = self.embed(visual_input(pixels, self.landmarks))[:, None]
        return stream, lengths_out, stopped, modes
