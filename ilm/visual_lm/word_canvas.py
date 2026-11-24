"""Conditional masked prediction of answer word images in a fixed reading order."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.dialogue_surface import SurfaceDialogue
from ilm.visual_lm.image_copy_search import copy_groups


class WordCanvas(SurfaceDialogue):
    def __init__(self, palette, mode):
        super().__init__(palette, mode)
        del self.feedback
        self.length_head = nn.Linear(self.width, self.max_words)
        self.length_embedding = nn.Embedding(self.max_words, self.width)
        self.position = nn.Embedding(self.max_words, self.width)
        self.ink_projection = nn.Linear(50, self.width, bias=False)
        self.mask_image = nn.Parameter(torch.zeros(self.width))
        self.context = nn.TransformerEncoderLayer(
            self.width, 4, 2 * self.width, dropout=0.0, batch_first=True
        )
        nn.init.normal_(self.length_embedding.weight, std=0.025)
        nn.init.normal_(self.position.weight, std=0.025)

    def paint(self, memory, appearance, visible, length):
        values, keys, edit, source_mask, preceding, initial = memory
        time = torch.arange(appearance.size(1), device=length.device)
        active = time[None] < length[:, None]
        ink = torch.where(
            visible[:, :, None], self.ink_projection(appearance), self.mask_image
        )
        slots = (
            initial[:, None]
            + self.length_embedding(length - 1)[:, None]
            + self.position(time)[None]
            + ink
        )
        hidden = self.context(slots, src_key_padding_mask=~active)
        match = self.step_key(hidden) @ keys.transpose(1, 2) / math.sqrt(self.width)
        previous = F.pad((appearance * visible[:, :, None])[:, :-1], (0, 0, 1, 0))
        match = match + self.transition * (previous @ preceding.transpose(1, 2))
        match = match.masked_fill(~source_mask[:, None], -1e4)
        latent = self.combine(torch.cat((hidden, match.softmax(-1) @ values), -1))
        return (
            self.channel(latent)[:, :, 1:],
            match[:, :, :, None] + edit[:, None],
            latent,
        )

    def draw_letters(self, latent):
        # Confidence is the probability of the entire greedy spelling, including its stop.
        glyphs, ended = self.write_words(latent)
        prefix = F.pad(glyphs, (1, 0))
        h, _ = self.spell(self.letter_inputs()[prefix], latent[None].contiguous())
        logits = self.letter_out(h)
        logits[:, 0, 0] = -torch.inf
        score = (
            logits.log_softmax(-1)
            .gather(-1, F.pad(glyphs, (0, 1))[:, :, None])
            .squeeze(-1)
        )
        length = (glyphs != 0).sum(-1)
        t = torch.arange(score.size(1), device=length.device)
        score = (
            score.masked_fill(t[None] > length[:, None], 0)
            .sum(1)
            .masked_fill(~ended, -torch.inf)
        )
        feedback = self.surface(self.palette[glyphs], length)[:, :50]
        return glyphs, ended, score, feedback

    def choose_images(
        self, bank, route, pointer, latent, groups, image_table, valid, pending
    ):
        spell = (route[:, :, 1] > route[:, :, 0]) & pending
        ptr = pointer.flatten(2).masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        mass = torch.zeros_like(ptr).scatter_add_(
            2, groups[:, None].expand_as(ptr), ptr
        )
        prob, group = mass.max(2)
        word = image_table.gather(1, group)
        confidence = route[:, :, 0] + prob.log()
        new_appearance = bank.features[word, :50]
        letters = torch.zeros(
            (*word.shape, self.max_letters), device=word.device, dtype=torch.long
        )
        ended = torch.ones_like(pending)
        if spell.any():
            glyphs, stopped, score, features = self.draw_letters(latent[spell])
            letters[spell] = glyphs
            ended[spell] = stopped
            confidence[spell] = route[:, :, 1][spell] + score
            new_appearance[spell] = features
            word = word.masked_fill(spell, 0)
        return word, letters, ended, confidence, new_appearance


def canvas_loss(net, ds, indices, labels):
    inputs, variants = ds.inputs(indices)
    memory, endpoints = net.memory(*inputs)
    length = ds.al[indices]
    w = int(length.max())
    target = ds.ans[indices, :w].long()
    active = torch.arange(w, device=length.device)[None] < length[:, None]
    # Hide a random nonempty subset. A dedicated full-mask component matches deployment.
    rate = torch.rand(len(indices), 1, device=length.device)
    rate = torch.where(torch.rand_like(rate) < 0.5, 1.0, rate)
    masked = (torch.rand(target.shape, device=length.device) < rate) & active
    forced = (torch.rand_like(rate[:, 0]) * length).long()
    masked[torch.arange(len(length), device=length.device), forced] |= ~masked.any(1)
    visible = active & ~masked
    route, pointer, latent = net.paint(
        memory, ds.bank.features[target, :50], visible, length
    )
    positions = variants.flatten(1)
    matches = (positions[:, None] == target[:, :, None]) & (target[:, :, None] != 0)
    copy = matches.any(-1)
    loss = F.cross_entropy(net.length_head(memory[-1]), length - 1)
    loss = loss + F.cross_entropy(route[masked], (~copy[masked]).long())
    selected = copy & masked
    if selected.any():
        logp = (
            pointer.flatten(2)
            .masked_fill(positions[:, None] == 0, -torch.inf)[selected]
            .log_softmax(-1)
        )
        loss = (
            loss - logp.masked_fill(~matches[selected], -torch.inf).logsumexp(1).mean()
        )
    spell = masked & ~copy
    if spell.any():
        letters = labels[target[spell]].long()
        n = min(25, int((letters != 0).sum(1).max()) + 1)
        loss = loss + net.spell_loss(latent[spell], letters[:, :n], net.letter_inputs())
    ev = ds.ev[indices]
    valid = ev[:, 0] >= 0
    if valid.any():
        loss = loss + 0.20 * sum(
            F.cross_entropy(endpoints[valid, :, j], ev[valid, j]) for j in (0, 1)
        )
    return loss


@torch.inference_mode()
def draw(
    net,
    bank,
    inputs,
    variants,
    *,
    explore=False,
    supplied_lengths=None,
    commit_rounds=None,
):
    """Return ordered word-image byte sequences, completion and predicted length per input."""
    mem, _ = net.memory(*inputs)
    length_scores = net.length_head(mem[-1]).log_softmax(-1)
    count = 8 if explore else 1
    length = length_scores.argsort(descending=True, stable=True)[:, :count] + 1
    if supplied_lengths is not None:
        if explore:
            raise ValueError("Supplied lengths cannot enter autonomous search")
        length = torch.as_tensor(supplied_lengths, device=length.device)[:, None]
        if (
            length.shape != (len(variants), 1)
            or not ((length >= 1) & (length <= net.max_words)).all()
        ):
            raise ValueError("Supplied lengths outside the writer geometry")
    orders = 4 if explore else 1
    repeats = count * orders
    length = length.flatten().repeat_interleave(orders)
    mem = tuple(v.repeat_interleave(repeats, 0) for v in mem)
    ids = variants.flatten(1).long()
    groups = copy_groups(variants)
    image_table = (
        torch.zeros_like(ids).scatter_(1, groups, ids).repeat_interleave(repeats, 0)
    )
    groups = groups.repeat_interleave(repeats, 0)
    valid = (ids != 0).repeat_interleave(repeats, 0)
    n = len(length)
    width = int(length.max())
    dev = length.device
    time = torch.arange(width, device=dev)[None]
    active = time < length[:, None]
    visible = torch.zeros_like(active)
    appearance = mem[-1].new_zeros(n, width, 50)
    saved_copy = torch.zeros((n, width), device=dev, dtype=torch.long)
    saved_letters = torch.zeros(
        (n, width, net.max_letters), device=dev, dtype=torch.long
    )
    complete = torch.ones_like(active)
    timeline = torch.zeros_like(saved_copy) if commit_rounds is not None else None
    joint = mem[-1].new_zeros(n)
    order_type = torch.arange(n, device=dev) % orders
    for step in range(1, 5):
        route, pointer, latent = net.paint(mem, appearance, visible, length)
        route = route.log_softmax(-1)
        word, letters, ended, confidence, new_appearance = net.choose_images(
            bank, route, pointer, latent, groups, image_table, valid, active & ~visible
        )
        priority = confidence.clamp_min(-1e20)
        if explore:
            priority = torch.where(order_type[:, None] == 1, -time, priority)
            priority = torch.where(order_type[:, None] == 2, time, priority)
            priority = torch.where(
                order_type[:, None] == 3,
                -(2 * time - length[:, None] + 1).abs(),
                priority,
            )
        rank = (
            priority.masked_fill(visible | ~active, -torch.inf)
            .argsort(dim=1, descending=True, stable=True)
            .argsort(1)
        )
        quota = (length * step + 3) // 4 - visible.sum(1)
        commit = (rank < quota[:, None]) & active & ~visible
        if timeline is not None:
            timeline.masked_fill_(commit, step)
        visible |= commit
        appearance = torch.where(commit[:, :, None], new_appearance, appearance)
        saved_copy = torch.where(commit, word, saved_copy)
        saved_letters = torch.where(commit[:, :, None], letters, saved_letters)
        complete = torch.where(commit, ended, complete)
        joint += confidence.masked_fill(~commit, 0).sum(1)
    if not torch.equal(visible, active):
        raise RuntimeError("Unfilled answer slot")
    stopped = complete.all(1).tolist()
    score = (
        joint / length
        + length_scores.repeat_interleave(repeats, 0).gather(1, (length - 1)[:, None])[
            :, 0
        ]
    ).tolist()
    palette = [bytes(v) for v in net.palette.cpu().numpy()]
    paths = []
    for i, (copies, glyphs, size) in enumerate(
        zip(saved_copy.tolist(), saved_letters.tolist(), length.tolist(), strict=True)
    ):
        words = tuple(
            net.prototype_words[-j - 1]
            if j < 0
            else bank.raw[j]
            if j
            else b"".join(palette[k] for k in g if k)
            for j, g in zip(copies[:size], glyphs)
        )
        paths.append((words, stopped[i], score[i]))
    if timeline is not None:
        commit_rounds.extend(
            row[:size]
            for row, size in zip(timeline.tolist(), length.tolist(), strict=True)
        )
    result = []
    for offset in range(0, n, repeats):
        # Emit the predicted-length, confidence-order answer first. Retain alternative proposals separately.
        result.append(paths[offset : offset + repeats])
    return result
