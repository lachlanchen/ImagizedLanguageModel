"""A visual dialogue reader with word-image pointers and an autoregressive glyph writer."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.gloss_focus import cell_features
from ilm.visual_lm.optical_passage_qa import OpticalPassageQA


class DialogueInk(nn.Module):
    max_words = 16
    max_letters = 24

    def __init__(self, palette):
        super().__init__()
        self.reader = OpticalPassageQA()
        del self.reader.trim
        self.register_buffer("palette", palette.clone())
        self.width = 128
        self.kind = nn.Parameter(torch.zeros(3, 112))
        self.initial = nn.Linear(336, self.width)
        self.feedback = nn.GRU(50 + self.width, self.width, batch_first=True)
        self.mem_key = nn.Linear(112, self.width)
        self.step_key = nn.Linear(self.width, self.width)
        self.edit = nn.Linear(112, 4)
        self.transition = nn.Parameter(torch.tensor(2.0))
        self.combine = nn.Sequential(
            nn.Linear(112 + self.width, self.width), nn.LayerNorm(self.width), nn.GELU()
        )
        self.channel = nn.Linear(self.width, 3)
        self.letter_in = nn.Linear(256, 48)
        self.spell = nn.GRU(48, self.width, batch_first=True)
        self.letter_out = nn.Linear(self.width, len(palette))

    def memory(self, p, pl, q, ql, history, hl):
        h, pm, qh, qm, global_q = self.reader.contextualize(p, pl, q, ql)
        evidence = self.reader.endpoints(h).masked_fill(~pm[..., None], -1e4)
        attention = evidence.mean(-1).softmax(-1)
        support = (h * attention[..., None]).sum(1)
        hm = self.reader.mask(history, hl)
        hh = self.reader.encode(history, hm)
        old = hh.sum(1) / hl.clamp(min=1)[:, None]
        initial = self.initial(torch.cat((support, global_q, old), -1)).tanh()
        states = torch.cat((h + self.kind[0], qh + self.kind[1], hh + self.kind[2]), 1)
        # At a candidate position, expose the preceding visible word as a cue for continuing a phrase.
        preceding = torch.cat(
            [F.pad(s[:, :-1, :50], (0, 0, 1, 0)) for s in (p, q, history)], 1
        )
        return (
            states,
            self.mem_key(states),
            self.edit(states),
            torch.cat((pm, qm, hm), 1),
            preceding,
            initial,
        ), evidence

    def step(self, memory, prev, state):
        values, keys, edit, mask, preceding, initial = memory
        h, next_state = self.feedback(
            torch.cat((prev, initial[:, None].expand(-1, prev.size(1), -1)), -1), state
        )
        match = self.step_key(h) @ keys.transpose(1, 2) / self.width**0.5
        match = match + self.transition * (prev @ preceding.transpose(1, 2))
        match = match.masked_fill(~mask[:, None], -1e4)
        latent = self.combine(torch.cat((h, match.softmax(-1) @ values), -1))
        return (
            self.channel(latent),
            match[..., None] + edit[:, None],
            latent,
            next_state,
        )

    def letter_inputs(self):
        return self.letter_in(cell_features(self.palette))

    def spell_loss(self, latent, letters, emb):
        # Slot zero represents end-of-word. Pads after the first end carry no loss.
        prefix = F.pad(letters[:, :-1], (1, 0))
        h, _ = self.spell(emb[prefix], latent[None].contiguous())
        logits = self.letter_out(h)
        mask = F.pad(letters[:, :-1] != 0, (1, 0), value=True)
        return F.cross_entropy(logits[mask], letters[mask])

    @torch.no_grad()
    def write_words(self, latent):
        n = len(latent)
        result = torch.zeros(
            n, self.max_letters, dtype=torch.long, device=latent.device
        )
        state = latent[None].contiguous()
        prev = result[:, 0].clone()
        done = torch.zeros(n, dtype=torch.bool, device=latent.device)
        emb = self.letter_inputs()
        for j in range(self.max_letters + 1):
            h, state = self.spell(emb[prev, None], state)
            scores = self.letter_out(h[:, 0])
            if j == 0:
                scores[:, 0] = -torch.inf
            prev = scores.argmax(1).masked_fill(done, 0)
            done = done | (prev == 0)
            if j < self.max_letters:
                result[:, j] = prev
            if bool(done.all()):
                break
        return result, done


class WordCodeDialogueInk(DialogueInk):
    """Predict a continuous word-image representation before spelling its glyphs."""

    def __init__(self, palette):
        super().__init__(palette)
        self.register_buffer("word_mean", torch.zeros(115))
        self.register_buffer("word_scale", torch.ones(115))
        self.word_code = nn.Sequential(
            nn.Linear(self.width, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 115),
        )
        self.spell_code = nn.Sequential(
            nn.Linear(115, self.width), nn.LayerNorm(self.width), nn.Tanh()
        )

    def observed_code(self, surface):
        return F.normalize(
            ((surface - self.word_mean) / self.word_scale).clamp(-5, 5), dim=-1
        )

    def predicted_code(self, latent):
        return F.normalize(self.word_code(latent), dim=-1)

    def supplied_loss(self, code, letters, emb):
        return super().spell_loss(self.spell_code(code), letters, emb)

    def spell_loss(self, latent, letters, emb):
        return self.supplied_loss(self.predicted_code(latent), letters, emb)

    def write_words(self, latent):
        return self.write_from_code(self.predicted_code(latent))

    def write_from_code(self, code):
        return super().write_words(self.spell_code(code))
