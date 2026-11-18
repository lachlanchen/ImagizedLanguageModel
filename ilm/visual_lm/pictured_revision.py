"""Revise a pictured draft using the visual context of both passages."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import visual_input


class PicturedReviser(nn.Module):
    def __init__(self, bank, landmarks, candidates, recipes, sizes, priors):
        super().__init__()
        for name, value in dict(
            bank=bank,
            landmarks=landmarks,
            candidates=candidates,
            recipes=recipes,
            sizes=sizes,
            priors=priors,
        ).items():
            self.register_buffer(name, value)
        self.register_buffer("margin", torch.tensor(0.0))
        bits = (
            ((bank[:, :, None].long() >> torch.arange(7, -1, -1)) & 1).bool().flatten(1)
        )
        feats = visual_input(bits, landmarks)
        self.register_buffer("features", feats, persistent=False)
        components = feats[recipes.long()]
        mask = torch.arange(recipes.size(1))[None] < sizes[:, None]
        flat = (components * mask[:, :, None]).flatten(1)
        # Store a reproducible projection of ordered ink and length. The model
        # predicts in this shared visual space, including a descriptor for deletion.
        gen = torch.Generator().manual_seed(3197)
        projected = flat @ (
            torch.randn(flat.size(1), 64, generator=gen) / flat.size(1) ** 0.5
        )
        self.register_buffer(
            "descriptors",
            F.normalize(
                torch.cat([projected, F.one_hot(sizes.long(), 5).float()], 1), dim=1
            ),
            persistent=False,
        )
        self.indices = {bytes(v): i for i, v in enumerate(bank.numpy()) if i}
        width = 128
        self.ink = nn.Sequential(nn.Linear(112, width), nn.LayerNorm(width), nn.GELU())
        self.reader = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.draft_reader = nn.GRU(
            width, width // 2, batch_first=True, bidirectional=True
        )
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.combine = nn.Sequential(
            nn.Linear(3 * width, width), nn.LayerNorm(width), nn.GELU()
        )
        self.shape = nn.Linear(width, 69)
        self.change = nn.Linear(width, 1)

    def encode(self, feat, lens, rnn):
        x = self.ink(feat)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        memory, _ = rnn(packed)
        mem, _ = nn.utils.rnn.pad_packed_sequence(
            memory, batch_first=True, total_length=x.size(1)
        )
        return mem + x

    def forward(self, source, n, draft, m, draft_ids):
        a = self.encode(source, n, self.reader)
        b = self.encode(draft, m, self.draft_reader)
        x = torch.arange(a.size(1), device=a.device)
        j = torch.arange(b.size(1), device=b.device)
        center = (j[None] + 0.5) * n[:, None] / m[:, None] - 0.5
        score = (self.query(b) @ self.key(a).transpose(1, 2)) * a.size(-1) ** -0.5
        score = (score.float() - 0.4 * (x - center[:, :, None]).abs()).masked_fill(
            x >= n[:, None, None], -torch.inf
        )
        context = score.softmax(-1).to(a.dtype) @ a
        state = self.combine(torch.cat((b, context, context * b), -1))
        ids = self.candidates[draft_ids.long()].long()
        shape = self.descriptors[ids.clamp_min(0)]
        logits = (self.shape(state).unsqueeze(-2) * shape).sum(-1)
        logits = logits + self.priors[draft_ids.long()]
        gate = self.change(state)
        logits = torch.cat((logits[:, :, :1], logits[:, :, 1:] + gate), dim=2)
        return logits.masked_fill(ids < 0, -torch.inf)

    @classmethod
    def restore(cls, path):
        from safetensors.torch import load_file

        state = load_file(str(path))
        model = cls(
            *(
                state[k]
                for k in (
                    "bank",
                    "landmarks",
                    "candidates",
                    "recipes",
                    "sizes",
                    "priors",
                )
            )
        )
        model.load_state_dict(state)
        return model.eval()

    @torch.inference_mode()
    def revise(self, source, draft):
        for stream in (source, draft):
            if stream.dtype != np.uint8 or stream.ndim != 2 or stream.shape[1] != 128:
                raise ValueError("Expected complete packed 32-by-32 cells")
        if not len(source):
            raise ValueError("Empty classical source")
        if not len(draft):
            return draft.copy()
        device = self.bank.device

        def view(cells):
            pixels = torch.from_numpy(np.unpackbits(cells, axis=1)).to(device)
            return visual_input(pixels, self.landmarks)[None]

        ids = torch.tensor(
            [self.indices.get(bytes(v), 0) for v in draft], device=device
        )
        scores = self(
            view(source),
            ids.new_tensor([len(source)]),
            view(draft),
            ids.new_tensor([len(draft)]),
            ids[None],
        )[0]
        scores[:, 0] += self.margin
        actions = scores.argmax(-1)
        pieces = self.candidates[ids, actions].long()
        parts = self.bank[self.recipes[pieces].long()].cpu().numpy()
        sizes = self.sizes[pieces].tolist()
        return np.concatenate(
            [
                part[:size] if action else cell[None]
                for part, size, action, cell in zip(
                    parts, sizes, actions.tolist(), draft
                )
            ]
        )
