"""A pixel-conditioned mixture over repeated page spans and newly written glyphs."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.ordered_translation import Block
from ilm.visual_lm.portable_writer import read_text, DEFAULT_FONT

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
MAX_WORD = 23
CONTEXT = 96
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/optical_passage_selector"


class Print:
    """Physical shapes for motor feedback, reading and deterministic alphabetic boundaries."""

    def __init__(self):
        self.letters = read_text(ALPHABET, DEFAULT_FONT)
        self.blank = read_text(" ", DEFAULT_FONT)[0]
        self.address = {bytes(v): i + 1 for i, v in enumerate(self.letters)}
        self.letter_id = {c: i + 1 for i, c in enumerate(ALPHABET)}
        self.coarse = (
            np.unpackbits(self.letters, axis=1)
            .reshape(52, 8, 4, 8, 4)
            .mean((2, 4), dtype=np.float32)
            .reshape(52, 64)
        )
        self.coarse = np.concatenate((np.zeros((1, 64), np.float32), self.coarse))
        self.gap_shape = {}
        for j, chars in enumerate((".!?", "\"'", ";", ",", ":", "-"), 1):
            for x in read_text(chars, DEFAULT_FONT):
                self.gap_shape[bytes(x)] = j
        self.df = json.loads((MODEL / "document_frequency.json").read_text())

    def scan(self, pixels):
        words = []
        marks = []
        current = []
        gap = 0
        for cell in pixels:
            shape = bytes(cell)
            k = self.address.get(shape)
            if k is not None:
                if not current and words:
                    marks.append(gap)
                current.append(k)
                gap = 0
            else:
                if current:
                    words.append(tuple(current))
                    current = []
                mark = 0 if shape == bytes(self.blank) else self.gap_shape.get(shape, 7)
                if mark and (not gap or mark < gap):
                    gap = mark
        if current:
            words.append(tuple(current))
            gap = 0
        if words:
            marks.append(gap)
        return words[-CONTEXT:], marks[-CONTEXT:]

    def ids(self, s):
        return tuple(self.letter_id.get(c, 0) for c in s)

    def pixels(self, seq):
        if any(k < 1 or k > 52 for k in seq):
            raise ValueError("Output requires printable letter shapes")
        return self.letters[np.array(seq, dtype=np.int64) - 1]

    def descriptor(self, seq):
        if not seq:
            return np.zeros(385, np.float32)
        ink = self.coarse[list(seq)]
        # Four normalized spatial bins preserve order across variable word lengths.
        n = len(ink)
        parts = [ink[0], ink[-1]]
        for b in range(4):
            start = b * n // 4
            parts.append(ink[start : max(start + 1, (b + 1) * n // 4)].mean(0))
        return np.concatenate((*parts, np.array([math.log1p(n) / 3], np.float32)))

    def idf(self, seq):
        s = "".join(ALPHABET[k - 1].lower() if k else "?" for k in seq)
        return math.log(97 / (1 + self.df.get(s, 0)))


class VisualBank:
    def __init__(self, face, words, device):
        self.face = face
        self.words = words
        self.device = device
        self.keys = torch.from_numpy(np.stack([face.descriptor(w) for w in words])).to(
            device
        )
        self.idf = torch.tensor(
            [face.idf(w) for w in words], device=device, dtype=torch.float32
        )
        codes = np.zeros((len(words), MAX_WORD + 1), np.int64)
        lengths = np.zeros(len(words), np.int64)
        for i, w in enumerate(words):
            if w and len(w) <= MAX_WORD and 0 not in w:
                codes[i, : len(w)] = w
                lengths[i] = len(w)
        self.codes = torch.as_tensor(codes, device=device)
        self.lengths = torch.as_tensor(lengths, device=device)


class PassageSelector(nn.Module):
    def __init__(self, coarse, width=128):
        super().__init__()
        self.read = nn.Sequential(
            nn.Linear(385, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.gap = nn.Embedding(8, width)
        self.task = nn.Embedding(2, width)
        self.layers = nn.ModuleList([Block(width) for _ in range(3)])
        self.final = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.copy_gate = nn.Linear(width, 1)
        self.initial = nn.Linear(width, width)
        self.feedback = nn.Sequential(nn.Linear(64, width), nn.LayerNorm(width))
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.spell_head = nn.Linear(width, 53)
        self.begin = nn.Parameter(torch.randn(width) * 0.02)
        self.end_head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.register_buffer("glyph_features", torch.as_tensor(coarse))
        pos = torch.arange(-CONTEXT, 1)[:, None] * torch.exp(
            -math.log(10000) * torch.arange(0, width, 2) / width
        )
        self.register_buffer(
            "position", torch.stack((pos.sin(), pos.cos()), -1).flatten(1)
        )

    def observe(self, bank, src, marks, task):
        valid = src.ne(0)
        if not valid.any(1).all():
            raise ValueError("Empty word context")
        x = self.read(bank.keys[src]) + self.gap(marks)
        x = (
            torch.cat((x, self.task(task)[:, None]), 1)
            + self.position[-(src.shape[1] + 1) :]
        )
        mask = F.pad(valid, (0, 1), value=True)[:, None, None, :]
        for block in self.layers:
            x, _ = block(x, mask)
        x = self.final(x)
        q = x[:, -1]
        score = (self.key(x[:, :-1]) * self.query(q)[:, None]).sum(
            -1
        ).float() / math.sqrt(q.shape[-1])
        # Uniform neural scores recover the corpus salience ordering up to the small smoothing term.
        score = score + (bank.idf[src] + 0.01).log()
        valid = valid & bank.lengths[src].gt(0)
        a = score.masked_fill(~valid, -1e20).softmax(-1) * valid
        g = self.copy_gate(q).float().sigmoid().squeeze(-1) * valid.any(1)
        return q, a, g

    def spell_logprob(self, q, codes, lengths):
        length = int(lengths.max()) + 1
        codes = codes[:, :length]
        inputs = torch.cat(
            (
                self.begin.expand(len(q), 1, -1),
                self.feedback(self.glyph_features[codes[:, :-1]]),
            ),
            1,
        )
        hidden, _ = self.recurrent(inputs, self.initial(q).tanh()[None].contiguous())
        score = (
            self.spell_head(hidden)
            .float()
            .log_softmax(-1)
            .gather(-1, codes[..., None])
            .squeeze(-1)
        )
        active = torch.arange(length, device=q.device)[None] <= lengths[:, None]
        return (score * active).sum(1)

    def end_logits(self, q, features):
        return self.end_head(torch.cat((q, self.read(features)), 1)).float().squeeze(1)

    def training_loss(self, bank, src, marks, task, target, end):
        q, attention, g = self.observe(bank, src, marks, task)
        mask = src.eq(target[:, None])
        present = mask.any(1)
        copy = (attention * mask).sum(1)
        log_spell = self.spell_logprob(q, bank.codes[target], bank.lengths[target])
        mixture = torch.logaddexp(
            (g * copy).clamp_min(1e-30).log(), (1 - g).clamp_min(1e-8).log() + log_spell
        )
        # Balance only strata represented in this batch.
        weights = torch.where(
            present,
            present.sum().clamp_min(1).reciprocal(),
            (~present).sum().clamp_min(1).reciprocal(),
        )
        nll = -(mixture * weights).sum() / weights.sum()
        stop = F.binary_cross_entropy_with_logits(
            self.end_logits(q, bank.keys[target]), end.float(), reduction="none"
        )
        ordinary = task.eq(0)
        return (
            nll + (stop * ordinary).sum() / ordinary.sum().clamp_min(1),
            mixture.detach(),
            present,
        )

    @torch.inference_mode()
    def beam(self, q, width=4):
        """Retain completed words separately. The final EOS uses its unmasked model probability."""
        n = len(q)
        device = q.device
        state = self.initial(q).tanh()
        previous = self.begin.expand(n, 1, -1)
        active = q.new_zeros(n, 1)
        history = torch.zeros(n, 1, MAX_WORD, dtype=torch.long, device=device)
        best = q.new_full((n,), -torch.inf)
        best_word = history[:, 0].clone()
        for j in range(MAX_WORD + 1):
            k = active.shape[1]
            hidden, h = self.recurrent(
                previous.reshape(n * k, 1, -1), state.reshape(1, n * k, -1).contiguous()
            )
            lp = self.spell_head(hidden[:, 0]).float().log_softmax(-1).reshape(n, k, 53)
            if j:
                ended, which = (active + lp[:, :, 0]).max(1)
                better = ended > best
                best_word = torch.where(
                    better[:, None],
                    history[torch.arange(n, device=device), which],
                    best_word,
                )
                best = torch.maximum(best, ended)
            if j == MAX_WORD or (best >= active.max(1).values).all():
                break
            active, idx = (active[:, :, None] + lp[:, :, 1:]).flatten(1).topk(width)
            parent, ids = idx // 52, idx % 52 + 1
            take = torch.arange(n, device=device)[:, None]
            history = history[take, parent]
            history[:, :, j] = ids
            state = h[0].reshape(n, k, -1)[take, parent]
            previous = self.feedback(self.glyph_features[ids])
        return best_word, best

    @torch.inference_mode()
    def predict(self, bank, src, marks, task):
        q, a, g = self.observe(bank, src, marks, task)
        beam, logp = self.beam(q)
        # Score each distinct visible spelling under both routes. Addresses are only a cache for images.
        host = src.cpu().tolist()
        candidates = []
        owners = []
        sources = []
        for i, row in enumerate(host):
            ids = list(
                dict.fromkeys(
                    k
                    for k in reversed(row)
                    if k and len(bank.words[k]) <= MAX_WORD and 0 not in bank.words[k]
                )
            )
            candidates.append(ids)
            owners.extend([i] * len(ids))
            sources.extend(ids)
        owner = torch.tensor(owners, device=q.device, dtype=torch.long)
        candidate = torch.tensor(sources, device=q.device, dtype=torch.long)
        scores = []
        pointers = []
        for ix in range(0, len(owner), 1024):
            row = owner[ix : ix + 1024]
            y = candidate[ix : ix + 1024]
            s = self.spell_logprob(q[row], bank.codes[y], bank.lengths[y])
            mass = (a[row] * src[row].eq(y[:, None])).sum(1) * g[row]
            scores.extend(
                torch.logaddexp(
                    mass.clamp_min(1e-30).log(), (1 - g[row]).clamp_min(1e-8).log() + s
                ).tolist()
            )
            pointers.extend(mass.tolist())
        proposals = beam.cpu().tolist()
        generate = ((1 - g).clamp_min(1e-8).log() + logp).tolist()
        result = []
        offset = 0
        for i, ids in enumerate(candidates):
            novel = tuple(c for c in proposals[i] if c)
            if ids:
                part = scores[offset : offset + len(ids)]
                j = int(np.argmax(part))
                out = bank.words[ids[j]] if part[j] >= generate[i] else novel
                pointer = bank.words[
                    ids[int(np.argmax(pointers[offset : offset + len(ids)]))]
                ]
            else:
                out = novel
                pointer = ()
            result.append((out, pointer, novel))
            offset += len(ids)
        features = torch.as_tensor(
            np.stack([bank.face.descriptor(r[0]) for r in result]), device=q.device
        )
        stops = self.end_logits(q, features).gt(0).tolist()
        return result, stops


class PassageWriter:
    def __init__(self, arm="intact", device="cpu"):
        if arm not in ("intact", "shuffled"):
            raise ValueError(arm)
        self.face = Print()
        self.device = device
        self.net = PassageSelector(self.face.coarse).to(device).eval()
        p = MODEL / arm
        from hashlib import sha256

        receipt = json.loads((p / "fit.json").read_text())
        if (
            sha256((p / "weights.safetensors").read_bytes()).hexdigest()
            != receipt["weights"]
        ):
            raise ValueError("Changed weights")
        self.net.load_state_dict(load_file(str(p / "weights.safetensors")))

    @torch.inference_mode()
    def write(self, image, final_word=True, limit=1):
        words, gaps = self.face.scan(image)
        if not words or limit < 1:
            raise ValueError("Supply an English passage and a positive output budget")
        emitted = []
        stopped = False
        for _ in range(1 if final_word else limit):
            forms = [(), *dict.fromkeys(words)]
            lookup = {s: i for i, s in enumerate(forms)}
            bank = VisualBank(self.face, forms, self.device)
            src = torch.tensor([[lookup[w] for w in words]], device=self.device)
            pred, ends = self.net.predict(
                bank,
                src,
                torch.tensor([gaps], device=self.device),
                torch.tensor([int(final_word)], device=self.device),
            )
            w = pred[0][0]
            if emitted:
                emitted.append(self.face.blank[None])
            emitted.append(self.face.pixels(w))
            words = (words + [w])[-CONTEXT:]
            gaps = (gaps + [0])[-CONTEXT:]
            stopped = ends[0]
            if final_word or stopped:
                break
        return np.concatenate(emitted), stopped
