"""Select a contiguous observed ink phrase using contextual source-image features."""

import gzip
import zlib

import msgpack
import numpy as np
import torch
from ilm.visual_lm.idiom_ink import visual_input


def unit(x):
    return x / np.maximum(1e-8, np.linalg.norm(x, axis=-1, keepdims=True))


class ContextualPhraseRealizer:
    def __init__(self, encoder, composer, heads):
        self.encoder = encoder.eval()
        self.composer = composer
        self.heads = heads

    @torch.inference_mode()
    def features(self, seq, positions):
        bits = torch.from_numpy(np.unpackbits(seq, axis=1)).bool()[None]
        dev = next(self.encoder.parameters()).device
        bits = bits.to(dev)
        mem, _, _, _ = self.encoder.read(
            visual_input(bits, self.encoder.landmarks),
            torch.tensor([len(seq)], device=dev),
        )
        mem = mem[0].float().cpu().numpy()
        glyphs = [zlib.crc32(c) for c in seq]
        out = []
        for i in positions:
            # Signed feature hashing pools identical visible neighbours. The
            # centre is omitted, since all alternatives belong to the same image.
            sparse = np.zeros(768, np.float32)
            for j, v in enumerate(glyphs):
                if j == i:
                    continue
                side = int(j > i)
                dist = abs(j - i)
                for group, magnitude in [
                    (0, 1),
                    (1 + side, 1),
                    (3 + side, 1.5 if dist == 1 else 0),
                ]:
                    bucket = (v + 2654435761 * group) % len(sparse)
                    sparse[bucket] += (1 if v & 0x80000000 else -1) * magnitude
            out.append((mem[i], unit(sparse)))
        return out

    def generate(self, packed, *, majority=False):
        positions = [i for i, v in enumerate(packed) if bytes(v) in self.heads]
        if not positions:
            return self.composer.generate(packed)
        features = None if majority else self.features(packed, positions)
        parts = []
        begin = 0
        for j, i in enumerate(positions):
            options, mean, weights, default = self.heads[bytes(packed[i])]
            if majority:
                choice = default
            else:
                dense, sparse = features[j]
                phi = np.r_[1.0, unit(dense - mean), 1.5 * sparse]
                choice = int((phi @ weights).argmax())
            if options[choice] is None:
                continue
            if begin < i:
                parts.append(self.composer.generate(packed[begin:i]))
            parts.append(np.frombuffer(options[choice], np.uint8).reshape(-1, 128))
            begin = i + 1
        if begin < len(packed):
            parts.append(self.composer.generate(packed[begin:]))
        return np.concatenate(parts) if parts else np.empty((0, 128), np.uint8)

    def save(self, path):
        with gzip.open(path, "xb") as f:
            msgpack.pack(
                (
                    1,
                    [
                        (
                            k,
                            opts,
                            m.astype("<f4").tobytes(),
                            w.astype("<f4").tobytes(),
                            prior,
                        )
                        for k, (opts, m, w, prior) in self.heads.items()
                    ],
                ),
                f,
                use_bin_type=True,
            )

    @classmethod
    def restore(cls, path, encoder, composer):
        with gzip.open(path, "rb") as f:
            version, rows = msgpack.unpack(f, raw=False)
        if version != 1:
            raise ValueError("Unknown physical phrase format")
        heads = {
            k: (
                opts,
                np.frombuffer(m, "<f4"),
                np.frombuffer(w, "<f4").reshape(-1, len(opts)),
                prior,
            )
            for k, opts, m, w, prior in rows
        }
        return cls(encoder, composer, heads)


def fit_head(samples):
    """Class-balanced ridge regression on centered recurrent and visual bag features."""
    dense = np.stack([d for d, _, _ in samples])
    mean = dense.mean(0)
    sparse = np.stack([s for _, s, _ in samples])
    labels = [g for _, _, g in samples]
    options = [None] + sorted(set(labels) - {None})
    ids = np.array([options.index(g) for g in labels])
    counts = np.bincount(ids, minlength=len(options))
    if not counts.all():
        raise ValueError(
            "Each observed centre needs a pass example and at least one translation"
        )
    phi = np.column_stack((np.ones(len(ids)), unit(dense - mean), 1.5 * sparse))
    weights = np.sqrt(len(ids) / (len(options) * counts[ids]))[:, None]
    X = phi * weights
    Y = np.eye(len(options))[ids] * weights
    if X.shape[0] < X.shape[1]:
        dual = X @ X.T
        dual.flat[:: len(dual) + 1] += 0.5
        coef = X.T @ np.linalg.solve(dual, Y)
    else:
        gram = X.T @ X
        gram.flat[:: len(gram) + 1] += 0.5
        coef = np.linalg.solve(gram, X.T @ Y)
    return options, mean, coef.astype(np.float32), 1 + int(counts[1:].argmax())
