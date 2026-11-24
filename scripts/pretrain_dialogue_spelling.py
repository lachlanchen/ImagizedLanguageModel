"""Learn a printable word-image inverse before fine-tuning it in dialogue."""

from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F

from scripts.measure_writer_energy import snapshot, consumed


def lexicon(bank, palette):
    symbols = {bytes(g): i for i, g in enumerate(palette.numpy())}
    indices = []
    labels = []
    for w in sorted(set(bank.raw.values())):
        if len(w) > 24 * 128:
            continue
        glyphs = [symbols.get(w[i : i + 128]) for i in range(0, len(w), 128)]
        if None in glyphs:
            continue
        indices.append(bank.bank.ids[w])
        labels.append(glyphs + [0] * (25 - len(glyphs)))
    return torch.tensor(indices, device=bank.device), torch.tensor(
        labels, dtype=torch.int16, device=bank.device
    )


def pretrain(net, bank, palette, seed):
    indices, labels = lexicon(bank, palette)
    features = bank.bank.features[indices]
    net.word_mean.copy_(features.mean(0))
    net.word_scale.copy_(features.std(0, unbiased=False).clamp(min=0.02))
    codes = net.observed_code(features)
    params = [
        p
        for m in (net.spell_code, net.letter_in, net.spell, net.letter_out)
        for p in m.parameters()
    ]
    opt = torch.optim.AdamW(params, lr=0.001)
    rng = np.random.default_rng(seed + 1)
    trace = []
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for step in range(5000):
        ids = rng.integers(len(indices), size=192)
        target = labels[ids].long()
        loss = net.supplied_loss(codes[ids], target, net.letter_inputs())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3.0)
        opt.step()
        if (step + 1) % 500 == 0:
            trace.append(float(loss.detach()))
            print(
                "word-inverse",
                step + 1,
                trace[-1],
                round(monotonic() - start, 1),
                flush=True,
            )
            if (
                monotonic() - start > 3000
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Word-image pretraining exceeded its ceiling")
    torch.cuda.synchronize()
    return dict(
        words=len(indices),
        presentations=960000,
        updates=5000,
        trace=trace,
        seconds=monotonic() - start,
        energy=consumed(meter, snapshot()),
        peak_bytes=torch.cuda.max_memory_allocated(),
    )


def alignment_loss(predicted, target, identity):
    paired = predicted @ target.T / 0.08
    matches = identity[:, None] == identity[None]
    contrast = -(
        paired.log_softmax(-1).masked_fill(~matches, -1e4).logsumexp(-1)
    ).mean()
    return 0.5 * (1 - (predicted * target).sum(-1)).mean() + 0.20 * contrast
