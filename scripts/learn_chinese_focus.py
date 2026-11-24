"""Learn which of two mentioned fields a pictured Chinese question actually requests."""

from functools import lru_cache
from time import monotonic
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from scripts.prepare_chinese_card_reply import (
    PUB,
    MODEL,
    FIELDS,
    WIDTH,
    Rasterizer,
    read,
    dump,
    sha,
)
from scripts.write_chinese_card_reply import pack
from ilm.visual_lm.book_prefix import observations
from scripts.train_semantic_page import gpu_owner, snapshot, consumed

# Each entry specifies the first or second marked field as the requested one.
FIT = (
    ("我不要看A，我想看B。", 1),
    ("我想看A，我不要看B。", 0),
    ("我需要A，而不是B。", 0),
    ("不是要A，我需要B。", 1),
    ("不必查A，请帮我查B。", 1),
    ("请帮我查A，不必查B。", 0),
    ("A已经知道了，能说一下B吗？", 1),
    ("能说一下A吗？B已经知道了。", 0),
    ("我要问的是A，不用提B。", 0),
    ("不用提A，我要问的是B。", 1),
    ("请问A如何？B就算了。", 0),
    ("A就算了，请问B如何？", 1),
    ("之前说了A，我现在关心B。", 1),
    ("我现在关心A，之前说了B。", 0),
    ("A不重要，告诉我B吧。", 1),
    ("告诉我A吧，B不重要。", 0),
    ("先查A，B以后再说。", 0),
    ("A以后再说，先查B。", 1),
    ("只要A，不需要B。", 0),
    ("不需要A，只要B。", 1),
)
FRESH = (
    ("我只问A，B可以略过。", 0),
    ("A可以略过，我只问B。", 1),
    ("暂时别管A，麻烦报一下B。", 1),
    ("麻烦报一下A，暂时别管B。", 0),
    ("与A相比，我更关心B。", 1),
    ("我更关心A，B排在其次。", 0),
    ("A我已经记下，剩下想了解的是B。", 1),
    ("剩下想了解的是A，B我已经记下。", 0),
    ("问的是A而非B。", 0),
    ("并非要打听A，要打听的是B。", 1),
    ("我问A，B先放一放。", 0),
    ("A先放一放，我问B。", 1),
)


@lru_cache(maxsize=1)
def patterns():
    labels = np.load(MODEL / "label-pixels.npy")[: len(FIELDS)]
    return [(i, row[: len(k)]) for i, (row, k) in enumerate(zip(labels, FIELDS))]


def mask_mentions(row, available):
    spans = []
    for index, pixels in patterns():
        if not available[index]:
            continue
        for start in np.flatnonzero((row == pixels[0]).all(1)):
            if np.array_equal(row[start : start + len(pixels)], pixels):
                spans.append((int(start), len(pixels), index))
    if len(spans) != 2 or spans[0][2] == spans[1][2]:
        return None
    spans.sort()
    before, after = spans
    if before[0] + before[1] > after[0]:
        return None
    # Replace variable-width names with the identical single field marker.
    marker = patterns()[0][1][:1]
    normalized = np.concatenate(
        (
            row[: before[0]],
            marker,
            row[before[0] + before[1] : after[0]],
            marker,
            row[after[0] + after[1] :],
        )
    )
    normalized = np.pad(normalized, ((0, WIDTH - len(normalized)), (0, 0)))
    return normalized, [before[0], after[0] - before[1] + 1], [before[2], after[2]]


class FieldFocus(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(256, 48)
        self.context = nn.GRU(48, 48, batch_first=True, bidirectional=True)
        self.score = nn.Linear(96, 1)

    def forward(self, ink, positions):
        ctx, _ = self.context(F.gelu(self.cell(ink)))
        seen = ctx[torch.arange(len(ctx), device=ctx.device)[:, None], positions]
        return self.score(seen).squeeze(-1)


def tensors(rows, positions, device):
    ink = observations({"tiles": np.stack(rows).reshape(-1, 128)}, "ink")
    return torch.as_tensor(ink, device=device).reshape(
        len(rows), WIDTH, 256
    ), torch.as_tensor(positions, device=device)


def fit():
    if (PUB / "focus-fit.json").exists():
        load_focus("cpu")
        return
    raster = Rasterizer()
    rng = np.random.default_rng(7821)
    torch.manual_seed(7821)
    pool = list(FIELDS[:-2])
    observations_ = []
    positions = []
    truth = []
    for frame, target in FIT:
        for prefix in ("", "好的，", "嗯，", "那麻烦你，", "我想了解一下，", "谢谢！"):
            for suffix in ("", "谢谢。", "我想确认一下。", "可以吗？"):
                a, b = rng.choice(pool, 2, replace=False)
                fields = [FIELDS.index(a), FIELDS.index(b)]
                query = prefix + frame.replace("A", a).replace("B", b) + suffix
                row, pos, which = mask_mentions(
                    pack(query, raster), [True] * len(FIELDS)
                )
                observations_.append(row)
                positions.append(pos)
                truth.append(which.index(fields[target]))
    with gpu_owner():
        x, pos = tensors(observations_, positions, "cuda")
        y = torch.tensor(truth, device="cuda")
        net = FieldFocus().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=0.0018)
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        energy = snapshot()
        for step in range(1000):
            ix = rng.integers(len(y), size=48)
            loss = F.cross_entropy(net(x[ix], pos[ix]), y[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0 and (
                not torch.isfinite(loss)
                or monotonic() - start > 180
                or torch.cuda.max_memory_allocated() > 2**30
            ):
                raise RuntimeError("Focus resource ceiling")
        torch.cuda.synchronize()
        p = MODEL / "focus.safetensors"
        tmp = p.with_suffix(".tmp")
        save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
        tmp.replace(p)
        dump(
            PUB / "focus-fit.json",
            dict(
                weights=sha(p),
                forms=len(FIT),
                rendered=len(y),
                updates=1000,
                parameters=sum(p.numel() for p in net.parameters()),
                seconds=monotonic() - start,
                energy=consumed(energy, snapshot()),
                peak=torch.cuda.max_memory_allocated(),
                held_fields=list(FIELDS[-2:]),
            ),
        )


def load_focus(device):
    p = MODEL / "focus.safetensors"
    if sha(p) != read(PUB / "focus-fit.json")["weights"]:
        raise ValueError("Focus weights differ")
    net = FieldFocus().to(device)
    net.load_state_dict(load_file(str(p), device=device))
    return net.eval()


@torch.inference_mode()
def reroute(net, physical, available, base_choice):
    result = base_choice.clone()
    items = [
        (i, mask_mentions(q, a)) for i, (q, a) in enumerate(zip(physical, available))
    ]
    used = [(i, item) for i, item in items if item is not None]
    if used:
        x, pos = tensors(
            [v[0] for _, v in used], [v[1] for _, v in used], base_choice.device
        )
        choice = net(x, pos).argmax(1).tolist()
        for (i, v), j in zip(used, choice):
            result[i] = int(v[2][j])
    return result, len(used)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    fit()
