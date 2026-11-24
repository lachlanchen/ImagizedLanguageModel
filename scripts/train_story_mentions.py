"""Fit the visual and byte-conditioned bilingual story mention readers on GPU0."""

import hashlib
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.story_mentions import StoryMentions, losses, MAX_SPAN
from scripts.prepare_story_mentions import (
    LANGS,
    CELLS,
    RUN,
    PUBLIC,
    MODEL,
    read,
    dump,
    sha,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_prose_fields import glyph_bank
from scripts.train_semantic_page import gpu_owner, snapshot, consumed


def data(split, device):
    path = RUN / (split + ".json")
    if sha(path) != read(PUBLIC / "corpus.json")["splits"][split]["sha256"]:
        raise ValueError("Data differs from the prepared split")
    rows = read(path)
    chars = sorted(set(" " + "".join(r["text"] for r in rows)))
    lookup = {s: i for i, s in enumerate(chars)}
    bank = glyph_bank(chars, Rasterizer())
    count = len(rows)
    source = np.full((count, CELLS), lookup[" "], np.int32)
    bounds = np.empty((count, 2), np.int32)
    lengths = np.empty(count, np.int32)
    names = np.zeros((count, CELLS, MAX_SPAN), bool)
    gold = np.zeros_like(names)
    for i, r in enumerate(rows):
        text = r["text"]
        lengths[i] = len(text)
        bounds[i] = r["query"]
        source[i, : len(text)] = [lookup[c] for c in text]
        for a, b, owner in r["names"]:
            names[i, a, b - a - 1] = True
            if owner == r["entity"]:
                gold[i, a, b - a - 1] = True
    packs = tuple(
        torch.as_tensor(x, device=device)
        for x in (source, bounds, lengths, names, gold)
    )
    return rows, bank, packs


def load(mode, device):
    info = read(PUBLIC / (mode + "-fit.json"))
    path = MODEL / (mode + ".safetensors")
    if sha(path) != info["sha256"]:
        raise ValueError("Model digest mismatch")
    net = StoryMentions().to(device)
    net.load_state_dict(load_file(str(path), device=str(device)))
    return net.eval()


def train(mode, rows, bank, packs):
    receipt = PUBLIC / (mode + "-fit.json")
    if receipt.exists():
        if read(receipt)["data"] != sha(RUN / "train.json"):
            raise ValueError("Checkpoint fitted on a different corpus")
        load(mode, "cpu")
        return
    torch.manual_seed(80371)
    rng = np.random.default_rng(97031)
    src, bounds, lengths, names, gold = packs
    obs = torch.as_tensor(observations(bank, mode), device="cuda")
    pools = [
        np.array([i for i, r in enumerate(rows) if r["lang"] == lang]) for lang in LANGS
    ]
    net = StoryMentions().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=9e-4, weight_decay=0.02, fused=True)
    digest = hashlib.sha256()
    trace = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_energy = snapshot()
    start = monotonic()
    for step in range(1, 2401):
        ix = np.concatenate([rng.choice(pool, 16) for pool in pools])
        digest.update(ix.tobytes())
        logits, edges = net(obs[src[ix].long()], bounds[ix], lengths[ix])
        loss = losses(logits, edges, gold[ix], names[ix], bounds[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1600:
            opt.param_groups[0]["lr"] = 2e-4
        if step == 1 or step % 200 == 0:
            value = loss.item()
            trace.append((step, value))
            print(mode, step, value, flush=True)
            if (
                not np.isfinite(value)
                or monotonic() - start > 720
                or torch.cuda.max_memory_allocated() > 4 * 2**30
            ):
                raise RuntimeError("Training exceeded registered limits")
    torch.cuda.synchronize()
    elapsed = monotonic() - start
    energy = consumed(start_energy, snapshot())
    final = MODEL / (mode + ".safetensors")
    tmp = final.with_suffix(".tmp")
    save_file({k: v.detach().cpu() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(final)
    dump(
        receipt,
        dict(
            sha256=sha(final),
            data=sha(RUN / "train.json"),
            updates=2400,
            batch=32,
            schedule=digest.hexdigest(),
            seconds=elapsed,
            energy=energy,
            peak=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in net.parameters()),
            trace=trace,
        ),
    )


def main():
    with gpu_owner():
        MODEL.mkdir(parents=True, exist_ok=True)
        rows, bank, packs = data("train", "cuda")
        for mode in ("ink", "byte"):
            train(mode, rows, bank, packs)
        if (
            read(PUBLIC / "ink-fit.json")["schedule"]
            != read(PUBLIC / "byte-fit.json")["schedule"]
        ):
            raise ValueError("Mismatched schedules")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
