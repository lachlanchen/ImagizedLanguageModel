"""Transfer the pictured Chinese writer to joint one- and two-field reply generation."""

import hashlib
import math
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.fact_sequence import FactSequence
from ilm.visual_lm.semantic_page import BLOCK
from scripts.prepare_chinese_two_fact import (
    PUB,
    MODEL,
    PARENT,
    WORK,
    SOURCE,
    FIELDS,
    WIDTH,
    sha,
    dump,
    read,
)
from scripts.train_chinese_card_reply import load_writer
from scripts.train_semantic_page import archive, gpu_owner, snapshot, consumed

ARMS = ("ink-natural", "ink-joined", "byte-joined")


def output_bank(device):
    path = MODEL / "palette.npy"
    meta = read(PUB / "corpus.json")
    if sha(path) != meta["palette"]:
        raise ValueError("Palette checksum differs")
    return torch.as_tensor(observations({"tiles": np.load(path)}, "ink"), device=device)


def initialize(classes, mode):
    parent = load_writer(mode, "cpu")
    net = FactSequence(classes, len(FIELDS), WIDTH)
    with torch.no_grad():
        dest = net.state_dict()
        for key, src in parent.state_dict().items():
            if key in ("output.3.weight", "output.3.bias"):
                dest[key][: len(src)].copy_(src)
                dest[key][len(src)].copy_(src[-1])
            else:
                dest[key].copy_(src)
    return net


def load(arm, device):
    receipt = read(PUB / (arm + "-fit.json"))
    path = MODEL / (arm + ".safetensors")
    if sha(path) != receipt["weights"]:
        raise ValueError("Writer hash mismatch")
    net = FactSequence(receipt["classes"], len(FIELDS), WIDTH).to(device)
    net.load_state_dict(load_file(str(path), device=str(device)))
    return net.eval()


def fit(arm, bank, train, joined, meta):
    path = MODEL / (arm + ".safetensors")
    receipt = PUB / (arm + "-fit.json")
    if receipt.exists():
        load(arm, "cpu")
        return
    mode = arm.split("-")[0]
    torch.manual_seed(6745)
    rng = np.random.default_rng(6745)
    net = initialize(meta["classes"], mode).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.00035, weight_decay=0.02, fused=True)
    cells = torch.as_tensor(observations(bank, mode), device="cuda")
    labels = cells[bank["labels"]]
    palette = output_bank("cuda")
    split = np.flatnonzero(train["fields"][:, 1] != len(FIELDS))[0]
    pools = [
        np.arange(split),
        np.arange(split, len(train["fields"])),
        np.arange(len(joined["fields"])),
    ]
    data = [
        {k: torch.as_tensor(v, device="cuda") for k, v in rows.items()}
        for rows in (train, joined)
    ]
    digest = hashlib.sha256()
    total = 0
    second_id = len(read(PARENT / "readback.json"))
    start = monotonic()
    energy = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for t in range(2400):
        group = (0, 1, 0, 2 if arm.endswith("joined") else 1)[t % 4]
        source = data[int(group == 2)]
        indices = rng.choice(pools[group], size=48)
        digest.update(bytes([group]))
        digest.update(indices.tobytes())
        fields = source["fields"][indices].long()
        y = source["answer"][indices].long()
        used = (y >= 0).sum(1)
        length = math.ceil(used.max().item() / BLOCK) * BLOCK
        y = y[:, :length]
        y = F.pad(y, (0, length - y.shape[1]), value=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b, _, mem, vis = net.plan(
                cells[source["query"][indices].long()],
                net.encode_rows(labels),
                source["domain"][indices].long(),
                source["available"][indices],
                fields,
            )
            logits = net(mem, vis, palette[y.clamp_min(0)])
            logits[:, :, second_id] = logits[:, :, second_id].masked_fill(
                ~vis[:, 3, None], -torch.inf
            )
            loss = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(), ignore_index=-1
            ) + 0.6 * (
                F.cross_entropy(a, fields[:, 0]) + F.cross_entropy(b, fields[:, 1])
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.param_groups[0]["lr"] = (
            0.00035
            * min(1.0, (t + 1) / 60)
            * (0.10 + 0.9 * (1 + math.cos(math.pi * t / 2400)) / 2)
        )
        opt.step()
        total += used.sum().item()
        if t % 200 == 0:
            print(arm, t, round(float(loss), 3), flush=True)
            if (
                not torch.isfinite(loss)
                or monotonic() - start > 600
                or torch.cuda.max_memory_allocated() > 2**30
            ):
                raise RuntimeError("Fit exceeded registered limits")
    torch.cuda.synchronize()
    elapsed = monotonic() - start
    energy = consumed(energy, snapshot())
    tmp = path.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(path)
    dump(
        receipt,
        dict(
            classes=meta["classes"],
            weights=sha(path),
            updates=2400,
            positions=total,
            examples=115200,
            schedule=digest.hexdigest(),
            seconds=elapsed,
            energy=energy,
            peak=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in net.parameters()),
            parent=read(SOURCE / (mode + "-fit.json"))["weights"],
        ),
    )


def main():
    meta = read(PUB / "corpus.json")
    bank = archive(WORK / "bank.npz", meta["bank"])
    train, joined = [
        archive(WORK / (s + ".npz"), meta["archives"][s]) for s in ("train", "joined")
    ]
    with gpu_owner():
        for arm in ARMS:
            fit(arm, bank, train, joined, meta)
    if (
        read(PUB / "ink-joined-fit.json")["schedule"]
        != read(PUB / "byte-joined-fit.json")["schedule"]
    ):
        raise ValueError("Byte control saw a different schedule")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
