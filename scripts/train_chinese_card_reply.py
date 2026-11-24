"""Jointly learn optical question routing and autoregressive human Chinese replies."""

import hashlib
import math
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from ilm.visual_lm.semantic_page import SemanticPage, BLOCK
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_chinese_card_reply import (
    WORK,
    PUB,
    MODEL,
    FIELDS,
    WIDTH,
    sha,
    dump,
)
from scripts.train_semantic_page import archive, read, gpu_owner, snapshot, consumed


def output_bank(device):
    if sha(MODEL / "palette.npy") != read(PUB / "corpus.json")["palette"]:
        raise ValueError("Changed Chinese output palette")
    return torch.as_tensor(
        observations({"tiles": np.load(MODEL / "palette.npy")}, "ink"), device=device
    )


def router(net, query, label_states, domains, available, teacher=None):
    q = net.encode_rows(query)
    domain = label_states[len(FIELDS) + domains]
    logits = net.concept(q + domain).float().masked_fill(~available, -torch.inf)
    pick = logits.argmax(1) if teacher is None else teacher
    memory = torch.stack((q, domain, label_states[pick]), 1)
    return logits, memory


def load_writer(mode, device):
    info = read(PUB / (mode + "-fit.json"))
    path = MODEL / (mode + ".safetensors")
    if sha(path) != info["weights"]:
        raise ValueError("Chinese writer hash differs")
    model = SemanticPage(info["classes"], len(FIELDS), WIDTH).to(device)
    model.load_state_dict(load_file(str(path), device=device))
    return model.eval()


def fit(mode, bank, data, meta):
    receipt = PUB / (mode + "-fit.json")
    if receipt.exists():
        load_writer(mode, "cpu")
        return
    torch.manual_seed(40171)
    rng = np.random.default_rng(40171)
    ink = torch.as_tensor(observations(bank, mode), device="cuda")
    labels = ink[bank["labels"]]
    out = output_bank("cuda")
    net = SemanticPage(len(out), len(FIELDS), WIDTH).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, fused=True, weight_decay=0.02)
    tensors = {k: torch.as_tensor(v, device="cuda") for k, v in data.items()}
    sizes = (data["answer"] >= 0).sum(1)
    groups = (sizes - 1) // 16
    pools = [np.flatnonzero(groups == i) for i in range(groups.max() + 1)]
    total_examples = 0
    positions = 0
    visited = np.zeros(len(sizes), bool)
    steps = 8000
    clock = monotonic()
    energy = snapshot()
    digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        batch = rng.choice(pools[groups[rng.integers(len(sizes))]], 48)
        digest.update(batch.tobytes())
        visited[batch] = True
        width = math.ceil(sizes[batch].max() / BLOCK) * BLOCK
        y = tensors["answer"][batch, :width].long()
        y = F.pad(y, (0, width - y.shape[1]), value=-1)
        gold = tensors["label"][batch].long()
        available = tensors["available"][batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            route, mem = router(
                net,
                ink[tensors["question"][batch].long()],
                net.encode_rows(labels),
                tensors["domain"][batch].long(),
                available,
                gold,
            )
            visible = torch.ones(mem.shape[:2], dtype=torch.bool, device="cuda")
            logits = net(mem, visible, out[y.clamp_min(0)])
            loss = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(), ignore_index=-1
            ) + 0.6 * F.cross_entropy(route, gold)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.param_groups[0]["lr"] = (
            0.001
            * min(1.0, step / 180)
            * (0.12 + 0.88 * (1 + math.cos(math.pi * step / steps)) / 2)
        )
        opt.step()
        positions += int(sizes[batch].sum())
        total_examples += len(batch)
        if step % 500 == 0:
            print(mode, step, loss.item(), round(monotonic() - clock, 1), flush=True)
            if (
                not torch.isfinite(loss)
                or monotonic() - clock > 900
                or torch.cuda.max_memory_allocated() > 8 * 2**30
            ):
                raise RuntimeError("Chinese training exceeded registered ceiling")
    torch.cuda.synchronize()
    path = MODEL / (mode + ".safetensors")
    tmp = path.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(path)
    dump(
        receipt,
        dict(
            weights=sha(path),
            classes=len(out),
            training=meta["archives"]["train"],
            steps=steps,
            exposures=total_examples,
            positions=positions,
            distinct=int(visited.sum()),
            schedule=digest.hexdigest(),
            seconds=monotonic() - clock,
            energy=consumed(energy, snapshot()),
            peak=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in net.parameters()),
        ),
    )


def main():
    meta = read(PUB / "corpus.json")
    data = archive(WORK / "train.npz", meta["archives"]["train"])
    bank = archive(WORK / "bank.npz", meta["bank"])
    with gpu_owner():
        for mode in ("ink", "byte"):
            fit(mode, bank, data, meta)
    if (
        read(PUB / "ink-fit.json")["schedule"]
        != read(PUB / "byte-fit.json")["schedule"]
    ):
        raise RuntimeError("Different Chinese fit schedules")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
