"""Learn ordered or unordered requests from ink, sharing the original joint writer's schedule."""

import hashlib
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.field_alignment import FieldAlignment, target_loss
from scripts.prepare_chinese_two_fact import (
    JOINED_SCHEDULE,
    MODEL,
    PUB,
    WORK,
    FIELDS,
    DOMAINS,
    read,
    dump,
    sha,
)
from scripts.train_semantic_page import archive, gpu_owner, snapshot, consumed


def supervision(rows, joined):
    targets = []
    ordered = []
    for r in rows:
        keys = r["keys"]
        literal = len(keys) == 2 and all(r["q"].count(k) == 1 for k in keys)
        if literal and not joined:
            keys = sorted(keys, key=r["q"].index)
        targets.append(
            [
                FIELDS.index(keys[0]),
                FIELDS.index(keys[1]) if len(keys) == 2 else len(FIELDS),
            ]
        )
        ordered.append(joined or literal)
    return targets, ordered


def load_alignment(mode, device):
    path = MODEL / ("alignment-" + mode + ".safetensors")
    if sha(path) != read(PUB / ("alignment-" + mode + "-fit.json"))["weights"]:
        raise ValueError("Bad alignment weights")
    model = FieldAlignment(len(FIELDS), len(DOMAINS)).to(device)
    model.load_state_dict(load_file(str(path), device=device))
    return model.eval()


def fit(mode, bank, tensors, pools):
    dest = PUB / ("alignment-" + mode + "-fit.json")
    if dest.exists():
        load_alignment(mode, "cpu")
        return
    torch.manual_seed(993)
    rng = np.random.default_rng(6745)
    model = FieldAlignment(len(FIELDS), len(DOMAINS)).cuda()
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.0013, weight_decay=0.01, fused=True
    )
    cells = torch.as_tensor(observations(bank, mode), device="cuda")
    digest = hashlib.sha256()
    start = monotonic()
    energy = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for t in range(2400):
        group = (0, 1, 0, 2)[t % 4]
        data = tensors[int(group == 2)]
        ix = rng.choice(pools[group], size=48)
        digest.update(bytes([group]))
        digest.update(ix.tobytes())
        scores = model(
            cells[data["query"][ix].long()],
            data["domain"][ix].long(),
            data["available"][ix],
        )
        loss = target_loss(scores, data["truth"][ix], data["ordered"][ix], len(FIELDS))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if t == 1600:
            opt.param_groups[0]["lr"] = 0.00035
        if t % 300 == 0:
            print(mode, t, float(loss), flush=True)
            if (
                not torch.isfinite(loss)
                or monotonic() - start > 300
                or torch.cuda.max_memory_allocated() > 2**30
            ):
                raise RuntimeError("Alignment fit exceeded ceiling")
    torch.cuda.synchronize()
    elapsed = monotonic() - start
    cost = consumed(energy, snapshot())
    if digest.hexdigest() != JOINED_SCHEDULE:
        raise ValueError("Changed schedule")
    path = MODEL / ("alignment-" + mode + ".safetensors")
    tmp = path.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, str(tmp))
    tmp.replace(path)
    dump(
        dest,
        dict(
            weights=sha(path),
            parameters=sum(p.numel() for p in model.parameters()),
            updates=2400,
            seconds=elapsed,
            energy=cost,
            peak=torch.cuda.max_memory_allocated(),
            schedule=digest.hexdigest(),
        ),
    )


def main():
    meta = read(PUB / "corpus.json")
    bank = archive(WORK / "bank.npz", meta["bank"])
    with gpu_owner():
        tensors = []
        for split in ("train", "joined"):
            raw = archive(WORK / (split + ".npz"), meta["archives"][split])
            fields, order = supervision(
                read(WORK / (split + ".json")), split == "joined"
            )
            data = {
                **{k: raw[k] for k in ("query", "domain", "available")},
                "truth": fields,
                "ordered": order,
            }
            tensors.append(
                {k: torch.as_tensor(v, device="cuda") for k, v in data.items()}
            )
        n, m = meta["splits"]["train"]
        pools = (np.arange(n), np.arange(n, n + m), np.arange(meta["joined"]))
        for mode in ("ink", "byte"):
            fit(mode, bank, tensors, pools)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
