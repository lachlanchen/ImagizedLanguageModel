"""Fit two matched optical translators, with restartable epoch checkpoints and development selection."""

from pathlib import Path
from time import monotonic
import math
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.ordered_translation import OrderedTranslator
from scripts.prepare_ordered_translation import ROOT, MODEL, WORK, PUB, read, dump, sha
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli

SEED = 713581
EPOCHS = 12


def pack(rows, device):
    wide = max(map(len, rows))
    a = np.zeros((len(rows), wide), np.int32)
    for i, r in enumerate(rows):
        a[i, : len(r)] = r
    return torch.from_numpy(a).to(device=device, dtype=torch.long)


def batches(rows, epoch):
    rng = np.random.default_rng(SEED + epoch)
    groups = {}
    for i, r in enumerate(rows):
        for target in (0, 1):
            en, zh = r["cells"]
            groups.setdefault((target, len(en) // 16, len(zh) // 12), []).append(i)
    out = []
    for (target, _, _), ids in groups.items():
        order = rng.permutation(ids)
        out.extend((target, order[j : j + 96]) for j in range(0, len(ids), 96))
    return [out[j] for j in rng.permutation(len(out))]


def meters():
    return dict(nll=0.0, targets=0, pairs=0, attention_sites=0)


def run(net, features, rows, batches_, inv, maps, opt=None, epoch=0):
    stats = meters()
    net.train(opt is not None)
    for step, (t, ids) in enumerate(batches_):
        src = pack([rows[i]["cells"][1 - t] for i in ids], "cuda")
        y = pack([rows[i]["cells"][t] + [1] for i in ids], "cuda")
        labels = maps[t][y]
        with torch.set_grad_enabled(opt is not None), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            logits = net(features, src, y, inv[t], t)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        n = labels.ne(-100).sum().item()
        if opt is not None:
            progress = epoch + step / len(batches_)
            warm = min(1.0, (progress + 0.01) / 0.75)
            opt.param_groups[0]["lr"] = (
                0.0011
                * warm
                * (0.15 + 0.85 * (1 + math.cos(math.pi * progress / EPOCHS)) / 2)
            )
            opt.zero_grad(set_to_none=True)
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
        lengths = src.ne(0).sum(1)
        blocks = (lengths + net.stride - 1) // net.stride
        stats["attention_sites"] += int(
            (3 * blocks.square() + 2 * blocks * y.ne(0).sum(1)).sum()
        )
        stats["nll"] += float(loss.detach())
        stats["targets"] += n
        stats["pairs"] += len(ids)
    stats["mean_nll"] = stats["nll"] / stats["targets"]
    return stats


def main():
    configure_cuda_cli()
    paths = [WORK / (s + ".json") for s in ("train", "dev", "test")] + [
        WORK / "inputs.npy",
        MODEL / "palette.npy",
        MODEL / "inventories.json",
    ]
    hashes = read(PUB / "corpus.json")["files"]
    for p in paths:
        if sha(p) != hashes[p.name]:
            raise ValueError(p)
    srcs = [
        "ilm/visual_lm/ordered_translation.py",
        "scripts/train_ordered_translation.py",
        "scripts/prepare_ordered_translation.py",
        "references/ordered_visual_translation_protocol.md",
        "references/evidence/ordered_translation_v1/corpus.json",
    ]
    binding = {p: sha(ROOT / p) for p in srcs}
    with gpu_owner():
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        train, dev = (read(WORK / (s + ".json")) for s in ("train", "dev"))
        features = torch.from_numpy(
            observations({"tiles": np.load(WORK / "inputs.npy")}, "ink")
        ).cuda()
        inv = [torch.tensor(v, device="cuda") for v in read(MODEL / "inventories.json")]
        maps = [torch.full((len(features),), -100, device="cuda") for _ in inv]
        for a, v in zip(maps, inv):
            a[v] = torch.arange(len(v), device="cuda")
        for stride in (4, 1):
            folder = MODEL / str(stride)
            folder.mkdir(exist_ok=True)
            end = folder / "fit.json"
            weights = folder / "weights.safetensors"
            if end.exists():
                if (
                    read(end)["binding"] != binding
                    or sha(weights) != read(end)["weights"]
                ):
                    raise ValueError("Completed fit differs")
                continue
            torch.manual_seed(SEED)
            net = OrderedTranslator(stride).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=0.001, fused=True)
            resume = WORK / f"resume-{stride}.pt"
            hist = []
            best = math.inf
            selected = 0
            if resume.exists():
                state = torch.load(resume, map_location="cpu", weights_only=True)
                if state["binding"] != binding:
                    raise ValueError("Resume changed")
                net.load_state_dict(state["net"])
                opt.load_state_dict(state["opt"])
                hist, best, selected = (
                    state["history"],
                    state["best"],
                    state["selected"],
                )
                if state["bestweights"] is not None:
                    atomic_save(lambda p: save_file(state["bestweights"], p), weights)
            for epoch in range(len(hist), EPOCHS):
                start = monotonic()
                meter = snapshot()
                torch.cuda.reset_peak_memory_stats()
                tr = run(
                    net, features, train, batches(train, epoch), inv, maps, opt, epoch
                )
                val = run(net, features, dev, batches(dev, 0), inv, maps)
                torch.cuda.synchronize()
                rec = dict(
                    epoch=epoch + 1,
                    train=tr,
                    dev=val,
                    seconds=monotonic() - start,
                    joules=consumed(meter, snapshot())["joules"],
                    peak=torch.cuda.max_memory_allocated(),
                )
                check_budget(start, sum(x["seconds"] for x in hist))
                hist.append(rec)
                if val["mean_nll"] < best:
                    best = val["mean_nll"]
                    selected = epoch + 1
                    atomic_save(
                        lambda p: save_file(
                            {
                                k: v.detach().cpu().contiguous()
                                for k, v in net.state_dict().items()
                            },
                            p,
                        ),
                        weights,
                    )
                atomic_save(
                    lambda p: torch.save(
                        dict(
                            binding=binding,
                            net=net.state_dict(),
                            opt=opt.state_dict(),
                            bestweights=load_file(str(weights)),
                            history=hist,
                            best=best,
                            selected=selected,
                        ),
                        p,
                    ),
                    resume,
                )
                print(
                    stride,
                    epoch + 1,
                    round(tr["mean_nll"], 3),
                    round(val["mean_nll"], 3),
                    round(rec["seconds"], 1),
                    flush=True,
                )
            dump(
                end,
                dict(
                    stride=stride,
                    weights=sha(weights),
                    binding=binding,
                    epochs=hist,
                    selected_epoch=selected,
                    parameters=sum(p.numel() for p in net.parameters()),
                ),
            )
            del net, opt


if __name__ == "__main__":
    main()
