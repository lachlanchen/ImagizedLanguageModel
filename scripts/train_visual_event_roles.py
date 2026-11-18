"""Train two matched image writers with original or exchanged named participants."""

import fcntl
import hashlib
import time

import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.idiom_ink import visual_input
from scripts.prepare_visual_event_roles import OUT, ROOT, swap, sha, load, write_new
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_causal_glyph_translator import prepare
from scripts.train_paired_student import restore
from scripts.train_visible_image_likelihood import emission_loss

PARENT = ROOT / "models/visual_edit_zh/marginal_glyph.safetensors"


def raster(text, renderer):
    return np.frombuffer(b"".join(map(renderer.tile, text)), np.uint8).reshape(-1, 128)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(61841)
    torch.set_float32_matmul_precision("high")
    data = load(OUT / "train.json")
    assert (
        sha(OUT / "train.json")
        == load(OUT / "preparation.json")["splits"]["train"]["sha256"]
    )
    assert (
        sha(PARENT)
        == load(ROOT / "references/evidence/visible_likelihood_20261119/fit.json")[
            "arms"
        ]["marginal"]["sha256"]
    )
    base = restore(PARENT)
    renderer = Rasterizer()
    ex = np.packbits(base.exemplars.numpy(), axis=1)
    bank = list(map(bytes, ex))
    index = {cell: i for i, cell in enumerate(bank)}

    def encode(s):
        ids = []
        for cell in map(bytes, raster(s, renderer)):
            if cell not in index:
                index[cell] = len(bank)
                bank.append(cell)
            ids.append(index[cell])
        return ids

    original = [dict(x=encode(r["source"]), z=encode(r["target"])) for r in data]
    reversed_pairs = [
        dict(
            x=encode(swap(r["source"], r["a"], r["b"])),
            z=encode(swap(r["target"], r["a"], r["b"])),
        )
        for r in data
    ]
    bits = (
        torch.from_numpy(
            np.unpackbits(
                np.frombuffer(b"".join(bank), np.uint8).reshape(-1, 128), axis=1
            )
        )
        .cuda()
        .bool()
    )
    features = visual_input(bits, base.landmarks.cuda())
    lookup = torch.full((len(bank),), -1, device="cuda", dtype=torch.long)
    lookup[: len(ex)] = torch.arange(len(ex), device="cuda")
    report = {}
    for arm, second in [("replay", original), ("swapped", reversed_pairs)]:
        model = restore(PARENT).cuda()
        rng = np.random.default_rng(61841)
        _, arrays = prepare(
            original + second, np.arange(len(bank)), examples=np.arange(len(ex))
        )
        x, y, col, route, addr, n, m = [
            torch.as_tensor(a, device="cuda") for a in arrays
        ]
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=0.0003
        )
        sched = hashlib.sha256()
        losses = []
        steps = 0
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(12):
            order = rng.permutation(len(data))
            total = 0.0
            sched.update(order.astype("<i4").tobytes())
            for j in range(0, len(data), 48):
                ids = order[j : j + 48]
                idx = torch.as_tensor(np.r_[ids, ids + len(data)], device="cuda")
                a, b = int(n[idx].max()), int(m[idx].max())
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = emission_loss(
                        model,
                        features,
                        x[idx, :a].long(),
                        y[idx, :b].long(),
                        n[idx],
                        m[idx],
                        col[idx, :b].long(),
                        route[idx, : b + 1],
                        addr[idx, :b].long(),
                        lookup,
                        True,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                steps += 1
                total += loss.item() * len(ids)
            losses.append(total / len(data))
            with open("/proc/meminfo") as f:
                reserve = int(
                    next(v for v in f if v.startswith("MemAvailable:")).split()[1]
                )
            if reserve < 24 * 2**20 or torch.cuda.max_memory_allocated() > 10 * 2**30:
                raise RuntimeError("Memory ceiling")
        path = OUT / (arm + ".safetensors")
        if path.exists():
            raise FileExistsError(path)
        save_file(
            {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(path)
        )
        report[arm] = dict(
            steps=steps,
            seconds=time.monotonic() - start,
            loss=losses,
            sha256=sha(path),
            target_cells_per_pass=int(m.sum()),
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            schedule=sched.hexdigest(),
        )
    assert (
        report["replay"]["schedule"] == report["swapped"]["schedule"]
        and report["replay"]["target_cells_per_pass"]
        == report["swapped"]["target_cells_per_pass"]
    )
    write_new(
        OUT / "fit.json", dict(arms=report, parent=sha(PARENT), sources=len(data))
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
