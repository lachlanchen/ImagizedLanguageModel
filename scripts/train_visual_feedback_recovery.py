"""Fit equal-exposure continuations with clean history and frozen-parent image mistakes."""

import copy
import fcntl
import hashlib
import time
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator, END
from ilm.visual_lm.idiom_ink import visual_input
from scripts.diagnose_causal_glyph_history import predict
from scripts.train_causal_glyph_translator import (
    OUT as PARENT,
    ROOT,
    prepare,
    read_rows,
    sha,
    tiles,
    write_new,
)
from scripts.align_visual_translation import load, require

OUT = ROOT / "artifacts/visual_feedback_recovery_v1"
SEED = 42503


def run():
    OUT.mkdir()
    began = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    parent = PARENT / "translator.safetensors"
    require(
        sha(parent) == load(PARENT / "fit.json")["checkpoint_sha256"], "Changed parent"
    )
    weights = load_file(str(parent))
    frozen = CausalGlyphTranslator(weights["landmarks"], weights["exemplars"])
    frozen.load_state_dict(weights)
    frozen.cuda().eval()
    rows = read_rows("train")
    bank, canonical = np.unique(tiles(), axis=0, return_inverse=True)
    _, data = prepare(rows, canonical)
    indices = {bytes(v): i for i, v in enumerate(bank)}
    size = len(bank)
    original = torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool()
    gold = data[1]
    substituted = gold.copy()
    early = 0
    # Bucket only for padding. Each preceding image comes from the registered parent.
    for batch in np.array_split(np.argsort(data[-1]), (len(rows) + 95) // 96):
        nx, ny = data[-2][batch], data[-1][batch]
        pic = original[data[0][batch, : nx.max()]]
        target = original[gold[batch, : ny.max()]]
        prediction, mode = predict(
            frozen, pic, torch.as_tensor(nx, device="cuda"), target
        )
        packed = np.packbits(prediction, axis=-1)
        for i, j in enumerate(batch):
            for t in range(ny[i]):
                if mode[i, t] == END:
                    early += 1
                else:
                    substituted[j, t] = indices.setdefault(
                        bytes(packed[i, t]), len(indices)
                    )
    if len(indices) > size:
        bank = np.frombuffer(b"".join(indices.keys()), np.uint8).reshape(-1, 128)
    full = torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool()
    features = visual_input(full, frozen.landmarks)
    restore_hash = hashlib.sha256(substituted.tobytes()).hexdigest()
    wrong = int((substituted != gold).sum())
    np.savez_compressed(OUT / "parent_choices.npz", indices=substituted, bank=bank)
    x, y, col, route, addr, n, m = [torch.as_tensor(v, device="cuda") for v in data]
    substitute = torch.as_tensor(substituted, device="cuda")
    fits = {}
    for arm in ("clean", "recovery"):
        net = copy.deepcopy(frozen).train()
        opt = torch.optim.AdamW(
            (p for p in net.parameters() if p.requires_grad), lr=0.0005
        )
        rng = np.random.default_rng(SEED)
        noise = torch.Generator(device="cuda")
        noise.manual_seed(SEED + 1)
        losses = []
        changes = 0
        steps = 0
        start = time.monotonic()
        schedule = hashlib.sha256()
        for epoch in range(4):
            batches = []
            for bucket in np.array_split(rng.permutation(len(rows)), 16):
                bucket = bucket[np.argsort(data[-1][bucket], kind="stable")]
                batches.extend(bucket[i : i + 96] for i in range(0, len(bucket), 96))
            rng.shuffle(batches)
            total = 0.0
            for batch in batches:
                schedule.update(batch.astype("<i4").tobytes())
                idx = torch.as_tensor(batch, device="cuda")
                a, b = int(n[idx].max()), int(m[idx].max())
                labels = y[idx, :b].long()
                history = labels
                if arm == "recovery":
                    rival = substitute[idx, :b].long()
                    changed = (rival != labels) & (
                        torch.rand(labels.shape, generator=noise, device="cuda") < 1 / 3
                    )
                    changes += int(changed.sum())
                    history = torch.where(changed, rival, labels)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net.loss(
                        features[x[idx, :a].long()],
                        n[idx],
                        features[labels],
                        m[idx],
                        col[idx, :b].long(),
                        route[idx, : b + 1].long(),
                        addr[idx, :b].long(),
                        history_features=features[history],
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                total += loss.item() * len(batch)
                steps += 1
            losses.append(total / len(rows))
            print(arm, epoch + 1, losses[-1], changes, flush=True)
            require(torch.cuda.max_memory_allocated() < 10 * 2**30, "Memory budget")
            with open("/proc/meminfo") as f:
                require(
                    int(next(s for s in f if s.startswith("MemAvailable:")).split()[1])
                    >= 24 * 2**20,
                    "Host reserve",
                )
        path = OUT / (arm + ".safetensors")
        save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()},
            str(path),
            metadata={"format": "ilm-causal-glyph-v1"},
        )
        fits[arm] = dict(
            loss=losses,
            steps=steps,
            changes=changes,
            seconds=time.monotonic() - start,
            schedule=schedule.hexdigest(),
            sha256=sha(path),
        )
        del net, opt
    require(
        fits["clean"]["schedule"] == fits["recovery"]["schedule"], "Batch order changed"
    )
    write_new(
        OUT / "fit.json",
        dict(
            parent=sha(parent),
            choices_sha256=restore_hash,
            early_end=early,
            wrong_glyphs=wrong,
            additional_images=len(bank) - size,
            targets=int(data[-1].sum()),
            fits=fits,
            allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            seconds=time.monotonic() - began,
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
