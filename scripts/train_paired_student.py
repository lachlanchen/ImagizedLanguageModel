"""Compare supplemental visual teacher targets with an equal-schedule archive replay."""

import fcntl
import hashlib
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator
from ilm.visual_lm.idiom_ink import visual_input
from scripts.train_causal_glyph_translator import (
    OUT as INITIAL,
    ROOT,
    prepare,
    sha,
    write_new,
)
from scripts.align_visual_translation import load, require

OUT = ROOT / "artifacts/paired_gloss_student_v1"
DATA = ROOT / "artifacts/paired_teacher_glosses_v1"
PARENT = INITIAL / "translator.safetensors"


def restore(path):
    state = load_file(str(path))
    model = CausalGlyphTranslator(state["landmarks"], state["exemplars"])
    model.load_state_dict(state)
    return model


def inputs(model):
    record = load(ROOT / "references/evidence/paired_teacher_glosses_20261119.json")
    require(
        record["quality"]["supplemental_candidate"]
        and sha(DATA / "images.npz") == record["preparation"]["image_sha256"],
        "Unqualified preparation",
    )
    with np.load(DATA / "images.npz") as f:
        exemplars = np.packbits(model.exemplars.numpy(), axis=1)
        bank = np.unique(
            np.concatenate(
                [exemplars, *[f[k] for k in ["source", "archive", "teacher"]]]
            ),
            axis=0,
        )
        index = {bytes(c): i for i, c in enumerate(bank)}
        examples = [index[bytes(c)] for c in exemplars]
        arrays = {}
        groups = np.flatnonzero(f["usable"])
        for k in ["source", "archive", "teacher"]:
            stream = f[k]
            offsets = f[k + "_offsets"]
            arrays[k] = [
                [index[bytes(c)] for c in stream[offsets[j] : offsets[j + 1]]]
                for j in groups
            ]
    count = len(groups)
    arms = {}
    for name in ["archive", "teacher"]:
        pairs = [
            dict(x=x, z=y)
            for x, y in zip(arrays["source"] * 2, arrays["archive"] + arrays[name])
        ]
        _, values = prepare(pairs, np.arange(len(bank)), examples=examples)
        arms[name] = [torch.as_tensor(v, device="cuda") for v in values]
    return bank, arms, count


def fit():
    OUT.mkdir()
    require(
        sha(PARENT) == load(INITIAL / "fit.json")["checkpoint_sha256"],
        "Changed starting weights",
    )
    torch.set_num_threads(2)
    torch.manual_seed(49924)
    torch.set_float32_matmul_precision("high")
    bank, arms, count = inputs(restore(PARENT))
    stats = {}
    for arm, (x, y, col, route, addr, n, m) in arms.items():
        model = restore(PARENT).cuda()
        feat = visual_input(
            torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool(), model.landmarks
        )
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.0003
        )
        rng = np.random.default_rng(49924)
        schedule = hashlib.sha256()
        losses = []
        steps = 0
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(12):
            order = rng.permutation(count)
            total = 0.0
            schedule.update(order.astype("<i4").tobytes())
            for j in range(0, count, 32):
                items = order[j : j + 32]
                ids = torch.as_tensor(np.r_[items, items + count], device="cuda")
                a, b = int(n[ids].max()), int(m[ids].max())
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.loss(
                        feat[x[ids, :a].long()],
                        n[ids],
                        feat[y[ids, :b].long()],
                        m[ids],
                        col[ids, :b].long(),
                        route[ids, : b + 1].long(),
                        addr[ids, :b].long(),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                total += loss.item() * len(items)
                steps += 1
            losses.append(total / count)
            require(torch.cuda.max_memory_allocated() < 10 * 2**30, "GPU reserve")
            with open("/proc/meminfo") as f:
                require(
                    int(next(l for l in f if l.startswith("MemAvailable:")).split()[1])
                    > 24 * 2**20,
                    "Host reserve",
                )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        ck = OUT / (arm + ".safetensors")
        save_file(
            {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(ck)
        )
        stats[arm] = dict(
            loss=losses,
            steps=steps,
            target_cells_per_pass=int(m.sum()),
            seconds=elapsed,
            allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            schedule=schedule.hexdigest(),
            checkpoint=sha(ck),
        )
        del opt, model, feat, loss
    require(
        stats["archive"]["schedule"] == stats["teacher"]["schedule"],
        "Source schedules diverged",
    )
    write_new(
        OUT / "fit.json",
        dict(
            arms=stats,
            parent=sha(PARENT),
            usable=count,
            passes=12,
            data=sha(DATA / "images.npz"),
            model_parameters=load(INITIAL / "fit.json")["parameters"],
            trainable=load(INITIAL / "fit.json")["trainable"],
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit()
