"""Train and resume the bilingual optical list reader on a single GPU."""

from argparse import ArgumentParser
from math import cos, pi
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import RunReader
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_multiple_ink_runs import ROOT, WORK, PUB, MODEL, read, dump, sha
from scripts.train_paired_evidence import gpu_owner
from scripts.train_visual_cursor import pad
from scripts.write_paired_evidence import reader

STEPS = 6000
BATCH = 12
SEED = 317091


def load(device="cpu"):
    path = MODEL / "reader.safetensors"
    if sha(path) != read(PUB / "fit.json")["weights"]:
        raise ValueError("Changed released weights")
    model = RunReader().to(device)
    model.load_state_dict(load_file(str(path), device=device))
    return model.eval()


def fit(run=None):
    spec = read(PUB / "corpus.json")
    path = WORK / "training.npz"
    if sha(path) != spec["files"][path.name]:
        raise ValueError("Training corpus changed")
    dest = run or MODEL
    dest.mkdir(parents=True, exist_ok=True)
    receipt = (run or PUB) / "fit.json"
    weights = dest / "reader.safetensors"
    if receipt.exists():
        if sha(weights) != read(receipt)["weights"]:
            raise ValueError("Weight checksum mismatch")
        return
    with np.load(path) as f:
        arr = {k: f[k] for k in f.files}
    splits = [np.flatnonzero(arr["lang"] == i) for i in range(2)]
    corpus = sha(PUB / "corpus.json")
    checkpoint = (run or WORK) / "resume.pt"
    bindings = dict(
        corpus=corpus,
        protocol=sha(ROOT / "references/multiple_ink_runs_protocol.md"),
        trainer=sha(Path(__file__)),
        reader=sha(ROOT / "ilm/visual_lm/multiple_ink_runs.py"),
        base=sha(ROOT / "models/paired_evidence/evidence.safetensors"),
    )
    with gpu_owner():
        torch.manual_seed(SEED)
        model = RunReader(reader("evidence", "cuda")).cuda().train()
        model.ink.requires_grad_(False)
        bank = model.ink(torch.from_numpy(observations(arr, "ink")).cuda()).detach()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.00045,
            weight_decay=0.02,
            fused=True,
        )
        first = 1
        trace = []
        earlier_s = earlier_j = 0.0
        if checkpoint.exists():
            st = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if st["bindings"] != bindings:
                raise ValueError("Resume binding mismatch")
            model.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            torch.set_rng_state(st["rng"])
            torch.cuda.set_rng_state(st["cuda"])
            first = st["step"] + 1
            trace = st["trace"]
            earlier_s = st["seconds"]
            earlier_j = st["joules"]
        start = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        for step in range(first, STEPS + 1):
            rng = np.random.default_rng([SEED, step])
            ix = np.concatenate([rng.choice(pool, BATCH // 2) for pool in splits])
            ps = [arr["p"][arr["po"][i] : arr["po"][i + 1]] for i in ix]
            qs = [arr["q"][arr["qo"][i] : arr["qo"][i + 1]] for i in ix]
            ys = [arr["y"][arr["po"][i] : arr["po"][i + 1]] for i in ix]
            q, s, y = [
                torch.as_tensor(pad(v), device="cuda").long() for v in (qs, ps, ys)
            ]
            opt.param_groups[0]["lr"] = (
                0.00045
                * min(1, step / 120)
                * (0.15 + 0.85 * (1 + cos(pi * step / STEPS)) / 2)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                role, count = model(bank, q, s)
                count_loss = F.cross_entropy(count.float(), (y == 1).sum(-1))
                keep = s >= 2
                # Balanced emphasis on rare positive cells, with matching log offsets
                # to keep the deployed logits on the underlying role scale.
                w = role.new_tensor([1.0, 5.0, 2.0])
                token = F.cross_entropy(
                    (role + w.log()).transpose(1, 2), y, reduction="none"
                )
                mass = keep * w[y]
                tag_loss = ((token * mass).sum(1) / mass.sum(1)).mean()
                loss = tag_loss + 0.18 * count_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            if step == 1 or step % 250 == 0:
                trace.append(
                    dict(
                        step=step,
                        role=float(tag_loss.detach()),
                        count=float(count_loss.detach()),
                    )
                )
                print(trace[-1], flush=True)
            if step % 1000 == 0:
                torch.cuda.synchronize()
                elapsed = earlier_s + monotonic() - start
                if elapsed > 3600 or torch.cuda.max_memory_allocated() > 16 * 2**30:
                    raise RuntimeError("Fit ceiling")
                tmp = checkpoint.with_suffix(".tmp")
                torch.save(
                    dict(
                        step=step,
                        net=model.state_dict(),
                        opt=opt.state_dict(),
                        trace=trace,
                        bindings=bindings,
                        seconds=elapsed,
                        joules=earlier_j + consumed(meter, snapshot())["joules"],
                        rng=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state(),
                    ),
                    tmp,
                )
                tmp.replace(checkpoint)
        torch.cuda.synchronize()
        secs = earlier_s + monotonic() - start
        energy = earlier_j + consumed(meter, snapshot())["joules"]
        tmp = weights.with_suffix(".tmp")
        save_file(model.cpu().state_dict(), str(tmp))
        tmp.replace(weights)
        dump(
            receipt,
            dict(
                weights=sha(weights),
                bindings=bindings,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                steps=STEPS,
                presentations=STEPS * BATCH,
                peak_bytes=torch.cuda.max_memory_allocated(),
                seconds=secs,
                joules=energy,
                trace=trace,
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path)
    fit(ap.parse_args().run_dir)
