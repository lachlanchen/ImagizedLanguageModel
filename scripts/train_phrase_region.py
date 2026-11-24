"""Fit the registered variable-length writer on the original paired optical schedule."""

import argparse
import math
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.phrase_region import PhraseRegionWriter
from scripts.train_region_writer import Corpus, batches, pack, gpu_owner, SEED
from scripts.prepare_region_writer import (
    MODEL as PARENT,
    WORK as DATA,
    PUB as RECORD,
    ROOT,
)
from scripts.prepare_phrase_confirmation import WORK, PUB
from scripts.prepare_open_exam import read, dump, sha
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli

MODEL = ROOT / "models/phrase_region"


def fit(model_dir=MODEL, run_dir=WORK):
    with gpu_owner():
        torch.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")
        corpus = Corpus("train", "cuda")
        order = batches(corpus, "selected")
        prior = read(PARENT / "selected/fit.json")
        routes = DATA / "selector/train-regions.npy"
        selector = PARENT / "selector/weights.safetensors"
        schedule_hash = sha256(
            np.asarray([r for b in order for r in b], dtype="<i4").tobytes()
        ).hexdigest()
        if (
            schedule_hash != prior["binding"]["order"]
            or sha(selector) != prior["binding"]["selector"]
            or sha(routes) != read(PARENT / "selector/fit.json")["train_region_hash"]
            or sha(RECORD / "corpus.json") != prior["binding"]["corpus"]
        ):
            raise ValueError("Unmatched training inputs")
        binding = dict(
            order=schedule_hash,
            selector=sha(selector),
            regions=sha(routes),
            corpus=sha(RECORD / "corpus.json"),
            confirmation=sha(PUB / "confirmation.json"),
            source={
                f: sha(ROOT / f)
                for f in (
                    "ilm/visual_lm/region_writer.py",
                    "ilm/visual_lm/phrase_region.py",
                    "scripts/train_region_writer.py",
                    "scripts/train_phrase_region.py",
                    "references/phrase_region_protocol.md",
                    "formal/span_list/OpticalPhrasePath.lean",
                )
            },
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        weights = model_dir / "weights.safetensors"
        receipt = model_dir / "fit.json"
        if receipt.exists():
            if read(receipt)["binding"] != binding or read(receipt)["weights"] != sha(
                weights
            ):
                raise ValueError("Completed fit differs")
            return
        picks = np.load(routes)
        net = PhraseRegionWriter().cuda()
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.001, weight_decay=0.01, fused=True
        )
        resume = run_dir / "resume.pt"
        start, seconds, energy, peak = 0, 0.0, 0.0, 0
        trace = []
        if resume.is_file():
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            if saved["binding"] != binding:
                raise ValueError("Incompatible resume")
            net.load_state_dict(saved["net"])
            opt.load_state_dict(saved["opt"])
            start, seconds, energy, peak, trace = [
                saved[k] for k in ("step", "seconds", "joules", "peak", "trace")
            ]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        clock, meter = monotonic(), snapshot()
        elapsed, joules = seconds, energy
        for j in range(start, len(order)):
            step = j + 1
            records = order[j]
            qs, ps, ys = [], [], []
            for i, y in records:
                r = corpus.rows[i]
                qs.append(corpus.seq[r[0]])
                ps.append(corpus.seq[r[1][picks[i]]])
                ys.append(np.r_[corpus.seq[y], np.int32(1)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx, h = net.encode(
                    corpus.features, corpus.n, pack(qs, "cuda"), pack(ps, "cuda")
                )
                loss = net.loss(ctx, h, pack(ys, "cuda")).mean()
            opt.param_groups[0]["lr"] = (
                0.001
                * min(1, step / 120)
                * (0.18 + 0.82 * (1 + math.cos(math.pi * step / len(order))) / 2)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            if step == 1 or step % 250 == 0 or step == len(order):
                trace.append(
                    [step, float(loss.detach()), seconds + monotonic() - clock]
                )
                print(trace[-1], flush=True)
            if step % 400 == 0 or step == len(order):
                torch.cuda.synchronize()
                elapsed = seconds + monotonic() - clock
                joules = energy + consumed(meter, snapshot())["joules"]
                peak = max(peak, torch.cuda.max_memory_allocated())
                tmp = resume.with_suffix(".tmp")
                torch.save(
                    dict(
                        binding=binding,
                        step=step,
                        net=net.state_dict(),
                        opt=opt.state_dict(),
                        seconds=elapsed,
                        joules=joules,
                        peak=peak,
                        trace=trace,
                    ),
                    tmp,
                )
                tmp.replace(resume)
                if elapsed > 10800 or peak > 18 * 2**30:
                    raise RuntimeError("Resource ceiling exceeded")
        tmp = weights.with_suffix(".tmp")
        save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(tmp)
        )
        tmp.replace(weights)
        dump(
            receipt,
            dict(
                binding=binding,
                weights=sha(weights),
                steps=len(order),
                presentations=sum(map(len, order)),
                parameters=sum(p.numel() for p in net.parameters()),
                seconds=elapsed,
                joules=joules,
                peak_bytes=peak,
                trace=trace,
            ),
        )


if __name__ == "__main__":
    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--run-dir", type=Path, default=WORK)
    a = p.parse_args()
    fit(a.model_dir, a.run_dir)
