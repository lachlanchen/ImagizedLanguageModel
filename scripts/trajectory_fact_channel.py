"""Trace autonomous writing on fixed bilingual development pages using saved weights."""

from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from scripts.prepare_fact_channel import WORK, PUB, MODEL, Ink, read, dump, sha
from scripts.prepare_marginal_confirmation import WORK as DEV
from scripts.train_region_writer import SEED, gpu_owner
from scripts.write_fact_channel import VisualAnswer
from scripts.evaluate_fact_channel import observed
from scripts.audit_region_writer import recount
from scripts.verse_gpu_runtime import configure_cuda_cli
from ilm.visual_lm.region_writer import RegionWriter


def main():
    previous = read(DEV / "confirmation.json")
    rows = [
        r
        for t in ("statements", "exam")
        for r in [x for x in previous if x["task"] == t][:64]
    ]
    results = {}
    with gpu_owner(), torch.inference_mode():
        for arm in ("direct", "human", "predicted"):
            actor = VisualAnswer(arm, "cuda")
            end = read(MODEL / arm / "fit.json")["steps"]
            weights = {
                step: WORK / arm / f"step-{step}.safetensors"
                for step in range(2500, end, 2500)
            }
            snaps = {0: None, **weights, end: MODEL / arm / "weights.safetensors"}
            entries = {}
            for step, f in snaps.items():
                if f is None:
                    torch.manual_seed(SEED)
                    actor.writer.load_state_dict(RegionWriter().state_dict())
                else:
                    actor.writer.load_state_dict(load_file(str(f)))
                ink = Ink(actor.palette)
                z = observed(actor, rows, ink)
                totals = defaultdict(lambda: defaultdict(int))
                for r, v in zip(rows, z, strict=True):
                    task = r["task"]
                    s = recount(
                        {**r, "task": "catalogue"} if task == "exam" else r,
                        v,
                        ink,
                        None,
                    )
                    for k in ("bitmap_exact", "fact", "stopped"):
                        totals[task][k] += s[k]
                    totals[task]["n"] += 1
                entries[step] = dict(
                    sha256=sha(f) if f is not None else None, totals=totals
                )
            results[arm] = entries
            dump(PUB / "trajectory.json", results)
            print(
                arm,
                [
                    (k, v["totals"]["statements"]["bitmap_exact"])
                    for k, v in entries.items()
                ],
                flush=True,
            )


if __name__ == "__main__":
    configure_cuda_cli()
    main()
