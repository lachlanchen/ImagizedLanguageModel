"""Capture prespecified save points from the atomically replaced optimization state."""

import time
from pathlib import Path
import torch
from safetensors.torch import save_file
from scripts.prepare_marginal_confirmation import WORK, PUB
from scripts.prepare_open_exam import sha, read, dump
from scripts.train_region_writer import SEED
from ilm.visual_lm.marginal_phrase import MarginalPhraseWriter


def capture():
    folder = WORK / "trajectory"
    folder.mkdir(exist_ok=True)
    state = WORK / "resume.pt"
    receipt = PUB / "trajectory-checkpoints.json"
    records = read(receipt) if receipt.exists() else {}
    for step in (0, 2400, 4800, 7200):
        f = folder / (str(step) + ".safetensors")
        if str(step) in records:
            if records[str(step)] is not None:
                assert sha(f) == records[str(step)]
            continue
        if step == 0:
            torch.manual_seed(SEED)
            weights = MarginalPhraseWriter().state_dict()
        else:
            stamp = None
            while True:
                if not state.exists() or stamp == state.stat().st_ino:
                    time.sleep(0.4)
                    continue
                with state.open("rb") as handle:
                    import os

                    stamp = os.fstat(handle.fileno()).st_ino
                    z = torch.load(handle, map_location="cpu", weights_only=True)
                if z["step"] < step:
                    continue
                if z["step"] != step:
                    weights = None
                    print(f"Missed {step}, current {z['step']}", flush=True)
                    break
                weights = z["net"]
                break
        if weights is None:
            records[str(step)] = None
            dump(receipt, records)
            continue
        tmp = f.with_suffix(".tmp")
        save_file({k: v.contiguous() for k, v in weights.items()}, str(tmp))
        tmp.replace(f)
        records[str(step)] = sha(f)
        dump(receipt, records)
        print(step, records[str(step)], flush=True)


if __name__ == "__main__":
    capture()
