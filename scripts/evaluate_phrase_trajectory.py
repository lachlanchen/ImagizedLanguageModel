"""Evaluate registered optimization snapshots on the older complete confirmation cohort."""

import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.phrase_region import PhraseRegionWriter
from scripts.prepare_marginal_confirmation import WORK, PUB
from scripts.prepare_phrase_confirmation import WORK as FIRST, PUB as FIRST_PUB
from scripts.prepare_region_writer import MODEL as BASE, Ink
from scripts.train_marginal_phrase import MODEL
from scripts.prepare_open_exam import read, dump, sha
from scripts.evaluate_phrase_region import group_counts, measurements
from scripts.evaluate_region_writer import run_group, e2e_scorer
from scripts.audit_region_writer import wording
from scripts.write_region_writer import model
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import unpack


def main():
    rows = read(FIRST / "confirmation.json")
    assert (
        sha(FIRST / "confirmation.json")
        == read(FIRST_PUB / "confirmation.json")["sha256"]
    )
    checkpoint = read(PUB / "trajectory-checkpoints.json")
    report = {}
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        sel = model("selector", "cuda")
        palette = np.load(BASE / "palette.npy")
        book = read(BASE / "readback.json")
        score = e2e_scorer()
        for step in (0, 2400, 4800, 7200, 9912):
            if step == 9912:
                raw = unpack(WORK / "previous-marginal.json.gz")["original"]
                signature = read(MODEL / "fit.json")["weights"]
            else:
                signature = checkpoint[str(step)]
                if signature is None:
                    report[step] = None
                    continue
                path = WORK / "trajectory" / (str(step) + ".safetensors")
                if sha(path) != signature:
                    raise ValueError("Snapshot differs")
                net = PhraseRegionWriter().cuda()
                net.load_state_dict(load_file(str(path)))
                net.eval()
                raw = run_group(
                    net, sel, "selected", rows, Ink(palette), book, score, wording
                )
            m = measurements(rows, raw, Ink(palette), score)
            report[step] = dict(scores=group_counts(rows, m, [], []), weights=signature)
            print(
                step,
                {
                    k: (v["bitmap_exact"], v["fact"])
                    for k, v in report[step]["scores"].items()
                },
                flush=True,
            )
    dump(PUB / "trajectory.json", report)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
