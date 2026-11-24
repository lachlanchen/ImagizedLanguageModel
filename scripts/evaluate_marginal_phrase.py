"""Compare marginal learning with both registered writers on the second confirmation set."""

from pathlib import Path
import numpy as np
import torch
from scripts.train_marginal_phrase import MODEL
from scripts.train_phrase_region import MODEL as LONGEST
from scripts.prepare_marginal_confirmation import WORK, PUB
from scripts.prepare_phrase_confirmation import WORK as FIRST, PUB as EARLIER
from scripts.prepare_region_writer import MODEL as BASE, Ink
from scripts.prepare_open_exam import read, dump, sha
from scripts.evaluate_region_writer import changed, run_group, e2e_scorer, offpage
from scripts.evaluate_phrase_region import measurements, group_counts, intervals
from scripts.write_region_writer import model, phrase_model
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.train_paired_evidence import gpu_owner
from scripts.probe_region_writer_transfer import transfer


def main():
    score = e2e_scorer()
    ext, dropped, norm = offpage()
    results = {}
    for suite, folder, receipt in (("new", WORK, PUB), ("previous", FIRST, EARLIER)):
        file = folder / "confirmation.json"
        if sha(file) != read(receipt / "confirmation.json")["sha256"]:
            raise ValueError("Changed confirmation")
        rows = read(file)
        edits = changed(rows)
        scores = {}
        metrics = {}
        with gpu_owner(), torch.inference_mode():
            torch.set_float32_matmul_precision("highest")
            sel = model("selector", "cuda")
            for arm in ("single", "longest", "marginal"):
                ink = Ink(np.load(BASE / "palette.npy"))
                chars = read(BASE / "readback.json")
                if suite == "previous" and arm != "marginal":
                    old = "single" if arm == "single" else "phrase"
                    z = unpack(FIRST / ("confirmation-" + old + ".json.gz"))
                else:
                    net = (
                        model("selected", "cuda")
                        if arm == "single"
                        else phrase_model(
                            LONGEST if arm == "longest" else MODEL, "cuda"
                        )
                    )
                    z = {
                        k: run_group(net, sel, "selected", rs, ink, chars, score, norm)
                        for k, rs in (("original", rows), ("edited", edits))
                    }
                    if suite == "new" and arm == "marginal":
                        z["coqa"] = run_group(
                            net, sel, "selected", ext, ink, chars, score, norm
                        )
                packed(WORK / (suite + "-" + arm + ".json.gz"), z)
                m = measurements(rows, z["original"], ink, score)
                e = measurements(edits, z["edited"], ink, score)
                metrics[arm] = m
                scores[arm] = group_counts(rows, m, edits, e)
                if "coqa" in z:
                    scores[arm]["coqa"] = dict(
                        tested=len(ext),
                        outside_geometry=dropped,
                        exact=sum(r["exact"] for r in z["coqa"]),
                    )
                print(suite, arm, scores[arm], flush=True)
        results[suite] = dict(
            scores=scores,
            paired={
                base: intervals(
                    rows, {"phrase": metrics["marginal"], "single": metrics[base]}
                )
                for base in ("single", "longest")
            },
        )
        dump(PUB / "answers.json", results)
    dump(
        PUB / "transfer.json",
        transfer([("marginal", "selected", lambda: phrase_model(MODEL, "cuda"))]),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
