#!/usr/bin/env python3
"""Complete the fixed fresh-source zero-correction equivalence observation."""
import argparse
import json
import os
from pathlib import Path
import traceback

import torch
from ilm.image_edit.baseline import save_json, sha
from ilm.native_unified.showo2_source import load, image_latent, reading_prefix
from ilm.native_unified.reading import add_reading_adapters, reader_gate, cached_decisions
from ilm.native_unified.state_alignment import add_bridge, corrected_prefix
from train_native_state_alignment import binding, SEED
from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--inputs", type=Path, default=Path("artifacts/native_state_alignment/assessment_v1/en_inputs.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", pid=os.getpid()))
    try:
        torch.set_num_threads(8)
        rows = [r for r in json.loads(args.inputs.read_text()) if r["condition"] == "normal"][:4]
        original = {r["id"]: r for r in json.loads(Path("artifacts/native_english_transfer/unadapted_hq_native432_v1/outputs.json").read_text())}
        with gpu_slot(), torch.inference_mode():
            b = load(args.assets, dtype=torch.bfloat16)
            b.vae = None
            torch.cuda.empty_cache()
            b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            add_reading_adapters(b, 16)
            add_bridge(b)
            b.model.eval()
            checks = []
            for row in rows:
                high, low = [args.inputs.parent / row[key] for key in ("input", "coarse_input")]
                if sha(high) != row["input_sha256"] or sha(low) != row["coarse_input_sha256"] or original[row["id"]]["input_sha256"] != sha(low):
                    raise ValueError("fixed source replay views changed")
                fine, _ = image_latent(b, high, row["seed"])
                coarse, _ = image_latent(b, low, row["seed"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix, mask = reading_prefix(b, coarse, ANSWER_INSTRUCTION)
                    with reader_gate(b.model):
                        corrected, other_mask = corrected_prefix(b, fine, prefix, mask)
                    if not torch.equal(prefix.view(torch.uint8), corrected.view(torch.uint8)) or not torch.equal(mask, other_mask):
                        raise ValueError("fresh zero correction changed source prefix/mask")
                    a = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
                    c = cached_decisions(b, corrected, other_mask, seed=row["readout_seed"], cap=96)
                if a != c or a != original[row["id"]]["ids"]:
                    raise ValueError("fresh zero correction changed native source decisions")
                checks.append(dict(id=row["id"], source_prefix_bit_exact=True, source_mask_bit_exact=True,
                    ids=a, corrected_ids=c, retained_source_replay_exact=True))
                print("zero correction exact", row["id"], flush=True)
            save_json(args.output / "report.json", dict(cases=4, readout_calls=8, all_exact=True, checks=checks,
                implementation=binding(), evaluator_sha256=sha(__file__), inputs_sha256=sha(args.inputs),
                optimizer_updates=0, fresh_source_initialization=True, learned_checkpoint_loaded=False,
                timing="Additional source control after main optimization started; original preflight scope deviation retained.",
                image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))
        save_json(args.output / "status.json", dict(state="completed", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
