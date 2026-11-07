#!/usr/bin/env python3
"""Test native-length pooled context from full-resolution semantic windows."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import canonical, score_controls
from score_native_visual_alignment import edit_distance
from score_native_english_transfer import score as score_english

ZH = Path("artifacts/native_chinese_interface/prepared_v1")
EN = Path("artifacts/native_english_transfer/prepared_v2")


def read(path):
    return json.loads(Path(path).read_text())


def binding():
    return {name: sha(name) for name in ("scripts/probe_native_pooled_reader.py", "ilm/native_unified/pooled_windows.py", "ilm/native_unified/windowed.py",
        "ilm/native_unified/showo2_source.py", "ilm/native_unified/reading.py")}


def preflight(args, b):
    import torch
    from ilm.native_unified.showo2_source import image_latent, reading_prefix
    from ilm.native_unified.pooled_windows import pooled_windowed_prefix as windowed_prefix
    from ilm.native_unified.reading import cached_decisions
    from probe_native_unified_source import ANSWER_INSTRUCTION
    root = Path("artifacts/native_english_transfer/native432_prepared_v1")
    rows = [r for r in read(root / "inputs.json") if r["condition"] == "normal"][:4]
    prior = {r["id"]: r for r in read("artifacts/native_english_transfer/unadapted_hq_native432_v1/outputs.json")}
    checks = []
    for row in rows:
        path = root / row["input"]
        if sha(path) != row["input_sha256"]:
            raise ValueError("native equivalence image changed")
        z, _ = image_latent(b, path, row["seed"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, ma = reading_prefix(b, z, ANSWER_INSTRUCTION)
            c, mc, geometry = windowed_prefix(b, z, ANSWER_INSTRUCTION)
            error = float((a - c).abs().max())
            if error != 0 or not torch.equal(ma, mc) or geometry["semantic_windows"] != 1:
                raise ValueError("single native window changed prefix or mask")
            ids_a = cached_decisions(b, a, ma, seed=row["readout_seed"], cap=96)
            ids_c = cached_decisions(b, c, mc, seed=row["readout_seed"], cap=96)
        if ids_a != ids_c or ids_a != prior[row["id"]]["ids"]:
            raise ValueError("single native window changed source decisions")
        checks.append(dict(id=row["id"], input_sha256=row["input_sha256"], prefix_error=error,
            mask_bit_exact=True, native_source_ids=ids_a, window_ids=ids_c,
            retained_source_ids_exact=True, geometry=geometry))
        print("equivalent", row["id"], flush=True)
    save_json(args.output / "report.json", dict(cases=4, native_readout_calls=8, all_exact=True,
        checks=checks, implementation=binding(), source_parameters_changed=False,
        independent_image_answers=False, general_ilm_goal_complete=False))


def evaluate(args, b):
    import torch
    from ilm.native_unified.showo2_source import image_latent
    from ilm.native_unified.pooled_windows import pooled_windowed_prefix as windowed_prefix
    from ilm.native_unified.reading import cached_decisions
    from probe_native_unified_source import ANSWER_INSTRUCTION
    proof = read(args.preflight / "report.json")
    if read(args.preflight / "status.json")["state"] != "completed" or proof["implementation"] != binding() or not proof["all_exact"]:
        raise ValueError("unchanged implementation and terminal source equivalence required")
    outputs, costs = [], {}
    for language, root, index in (("zh", ZH, "source_1024_inputs.json"), ("en", EN, "inputs.json")):
        rows = read(root / index)
        if len(rows) != (48 if language == "zh" else 80):
            raise ValueError("complete original language cohort required")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for i, row in enumerate(rows):
            if any(k in row for k in ("expected", "answer", "references", "text")):
                raise ValueError("input-only raster index required")
            path = root / row["input"]
            if sha(path) != row["input_sha256"]:
                raise ValueError("original full-resolution pixels changed")
            with Image.open(path) as image:
                if image.size != (1024, 1024):
                    raise ValueError("original whole page required")
            z, _ = image_latent(b, path, row["seed"])
            codec = None
            if language == "zh" and row["condition"] == "normal":
                raw = b.vae.batch_decode(z[:, :, None]).squeeze(2)[0].float().permute(1, 2, 0).cpu().numpy()
                target = args.output / f"codec/{i:02d}.png"
                target.parent.mkdir(exist_ok=True)
                Image.fromarray(np.rint((raw + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(target)
                codec = dict(image=str(target.relative_to(args.output)), sha256=sha(target))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask, geometry = windowed_prefix(b, z, ANSWER_INSTRUCTION)
                if geometry["grid"] != [64, 64] or geometry["semantic_windows"] != 9 or prefix.shape[1] != 768:
                    raise ValueError("whole-image windows/context differ from protocol")
                ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
            result = {key: row[key] for key in ("id", "pair", "condition", "input_sha256", "seed", "readout_seed")}
            result.update(language=language, ids=ids, text=b.tokenizer.decode(ids, skip_special_tokens=True),
                stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id), geometry=geometry,
                prefix_positions=prefix.shape[1], codec=codec, student_answer=False)
            outputs.append(result)
            save_json(args.output / "outputs.json", outputs)
            save_json(args.output / "status.json", dict(state="running", pid=os.getpid(), complete_calls=len(outputs), planned_calls=128))
            print(language, row["id"], repr(result["text"]), flush=True)
        costs[language] = dict(cases=len(rows), inputs_sha256=sha(root / index), elapsed_seconds=time.monotonic() - started,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3, includes_codec_reconstruction=language == "zh")
    save_json(args.output / "report.json", dict(cases=128, costs=costs, implementation=binding(),
        preflight_sha256=sha(args.preflight / "report.json"), source_counts=b.counts,
        source_model_sha256=sha(b.files.model), body_dtype="bfloat16", codec_dtype="float32", cap=96,
        optimizer_updates=0, parameters_added=0, source_pixels_resized=False, source_context_resampled=True,
        auxiliary_unicode_head=True, reference_answers_loaded=False,
        image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))


def score(args):
    report = read(args.run / "report.json")
    if read(args.run / "status.json")["state"] != "completed" or report["cases"] != 128 or report["implementation"] != binding():
        raise ValueError("complete source-bound observation required")
    if report["preflight_sha256"] != sha(args.preflight / "report.json"):
        raise ValueError("native equivalence report changed")
    outputs = read(args.run / "outputs.json")
    models = {}
    files = {}

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != sha(target):
            raise ValueError("export changed")
        files[name] = sha(target)

    for language, root, index, scorer in (("zh", ZH, "source_1024_inputs.json", score_controls), ("en", EN, "inputs.json", score_english)):
        if report["costs"][language]["inputs_sha256"] != sha(root / index):
            raise ValueError("original input index changed")
        models[language] = scorer(read(root / index), read(root / "references.json"), [r for r in outputs if r["language"] == language])
        retain(root / index, language + "_inputs.json")
        retain(root / "references.json", language + "_references.json")
    visible = {r["id"]: r["text"] for r in read(ZH / "source_text_inputs.json") if r["condition"] == "normal"}
    recognition = []
    for row in outputs:
        if row["language"] != "zh" or row["condition"] != "normal":
            continue
        path = args.run / row["codec"]["image"]
        if sha(path) != row["codec"]["sha256"]:
            raise ValueError("full-resolution codec reconstruction changed")
        retain(path, row["codec"]["image"])
        result = subprocess.run(["tesseract", str(path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
            env=dict(os.environ, OMP_THREAD_LIMIT="1"), capture_output=True, text=True, check=True)
        target, actual = canonical(visible[row["id"]], "zh"), canonical(result.stdout, "zh")
        recognition.append(dict(id=row["id"], text=result.stdout, expected=visible[row["id"]],
            source_sha256=sha(path), canonical_exact=actual == target,
            edit_errors=edit_distance(actual, target), reference_characters=len(target)))
        save_json(args.output / "recognition.json", recognition)
    if len(recognition) != 32:
        raise ValueError("complete Chinese codec observation required")
    files["recognition.json"] = sha(args.output / "recognition.json")
    for name in ("outputs.json", "report.json"):
        retain(args.run / name, "reports/" + name)
    retain(args.preflight / "report.json", "reports/native_equivalence.json")
    for name in ("scripts/probe_native_pooled_reader.py", "ilm/native_unified/pooled_windows.py", "ilm/native_unified/windowed.py"):
        retain(Path(name), "implementation/" + Path(name).name)
    save_json(args.output / "result.json", dict(models=models, files_sha256=files,
        codec_readability=dict(cases=32, exact=sum(r["canonical_exact"] for r in recognition),
            character_edit_error=sum(r["edit_errors"] for r in recognition) / sum(r["reference_characters"] for r in recognition)),
        original_evidence_sha256={name: sha("references/evidence/" + name + "/result.json") for name in ("native_chinese_interface_result", "native_reader_geometry_result", "native_window_reader_result")},
        source_report=report, auxiliary_unicode_head=True, pixels_resized=False,
        independent_ocr_conditioning=False, image_answer_generation_demonstrated=False, general_ilm_goal_complete=False,
        copyright="CMRC 2018 and SQuAD 1.1: CC-BY-SA-4.0, Cui et al. 2019 and Rajpurkar et al. 2016. Native source/model: NUS Show Lab, Apache-2.0."))
    print(json.dumps(dict(zh={k: v for k, v in models['zh'].items() if k != 'pairs'}, en=models['en']['summary']), ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "evaluate", "score"), required=True)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--preflight", type=Path, default=Path("artifacts/native_pooled_reader/preflight_v1"))
    p.add_argument("--run", type=Path, default=Path("artifacts/native_pooled_reader/source_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid()))
    try:
        if args.stage == "score":
            score(args)
        else:
            import torch
            from ilm.native_unified.showo2_source import load
            from probe_native_unified_source import gpu_slot
            torch.set_num_threads(8)
            with gpu_slot(), torch.inference_mode():
                b = load(args.assets, dtype=torch.bfloat16)
                b.vae = None
                torch.cuda.empty_cache()
                b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
                {"preflight": preflight, "evaluate": evaluate}[args.stage](args, b)
        save_json(args.output / "status.json", dict(state="completed", stage=args.stage, pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", stage=args.stage, pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
