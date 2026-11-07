#!/usr/bin/env python3
"""Calibrate frozen Chinese reading on original pixels, native scale and text."""
import argparse
import hashlib
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

INDEX_SHA = "4994faea12d72f6522630ec370fa9355b23ccceb84060dc43829f2437d6e77e6"
REF_SHA = "a498fb556746e4a8c90e9d1995f2249e578d612424efe3bf92edc75cabffed3f"
SOURCE_SHA = "742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c"
ARMS = ("source_1024", "source_432", "source_text")


def read(path):
    return json.loads(Path(path).read_text())


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(args):
    from ilm.image_edit.paired_data import qa_candidates
    original = Path("artifacts/native_visual_access/question_controls_v1")
    source = Path("data/visual_qa/cmrc2018_short/train.jsonl")
    manifest = Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl")
    if sha(original / "inputs.json") != INDEX_SHA or sha(original / "references.json") != REF_SHA or sha(source) != SOURCE_SHA:
        raise ValueError("original Chinese cohort or source changed")
    prior = read(original / "preparation.json")
    if sha(manifest) != prior["manifest_sha256"]:
        raise ValueError("original page manifest changed")
    pages = {r["id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    prompts = {r["id"]: r["prompt"] for pair in qa_candidates(source) for r in pair}
    rows = read(original / "inputs.json")
    if len(rows) != 48 or sum(r["condition"] == "normal" for r in rows) != 32:
        raise ValueError("complete Chinese intervention cohort required")
    inputs = {key: [] for key in ARMS}
    checks = []
    files = {}
    last_text = None
    for i, original_row in enumerate(rows):
        path = original / original_row["input"]
        if sha(path) != original_row["input_sha256"]:
            raise ValueError("original source pixels changed")
        row = dict(original_row, input=os.path.relpath(path.resolve(), args.output.resolve()))
        inputs["source_1024"].append(row)
        image_path = args.output / f"inputs/{i:02d}.png"
        image_path.parent.mkdir(exist_ok=True)
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise ValueError("full original square required")
            image.convert("RGB").resize((432, 432), Image.Resampling.BICUBIC).save(image_path)
        files[str(image_path.relative_to(args.output))] = sha(image_path)
        inputs["source_432"].append(dict(original_row, input=str(image_path.relative_to(args.output)),
            original_input_sha256=original_row["input_sha256"], input_sha256=sha(image_path)))
        if row["condition"] == "normal":
            text = prompts[row["id"]]
            if text_sha(text) != pages[row["id"]]["input_geometry"]["text_sha256"]:
                raise ValueError("original visible transcript differs")
            last_text = text
        else:
            if last_text is None or not row["id"].endswith(row["pair"]):
                raise ValueError("erased page order differs")
            text, separator, _ = last_text.rpartition("\n问题：")
            if not separator:
                raise ValueError("explicit question paragraph missing")
        inputs["source_text"].append(dict(row, text=text, input_sha256=text_sha(text),
            original_input_sha256=original_row["input_sha256"]))
        checks.append(dict(id=row["id"], source_image_sha256=original_row["input_sha256"], source_text_sha256=text_sha(text)))
    for arm in ARMS:
        save_json(args.output / (arm + "_inputs.json"), inputs[arm])
    shutil.copy2(original / "references.json", args.output / "references.json")
    save_json(args.output / "preparation.json", dict(cases_per_arm=48, arms=list(ARMS),
        original_inputs_sha256=INDEX_SHA, references_sha256=REF_SHA, source_sha256=SOURCE_SHA,
        inputs_sha256={arm: sha(args.output / (arm + "_inputs.json")) for arm in ARMS},
        files_sha256=files, checks=checks, implementation_sha256=sha(__file__),
        same_latent_and_readout_seeds=True, content_cropped=False, optimizer_updates=0,
        source_text_control_is_privileged=True, image_answer_generation_demonstrated=False,
        general_ilm_goal_complete=False))
    print("Prepared all 48 conditions in three source arms; no model calls.")


def authenticated_preparation(args):
    p = read(args.prepared / "preparation.json")
    if sha(args.prepared / "references.json") != REF_SHA:
        raise ValueError("fixed Chinese references changed")
    for arm, digest in p["inputs_sha256"].items():
        if sha(args.prepared / (arm + "_inputs.json")) != digest:
            raise ValueError("prepared inputs changed")
    for name, digest in p["files_sha256"].items():
        if sha(args.prepared / name) != digest:
            raise ValueError("prepared resized page changed")
    return p


def evaluate(args):
    import torch
    from ilm.native_unified.showo2_source import load, image_latent, reading_prefix, MODEL_SHA256
    from ilm.native_unified.reading import cached_decisions
    from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION
    p = authenticated_preparation(args)
    torch.set_num_threads(8)
    outputs = []
    costs = {}
    with gpu_slot(), torch.inference_mode():
        b = load(args.assets, dtype=torch.bfloat16)
        b.vae = None
        torch.cuda.empty_cache()
        b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
        for arm in ARMS:
            rows = read(args.prepared / (arm + "_inputs.json"))
            if len(rows) != 48:
                raise ValueError("complete fixed arm required")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for i, row in enumerate(rows):
                if any(k in row for k in ("expected", "references", "answer")):
                    raise ValueError("answer labels forbidden in inference index")
                codec = None
                if arm == "source_text":
                    if text_sha(row["text"]) != row["input_sha256"]:
                        raise ValueError("original visible text changed")
                    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + row["text"] + "\nFollow the request. Output only the requested answer, with no explanation.\n<|im_start|>assistant\n"
                    ids = torch.tensor([b.tokenizer(text, add_special_tokens=False).input_ids], device=b.device)
                    prefix = b.model.showo.model.embed_tokens(ids)
                    mask = b.native.masks.causal_attn_mask_naive(1, prefix.shape[1], b.device).to(b.dtype)
                else:
                    path = args.prepared / row["input"]
                    if sha(path) != row["input_sha256"]:
                        raise ValueError("source image changed")
                    latent, _ = image_latent(b, path, row["seed"])
                    if arm == "source_432" and row["condition"] == "normal":
                        raw = b.vae.batch_decode(latent[:, :, None]).squeeze(2)[0].float().permute(1, 2, 0).cpu().numpy()
                        target = args.output / f"codec/{i:02d}.png"
                        target.parent.mkdir(exist_ok=True)
                        Image.fromarray(np.rint((raw + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(target)
                        codec = dict(image=str(target.relative_to(args.output)), sha256=sha(target))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prefix, mask = reading_prefix(b, latent, ANSWER_INSTRUCTION)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
                result = {key: row[key] for key in ("id", "pair", "condition", "input_sha256", "seed", "readout_seed")}
                result.update(arm=arm, ids=ids, text=b.tokenizer.decode(ids, skip_special_tokens=True),
                    stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id), prefix_positions=prefix.shape[1],
                    codec=codec, student_answer=False)
                outputs.append(result)
                save_json(args.output / "outputs.json", outputs)
                save_json(args.output / "status.json", dict(state="running", pid=os.getpid(), arm=arm, complete_calls=len(outputs), planned_calls=144))
                print(arm, row["id"], repr(result["text"]), flush=True)
            costs[arm] = dict(elapsed_seconds=time.monotonic() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                includes_codec_reconstruction=arm == "source_432")
        save_json(args.output / "report.json", dict(cases=144, costs=costs, counts=b.counts,
            source_model_sha256=MODEL_SHA256, implementation_sha256=sha(__file__),
            preparation_sha256=sha(args.prepared / "preparation.json"), inputs_sha256=p["inputs_sha256"],
            cap=96, body_dtype="bfloat16", codec_dtype="float32", optimizer_updates=0, adapters_added=0,
            auxiliary_unicode_head=True, reference_answers_loaded=False,
            image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))


def score(args):
    p = authenticated_preparation(args)
    report = read(args.run / "report.json")
    if read(args.run / "status.json")["state"] != "completed" or report["cases"] != 144 or report["implementation_sha256"] != sha(__file__):
        raise ValueError("terminal source-bound full observation required")
    if report["preparation_sha256"] != sha(args.prepared / "preparation.json") or report["inputs_sha256"] != p["inputs_sha256"]:
        raise ValueError("model input bindings differ")
    refs = read(args.prepared / "references.json")
    outputs = read(args.run / "outputs.json")
    models = {arm: score_controls(read(args.prepared / (arm + "_inputs.json")), refs,
              [r for r in outputs if r["arm"] == arm]) for arm in ARMS}
    files = {}

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != sha(target):
            raise ValueError("evidence copy differs")
        files[name] = sha(target)

    for name in (*p["files_sha256"], "references.json", "preparation.json", *(a + "_inputs.json" for a in ARMS)):
        retain(args.prepared / name, name)
    for name in ("outputs.json", "report.json"):
        retain(args.run / name, "reports/" + name)
    source_text = {r["id"]: r["text"] for r in read(args.prepared / "source_text_inputs.json") if r["condition"] == "normal"}
    recognition = []
    for arm in ("source_1024", "source_432"):
        rows = [r for r in read(args.prepared / (arm + "_inputs.json")) if r["condition"] == "normal"]
        for row in rows:
            observations = [(arm, args.prepared / row["input"])]
            if arm == "source_432":
                out = next(r for r in outputs if r["arm"] == arm and r["id"] == row["id"])
                path = args.run / out["codec"]["image"]
                if sha(path) != out["codec"]["sha256"]:
                    raise ValueError("source codec pixels changed")
                retain(path, out["codec"]["image"])
                observations.append(("codec_432", path))
            for kind, path in observations:
                result = subprocess.run(["tesseract", str(path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                    env=dict(os.environ, OMP_THREAD_LIMIT="1"), capture_output=True, text=True, check=True)
                target, actual = canonical(source_text[row["id"]], "zh"), canonical(result.stdout, "zh")
                recognition.append(dict(id=row["id"], kind=kind, text=result.stdout, expected=source_text[row["id"]],
                    image_sha256=sha(path), canonical_exact=actual == target,
                    edit_errors=edit_distance(actual, target), reference_characters=len(target)))
                save_json(args.output / "recognition.json", recognition)
    files["recognition.json"] = sha(args.output / "recognition.json")
    calibration = {}
    for kind in ("source_1024", "source_432", "codec_432"):
        rows = [r for r in recognition if r["kind"] == kind]
        if len(rows) != 32:
            raise ValueError("complete source readability assessment required")
        calibration[kind] = dict(cases=32, exact=sum(r["canonical_exact"] for r in rows),
            character_edit_error=sum(r["edit_errors"] for r in rows) / sum(r["reference_characters"] for r in rows))
    retain(Path(__file__), "implementation/" + Path(__file__).name)
    save_json(args.output / "result.json", dict(models=models, source_readability=calibration,
        files_sha256=files, source_report=report, original_input_evidence="../native_visual_access_result/result.json",
        original_input_result_sha256=sha("references/evidence/native_visual_access_result/result.json"),
        independent_ocr_conditioning=False, cases_filtered_using_ocr=False,
        text_control_is_privileged=True, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, general_ilm_goal_complete=False,
        copyright="CMRC 2018: CC-BY-SA-4.0, Cui et al., EMNLP-IJCNLP 2019. Native source/model: NUS Show Lab, Apache-2.0."))
    print(json.dumps({a: {k: v for k, v in r.items() if k != "pairs"} for a, r in models.items()}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "evaluate", "score"), required=True)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_chinese_interface/prepared_v1"))
    p.add_argument("--run", type=Path, default=Path("artifacts/native_chinese_interface/source_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid()))
    try:
        {"prepare": prepare, "evaluate": evaluate, "score": score}[args.stage](args)
        save_json(args.output / "status.json", dict(state="completed", stage=args.stage, pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", stage=args.stage, pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
