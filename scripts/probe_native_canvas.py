#!/usr/bin/env python3
"""Compare complete-request layouts with one unchanged dense checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from PIL import Image, ImageChops
from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import compare, totals, score_controls
from score_native_english_transfer import score as score_english

CHECKPOINT = "016429ec5454cdd681693d881df4c6b7d52fb43dcae7b9e15857d2d9cd32b483"
ARMS = ("legacy", "square", "tight")


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    assessment = Path("artifacts/native_state_alignment/assessment_v1")
    original = read(assessment / "inputs.json")
    legacy_root = Path("artifacts/native_visual_access")
    legacy_copy = {r["id"]: r for r in read(legacy_root / "full_validation_inputs.json") if r["task"] == "copy"}
    evidence = Path("references/evidence")
    legacy_outputs = []
    receipts = {}
    for folder, name in (("native_visual_alignment_result", "reports/full_validation_v1_outputs.json"),
                         ("native_visual_alignment_result", "reports/question_readout_v1_outputs.json"),
                         ("native_english_transfer_result", "reports/after_outputs.json")):
        p = evidence / folder / name
        if sha(p) != read(evidence / folder / "result.json")["files_sha256"][name]:
            raise ValueError("retained baseline responses changed")
        values = read(p)
        if "full_validation" in name:
            values = [r for r in values if r["id"] in legacy_copy]
        legacy_outputs.extend(values)
        receipts[str(p)] = sha(p)
    prior = {r["id"]: r for r in legacy_outputs}
    if len(original) != 160 or len(prior) != 160:
        raise ValueError("complete matched 160-call legacy comparison required")
    args.output.mkdir(parents=True, exist_ok=False)
    files, checks = {}, []
    indices = {arm: [] for arm in ARMS}
    for i, source in enumerate(original):
        value = prior[source["id"]]
        full = assessment / source["input"]
        old = legacy_root / legacy_copy[source["id"]]["input"] if source["cohort"] == "copy" else full
        if sha(full) != source["input_sha256"] or sha(old) != value["input_sha256"]:
            raise ValueError("original or legacy whole-request raster changed")
        with Image.open(full) as image:
            square = image.convert("RGB")
        with Image.open(old) as image:
            legacy = image.convert("RGB")
        restored = Image.new("RGB", square.size, "white")
        restored.paste(legacy, (0, 0))
        if square.size != (1024, 1024) or ImageChops.difference(restored, square).getbbox() is not None:
            raise ValueError("legacy square restoration changes a source pixel")
        box = ImageChops.difference(square, Image.new("RGB", square.size, "white")).getbbox()
        if box is None:
            crop_box = (0, 0, 16, 16)
        else:
            crop_box = (max(0, (box[0] - 16) // 16 * 16), max(0, (box[1] - 16) // 16 * 16),
                min(1024, (box[2] + 16 + 15) // 16 * 16), min(1024, (box[3] + 16 + 15) // 16 * 16))
        tight = square.crop(crop_box)
        restored = Image.new("RGB", square.size, "white")
        restored.paste(tight, crop_box[:2])
        if ImageChops.difference(restored, square).getbbox() is not None:
            raise ValueError("whole-request tight crop loses a source pixel")
        row = {key: source[key] for key in ("id", "cohort", "task", "language", "pair", "condition", "source_id") if key in source}
        row.update(seed=value["seed"], readout_seed=value["readout_seed"])
        for arm, image in (("legacy", legacy), ("square", square), ("tight", tight)):
            name = f"{arm}/{i:03d}.png"
            (args.output / arm).mkdir(exist_ok=True)
            image.save(args.output / name)
            digest = sha(args.output / name)
            if arm == "legacy" and digest != value["input_sha256"]:
                # Preserve the exact original PNG container as well as pixels.
                shutil.copy2(old, args.output / name)
                digest = sha(args.output / name)
            files[name] = digest
            indices[arm].append(dict(row, input=name, input_sha256=digest, dimensions=list(image.size),
                visual_positions=image.width // 16 * (image.height // 16)))
        checks.append(dict(id=source["id"], source_sha256=source["input_sha256"], legacy_sha256=value["input_sha256"],
            exact_square_restoration=True, exact_tight_restoration=True, crop_box=crop_box, original_ink_box=box))
    for arm, rows in indices.items():
        save_json(args.output / (arm + "_inputs.json"), rows)
    save_json(args.output / "legacy_outputs.json", [prior[r["id"]] for r in original])
    save_json(args.output / "preparation.json", dict(cases_per_arm=160, planned_calls=480, arms=list(ARMS),
        inputs_sha256={arm: sha(args.output / (arm + "_inputs.json")) for arm in ARMS}, files_sha256=files,
        legacy_outputs_sha256=sha(args.output / "legacy_outputs.json"), baseline_receipts_sha256=receipts,
        source_assessment_sha256=sha(assessment / "inputs.json"), checkpoint_sha256=CHECKPOINT, checks=checks,
        implementation_sha256=sha(__file__), model_calls=0, optimizer_updates=0, content_resized=False,
        source_pixels_preserved=True, crops_use_answers_or_ocr=False, general_ilm_goal_complete=False))
    print("Prepared 480 whole-request layouts; every crop/pad reconstruction is exact.")


def authenticated(args):
    p = read(args.prepared / "preparation.json")
    for arm, digest in p["inputs_sha256"].items():
        if sha(args.prepared / (arm + "_inputs.json")) != digest:
            raise ValueError("fixed canvas cohort changed")
    for name, digest in p["files_sha256"].items():
        if sha(args.prepared / name) != digest:
            raise ValueError("whole-request raster changed")
    if sha(args.prepared / "legacy_outputs.json") != p["legacy_outputs_sha256"]:
        raise ValueError("legacy comparison changed")
    return p


def evaluate(args):
    import torch
    from ilm.native_unified.showo2_source import load, image_latent, reading_prefix
    from ilm.native_unified.reading import add_reading_adapters, reader_gate, restore, cached_decisions
    from train_native_visual_alignment import binding
    from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION

    preparation = authenticated(args)
    if sha(args.checkpoint) != CHECKPOINT:
        raise ValueError("the fixed completed dense checkpoint is required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["completed_updates"] != 2560 or saved["metadata"]["implementation"] != binding():
        raise ValueError("dense checkpoint implementation changed")
    prior = {r["id"]: r for r in read(args.prepared / "legacy_outputs.json")}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", pid=os.getpid()))
    try:
        torch.set_num_threads(8)
        with gpu_slot(), torch.inference_mode():
            b = load(args.assets, dtype=torch.bfloat16)
            add_reading_adapters(b, 16)
            restore(b.model, saved["adapters"])
            b.vae = None
            torch.cuda.empty_cache()
            b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
            b.model.eval()
            outputs, costs = [], {}
            with reader_gate(b.model):
                for arm in ARMS:
                    rows = read(args.prepared / (arm + "_inputs.json"))
                    started = time.monotonic()
                    torch.cuda.reset_peak_memory_stats()
                    for row in rows:
                        if any(k in row for k in ("expected", "transcript", "answer", "references")):
                            raise ValueError("input-only canvas assessment required")
                        z, _ = image_latent(b, args.prepared / row["input"], row["seed"])
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                            ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
                        if arm == "legacy" and ids != prior[row["id"]]["ids"]:
                            raise ValueError("legacy source readout does not replay exactly: " + row["id"])
                        outputs.append(dict(row, arm=arm, prefix_positions=prefix.shape[1], ids=ids,
                            text=b.tokenizer.decode(ids, skip_special_tokens=True), stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id)))
                        save_json(args.output / "outputs.json", outputs)
                        save_json(args.output / "status.json", dict(state="running", pid=os.getpid(), arm=arm,
                            complete_calls=len(outputs), planned_calls=480))
                        print(arm, row["id"], repr(outputs[-1]["text"]), flush=True)
                    costs[arm] = dict(calls=160, elapsed_seconds=time.monotonic() - started,
                        peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3)
            save_json(args.output / "report.json", dict(calls=480, costs=costs, checkpoint_sha256=CHECKPOINT,
                preparation_sha256=sha(args.prepared / "preparation.json"), outputs_sha256=sha(args.output / "outputs.json"),
                implementation=binding(), evaluator_sha256=sha(__file__), legacy_decision_replays_exact=160,
                cap=96, optimizer_updates=0, auxiliary_unicode_head=True, image_answer_generation_demonstrated=False,
                original_transcripts_loaded=False, annotated_references_loaded=False, general_ilm_goal_complete=False))
        save_json(args.output / "status.json", dict(state="completed", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


def score(args):
    preparation = authenticated(args)
    report = read(args.run / "report.json")
    if read(args.run / "status.json")["state"] != "completed" or report["calls"] != 480 or report["legacy_decision_replays_exact"] != 160 or report["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("complete fixed 480-call experiment required")
    if report["preparation_sha256"] != sha(args.prepared / "preparation.json") or report["outputs_sha256"] != sha(args.run / "outputs.json"):
        raise ValueError("canvas evidence changed")
    outputs = read(args.run / "outputs.json")
    if len(outputs) != 480:
        raise ValueError("all complete responses required")
    copy_refs = read("artifacts/native_state_alignment/prepared_v1/copy_references.json")
    zh_refs = read("artifacts/native_chinese_interface/prepared_v1/references.json")
    en_refs = read("artifacts/native_english_transfer/prepared_v2/references.json")
    scores = {}
    for arm in ARMS:
        rows = read(args.prepared / (arm + "_inputs.json"))
        out = [r for r in outputs if r["arm"] == arm]
        if [r["id"] for r in rows] != [r["id"] for r in out]:
            raise ValueError("fixed full canvas cohort order differs")
        for row, value in zip(rows, out):
            if any(row[k] != value[k] for k in ("input_sha256", "seed", "readout_seed", "cohort")):
                raise ValueError("canvas input or matched seed differs")
        copying = [compare(r, ref) for r, ref in zip([r for r in out if r["cohort"] == "copy"], copy_refs)]
        scores[arm] = dict(copy=dict(summary=totals(copying), cases=copying,
            languages={lang: totals([r for r in copying if r["language"] == lang]) for lang in ("zh", "en")}),
            zh=score_controls([r for r in rows if r["cohort"] == "zh"], zh_refs, [r for r in out if r["cohort"] == "zh"]),
            en=score_english([r for r in rows if r["cohort"] == "en"], en_refs, [r for r in out if r["cohort"] == "en"]))
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(path, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[name] = sha(target)
        if files[name] != sha(path):
            raise ValueError("retained canvas evidence changed")

    for name in ("report.json", "outputs.json"):
        retain(args.run / name, "reports/" + name)
    for name in ("preparation.json", "legacy_outputs.json", *[arm + "_inputs.json" for arm in ARMS]):
        retain(args.prepared / name, "inputs/" + name)
    for name in preparation["files_sha256"]:
        retain(args.prepared / name, "inputs/" + name)
    save_json(args.output / "result.json", dict(arms=scores, files_sha256=files, costs=report["costs"],
        checkpoint_sha256=CHECKPOINT, source_evaluator_sha256=report["evaluator_sha256"], scorer_sha256=sha(__file__),
        legacy_decision_replays_exact=160, actual_calls=480, optimizer_updates=0, source_pixels_preserved=True,
        content_resized=False, original_source_432_english_reference=11, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, general_ilm_goal_complete=False,
        copyright="CMRC 2018/SQuAD 1.1 CC-BY-SA-4.0, Cui et al. 2019/Rajpurkar et al. 2016; Show-o2 NUS Show Lab Apache-2.0."))
    print(json.dumps({arm: dict(copy=s["copy"]["summary"], zh=s["zh"]["normal"], en=s["en"]["summary"]) for arm, s in scores.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "evaluate", "score"), required=True)
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_canvas_control/prepared_v1"))
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/native_visual_alignment/learning_v1/checkpoint.pt"))
    p.add_argument("--run", type=Path, default=Path("artifacts/native_canvas_control/assessment_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
