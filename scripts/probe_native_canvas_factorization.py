#!/usr/bin/env python3
"""Cross fixed source visual features with short/long native language contexts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from PIL import Image, ImageChops
import torch

from ilm.image_edit.baseline import save_json, sha
from ilm.native_unified.showo2_source import image_latent, reading_prefix
from ilm.native_unified.reading import reader_gate, cached_decisions
from ilm.native_unified.state_alignment import visual_start
from train_native_reader_attention import source_bundle, binding as source_binding, SOURCE
from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION
from score_native_visual_access import compare, totals

ARMS = ("strip_short", "square_long", "square_short", "strip_long")


def read(path):
    return json.loads(Path(path).read_text())


def binding():
    return dict(source_binding(), **{__file__.replace(str(Path.cwd()) + "/", ""): sha(__file__)})


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cohort(args):
    receipt = read(args.inputs / "result.json")
    if receipt["checkpoint_sha256"] != SOURCE or receipt["legacy_decision_replays_exact"] != 160:
        raise ValueError("complete fixed-source canvas calibration required")
    for name, digest in receipt["files_sha256"].items():
        if sha(args.inputs / name) != digest:
            raise ValueError("source canvas evidence changed")
    rows = {arm: [r for r in read(args.inputs / "inputs" / (arm + "_inputs.json")) if r["cohort"] == "copy"] for arm in ("legacy", "square")}
    if any(len(r) != 32 for r in rows.values()) or [r["id"] for r in rows["legacy"]] != [r["id"] for r in rows["square"]]:
        raise ValueError("complete matched 32-copy cohort required")
    originals = read(args.inputs / "reports/outputs.json")
    prior = {arm: {r["id"]: r for r in originals if r["arm"] == arm and r["cohort"] == "copy"} for arm in rows}
    for short, full in zip(rows["legacy"], rows["square"]):
        if any(short[k] != full[k] for k in ("id", "seed", "readout_seed", "language")):
            raise ValueError("matched source seeds/identity required")
        for row in (short, full):
            if sha(args.inputs / "inputs" / row["input"]) != row["input_sha256"]:
                raise ValueError("source pixels differ from their inventory")
        with Image.open(args.inputs / "inputs" / short["input"]) as s, Image.open(args.inputs / "inputs" / full["input"]) as q:
            if s.size != (1024, 256) or q.size != (1024, 1024):
                raise ValueError("fixed source geometries required")
            restored = Image.new("RGB", q.size, "white")
            restored.paste(s.convert("RGB"), (0, 0))
            if ImageChops.difference(restored, q.convert("RGB")).getbbox() is not None:
                raise ValueError("source pair differs beyond white bottom padding")
    return receipt, rows, prior


def evaluate(args):
    receipt, rows, prior = cohort(args)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", pid=os.getpid()))
    try:
        torch.set_num_threads(8)
        with gpu_slot(), torch.inference_mode():
            b = source_bundle(args)
            b.vae = None
            torch.cuda.empty_cache()
            b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
            outputs, cached, masks, checks = [], {"legacy": {}, "square": {}}, {}, []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            start = visual_start(b)

            def infer(arm, row, prefix, mask, hashes):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
                out = dict(id=row["id"], language=row["language"], seed=row["seed"], readout_seed=row["readout_seed"],
                    arm=arm, prefix_positions=prefix.shape[1], prefix_sha256=tensor_sha(prefix), source_inputs_sha256=hashes,
                    hybrid_prefix=arm in ("square_short", "strip_long"), ids=ids, text=b.tokenizer.decode(ids, skip_special_tokens=True),
                    stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id))
                return out

            def retain(out):
                outputs.append(out)
                save_json(args.output / "outputs.json", outputs)
                save_json(args.output / "status.json", dict(state="running", pid=os.getpid(), arm=out["arm"], complete_calls=len(outputs), planned_calls=128))
                print(out["arm"], out["id"], repr(out["text"]), flush=True)

            with reader_gate(b.model):
                for layout, arm, visual in (("legacy", "strip_short", 1024), ("square", "square_long", 4096)):
                    for row in rows[layout]:
                        z, _ = image_latent(b, args.inputs / "inputs" / row["input"], row["seed"])
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                        if prefix.shape[1] != visual + 39 or not bool(torch.isfinite(prefix).all()):
                            raise ValueError("source visual layout differs")
                        out = infer(arm, row, prefix, mask, {layout: row["input_sha256"]})
                        if out["ids"] != prior[layout][row["id"]]["ids"]:
                            raise ValueError("complete native replay failed: " + arm + " " + row["id"])
                        cached[layout][row["id"]] = prefix.cpu()
                        if layout not in masks:
                            masks[layout] = mask.cpu()
                        elif not torch.equal(masks[layout], mask.cpu()):
                            raise ValueError("native source mask differs within fixed layout")
                        retain(out)
                if len(outputs) != 64:
                    raise ValueError("finish every source replay before hybrid calls")
                for short, full in zip(rows["legacy"], rows["square"]):
                    s, q = cached["legacy"][short["id"]], cached["square"][short["id"]]
                    if not torch.equal(s[:, :start].view(torch.uint8), q[:, :start].view(torch.uint8)) or not torch.equal(s[:, start + 1024:].view(torch.uint8), q[:, start + 4096:].view(torch.uint8)):
                        raise ValueError("nonvisual source header/suffix differs")
                    a, c = s[:, start:start + 1024].float(), q[:, start:start + 1024].float()
                    checks.append(dict(id=short["id"], nonvisual_header_bit_exact=True, nonvisual_suffix_bit_exact=True,
                        top_features_mean_absolute_difference=float((a - c).abs().mean()),
                        top_features_maximum_absolute_difference=float((a - c).abs().max())))
                for arm in ("square_short", "strip_long"):
                    for short, full in zip(rows["legacy"], rows["square"]):
                        s, q = [cached[k][short["id"]].to(b.device) for k in ("legacy", "square")]
                        if arm == "square_short":
                            prefix = torch.cat((q[:, :start + 1024], q[:, start + 4096:]), dim=1)
                            mask = masks["legacy"].to(b.device)
                        else:
                            prefix = torch.cat((q[:, :start], s[:, start:start + 1024], q[:, start + 1024:]), dim=1)
                            mask = masks["square"].to(b.device)
                        if prefix.shape[1] != (1063 if arm == "square_short" else 4135):
                            raise ValueError("hybrid layout differs from declared source context")
                        retain(infer(arm, short, prefix, mask, {"legacy": short["input_sha256"], "square": full["input_sha256"]}))
            save_json(args.output / "report.json", dict(calls=128, source_decision_replays_exact=64, checks=checks,
                source_checkpoint_sha256=SOURCE, source_result_sha256=sha(args.inputs / "result.json"), implementation=binding(),
                outputs_sha256=sha(args.output / "outputs.json"), elapsed_seconds=time.monotonic() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                complete_model_parameters=sum(p.numel() for p in b.model.parameters()), codec_parameters=b.counts["vae"],
                trainable_parameters=sum(p.numel() for p in b.model.parameters() if p.requires_grad), cap=96,
                optimizer_updates=0, annotated_references_loaded=False, auxiliary_unicode_head=True,
                hybrid_prefixes_are_not_real_image_encodings=True, unique_internal_cause_established=False,
                image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))
        save_json(args.output / "status.json", dict(state="completed", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


def score(args):
    receipt, rows, prior = cohort(args)
    report = read(args.run / "report.json")
    if read(args.run / "status.json")["state"] != "completed" or report["calls"] != 128 or report["source_decision_replays_exact"] != 64 or report["implementation"] != binding():
        raise ValueError("completed code-bound factorization required")
    if report["source_result_sha256"] != sha(args.inputs / "result.json") or report["outputs_sha256"] != sha(args.run / "outputs.json") or len(report["checks"]) != 32 or report["source_checkpoint_sha256"] != SOURCE:
        raise ValueError("source/output/prefix evidence changed")
    if [c["id"] for c in report["checks"]] != [r["id"] for r in rows["legacy"]] or not all(c["nonvisual_header_bit_exact"] and c["nonvisual_suffix_bit_exact"] for c in report["checks"]):
        raise ValueError("complete nonvisual equality checks required")
    reference_root = Path("references/evidence/native_state_alignment_result")
    refs_path = reference_root / "data/training_copy_references.json"
    if sha(refs_path) != read(reference_root / "result.json")["files_sha256"]["data/training_copy_references.json"]:
        raise ValueError("original copying references changed")
    refs = read(refs_path)
    outputs = read(args.run / "outputs.json")
    if len(outputs) != 128 or [r["id"] for r in refs] != [r["id"] for r in rows["legacy"]]:
        raise ValueError("all declared outputs and references required")
    arms = {}
    for arm in ARMS:
        out = [r for r in outputs if r["arm"] == arm]
        if [r["id"] for r in out] != [r["id"] for r in refs]:
            raise ValueError("complete fixed arm order required")
        for row, square, value in zip(rows["legacy"], rows["square"], out):
            if any(row[k] != value[k] for k in ("seed", "readout_seed", "language")):
                raise ValueError("matched seed or language changed")
            expected_hashes = {"legacy": row["input_sha256"], "square": square["input_sha256"]}
            if arm in ("strip_short", "square_long"):
                layout = "legacy" if arm == "strip_short" else "square"
                expected_hashes = {layout: expected_hashes[layout]}
                if any(value[k] != prior[layout][row["id"]][k] for k in ("ids", "text", "stopped")):
                    raise ValueError("source replay evidence differs")
            if value["source_inputs_sha256"] != expected_hashes or value["prefix_positions"] != (1063 if arm.endswith("short") else 4135) or value["hybrid_prefix"] != (arm in ("square_short", "strip_long")):
                raise ValueError("declared arm input or layout differs")
        cases = [compare(o, ref) for o, ref in zip(out, refs)]
        arms[arm] = dict(summary=totals(cases), cases=cases, languages={lang: totals([r for r in cases if r["language"] == lang]) for lang in ("en", "zh")})
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ("report.json", "outputs.json"):
        target = args.output / "reports" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.run / name, target)
        files[str(target.relative_to(args.output))] = sha(target)
    save_json(args.output / "result.json", dict(arms=arms, files_sha256=files, execution=report, implementation=binding(),
        actual_calls=128, optimizer_updates=0, source_result_sha256=sha(args.inputs / "result.json"),
        reference_result_sha256=sha(reference_root / "result.json"), auxiliary_unicode_head=True,
        natural_qa_assessed=False, unique_internal_cause_established=False, image_answer_generation_demonstrated=False,
        general_ilm_goal_complete=False, copyright="Synthetic copying requests; Show-o2 source/model NUS Show Lab Apache-2.0."))
    print(json.dumps({arm: {lang: value["exact_stopped"] for lang, value in data["languages"].items()} for arm, data in arms.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("evaluate", "score"), required=True)
    p.add_argument("--inputs", type=Path, default=Path("references/evidence/native_canvas_control_result"))
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--source", type=Path, default=Path("artifacts/native_visual_alignment/learning_v1/checkpoint.pt"))
    p.add_argument("--run", type=Path, default=Path("artifacts/native_canvas_factorization/assessment_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    {"evaluate": evaluate, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
