#!/usr/bin/env python3
"""Diagnose fit on fixed seen prompts without another optimizer update."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import canonical

ARMS = ("source_image", "learner_image", "source_text")
GROUPS = (("copy", "zh"), ("copy", "en"), ("qa", "zh"), ("qa", "en"))


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    training = read(args.training / "training.json")
    preparation = read(args.training / "preparation.json")
    cache = read(args.root / "cache_v1/manifest.json")
    if sha(args.training / "training.json") != preparation["training_sha256"] or cache["training_sha256"] != preparation["training_sha256"]:
        raise ValueError("authenticated 640-example inventory required")
    cached = {r["id"]: r for r in cache["records"]}
    selected = [r for task, language in GROUPS for r in [r for r in training if (r["task"], r["language"]) == (task, language)][:8]]
    if len(selected) != 32 or len({r["id"] for r in selected}) != 32:
        raise ValueError("eight unique seen prompts in every group required")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, references, text_inputs = [], [], []
    for i, row in enumerate(selected):
        high = args.training / row["input"]
        low = args.root / "cache_v1" / cached[row["id"]]["coarse_input"]
        if sha(high) != row["input_sha256"] or sha(low) != cached[row["id"]]["coarse_input_sha256"]:
            raise ValueError("seen source/derived page changed")
        visible_sha = hashlib.sha256(row["transcript"].encode()).hexdigest()
        if visible_sha != row["transcript_sha256"]:
            raise ValueError("original visible transcript changed")
        inputs.append(dict(id=row["id"], task=row["task"], language=row["language"],
            input=os.path.relpath(high.resolve(), args.output.resolve()), input_sha256=sha(high),
            coarse_input=os.path.relpath(low.resolve(), args.output.resolve()), coarse_input_sha256=sha(low),
            seed=row["seed"], readout_seed=20270621 + i))
        alternatives = [a["text"] for a in row.get("annotations", [])] or [row["expected"]]
        references.append(dict(id=row["id"], task=row["task"], language=row["language"],
            expected=row["expected"], alternatives=alternatives))
        text_inputs.append(dict(id=row["id"], text=row["transcript"], transcript_sha256=visible_sha))
    for name, value in (("inputs.json", inputs), ("references.json", references), ("text_inputs.json", text_inputs)):
        save_json(args.output / name, value)
    save_json(args.output / "preparation.json", dict(cases=32, selection="first eight per task/language in training inventory order",
        training_sha256=preparation["training_sha256"], cache_manifest_sha256=sha(args.root / "cache_v1/manifest.json"),
        files_sha256={name: sha(args.output / name) for name in ("inputs.json", "references.json", "text_inputs.json")},
        implementation_sha256=sha(__file__), model_calls=0, optimizer_updates=0, seen_training_prompts=True,
        held_out_assessment=False, general_ilm_goal_complete=False))
    print("Prepared 32 fixed seen prompts; no model calls.")


def authenticated(args):
    preparation = read(args.prepared / "preparation.json")
    for name, digest in preparation["files_sha256"].items():
        if sha(args.prepared / name) != digest:
            raise ValueError("fixed seen cohort changed")
    rows = read(args.prepared / "inputs.json")
    if len(rows) != 32 or Counter((r["task"], r["language"]) for r in rows) != Counter({g: 8 for g in GROUPS}):
        raise ValueError("complete balanced seen cohort required")
    return preparation, rows


def evaluate(args):
    import torch
    from ilm.native_unified.showo2_source import load, image_latent, reading_prefix
    from ilm.native_unified.reading import add_reading_adapters, reader_gate, restore, cached_decisions
    from ilm.native_unified.state_alignment import add_bridge, corrected_prefix, teacher_prefix
    from train_native_state_alignment import binding, BUDGET
    from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION

    preparation, rows = authenticated(args)
    primary = read(args.primary / "result.json")
    if read(args.root / "controller_status.json")["state"] != "completed" or primary["actual_updates"] != BUDGET:
        raise ValueError("wait for completed primary controller and assessment")
    checkpoint = args.root / "learning_v1/checkpoint.pt"
    if sha(checkpoint) != primary["training"]["checkpoint_sha256"]:
        raise ValueError("final assessed checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["completed_updates"] != BUDGET or saved["metadata"]["implementation"] != binding():
        raise ValueError("completed unchanged learner required")
    if primary["training"]["metadata"]["cache_manifest_sha256"] != preparation["cache_manifest_sha256"]:
        raise ValueError("seen prompts belong to another training cache")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", pid=os.getpid()))
    outputs = []
    try:
        torch.set_num_threads(8)
        with gpu_slot(), torch.inference_mode():
            bundle = load(args.assets, dtype=torch.bfloat16)
            bundle.vae = None
            torch.cuda.empty_cache()
            bundle.vae = bundle.native.vae.WanVAE(vae_pth=str(bundle.files.vae), dtype=torch.float32, device=bundle.device)
            bundle.model.eval()
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for arm in ("source_image", "source_text", "learner_image"):
                texts = {r["id"]: r for r in read(args.prepared / "text_inputs.json")} if arm == "source_text" else None
                if arm == "learner_image":
                    add_reading_adapters(bundle, 16)
                    add_bridge(bundle)
                    restore(bundle.model, saved["adapters"])
                    bundle.model.eval()
                for row in rows:
                    if texts is not None:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prefix, mask = teacher_prefix(bundle, texts[row["id"]]["text"])
                    else:
                        high, low = [args.prepared / row[k] for k in ("input", "coarse_input")]
                        if sha(high) != row["input_sha256"] or sha(low) != row["coarse_input_sha256"]:
                            raise ValueError("seen source/derived pixels changed")
                        coarse, _ = image_latent(bundle, low, row["seed"])
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prefix, mask = reading_prefix(bundle, coarse, ANSWER_INSTRUCTION)
                        if arm == "learner_image":
                            fine, _ = image_latent(bundle, high, row["seed"])
                            with reader_gate(bundle.model), torch.autocast("cuda", dtype=torch.bfloat16):
                                prefix, mask = corrected_prefix(bundle, fine, prefix, mask)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ids = cached_decisions(bundle, prefix, mask, seed=row["readout_seed"], cap=96)
                    out = dict(row, arm=arm, ids=ids, text=bundle.tokenizer.decode(ids, skip_special_tokens=True),
                        stopped=bool(ids and ids[-1] == bundle.tokenizer.eos_token_id))
                    outputs.append(out)
                    save_json(args.output / "outputs.json", outputs)
                    save_json(args.output / "status.json", dict(state="running", pid=os.getpid(), complete_calls=len(outputs), planned_calls=96))
                    print(arm, row["id"], repr(out["text"]), flush=True)
            save_json(args.output / "report.json", dict(calls=96, preparation_sha256=sha(args.prepared / "preparation.json"),
                outputs_sha256=sha(args.output / "outputs.json"), checkpoint_sha256=sha(checkpoint), implementation=binding(),
                evaluator_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3, cap=96,
                image_arms_receive_original_text=False, reference_answers_loaded=False, new_optimizer_updates=0,
                auxiliary_unicode_head=True, image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))
        save_json(args.output / "status.json", dict(state="completed", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


def score(args):
    preparation, rows = authenticated(args)
    report = read(args.run / "report.json")
    if read(args.run / "status.json")["state"] != "completed" or report["calls"] != 96 or report["preparation_sha256"] != sha(args.prepared / "preparation.json") or report["outputs_sha256"] != sha(args.run / "outputs.json"):
        raise ValueError("complete authenticated seen-prompt readouts required")
    references = {r["id"]: r for r in read(args.prepared / "references.json")}
    outputs = read(args.run / "outputs.json")
    groups, cases = {}, []
    for arm in ARMS:
        selected = [r for r in outputs if r["arm"] == arm]
        if [r["id"] for r in selected] != [r["id"] for r in rows]:
            raise ValueError("all 32 seen prompts in fixed order required per arm")
        for row, out in zip(rows, selected):
            if any(row[k] != out[k] for k in ("input_sha256", "coarse_input_sha256", "seed", "readout_seed", "task", "language")):
                raise ValueError("seen input/seed differs")
            ref = references[row["id"]]
            actual = canonical(out["text"], row["language"])
            case = dict(id=row["id"], arm=arm, task=row["task"], language=row["language"], text=out["text"],
                expected=ref["expected"], alternatives=ref["alternatives"], stopped=out["stopped"],
                training_target_exact_stopped=out["stopped"] and actual == canonical(ref["expected"], row["language"]),
                annotation_exact_stopped=out["stopped"] and actual in {canonical(t, row["language"]) for t in ref["alternatives"]})
            cases.append(case)
        for task, language in GROUPS:
            group = [r for r in cases if (r["arm"], r["task"], r["language"]) == (arm, task, language)]
            groups[f"{arm}/{task}/{language}"] = dict(cases=8,
                training_target_exact_stopped=sum(r["training_target_exact_stopped"] for r in group),
                annotation_exact_stopped=sum(r["annotation_exact_stopped"] for r in group), stopped=sum(r["stopped"] for r in group))
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for source, name in [(args.prepared / n, "data/" + n) for n in ("preparation.json", "inputs.json", "references.json", "text_inputs.json")] + [(args.run / n, "reports/" + n) for n in ("report.json", "outputs.json")]:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = sha(target)
        if files[name] != sha(source):
            raise ValueError("evidence copy differs")
    save_json(args.output / "result.json", dict(groups=groups, cases=cases, files_sha256=files,
        checkpoint_sha256=report["checkpoint_sha256"], implementation_sha256=sha(__file__), new_optimizer_updates=0,
        seen_training_prompts=True, generalization_demonstrated=False, image_answer_generation_demonstrated=False,
        general_ilm_goal_complete=False))
    print(json.dumps(groups, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "evaluate", "score"), required=True)
    p.add_argument("--root", type=Path, default=Path("artifacts/native_state_alignment"))
    p.add_argument("--training", type=Path, default=Path("artifacts/native_state_alignment/prepared_v1"))
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_state_fit/prepared_v1"))
    p.add_argument("--primary", type=Path, default=Path("references/evidence/native_state_alignment_result"))
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--run", type=Path, default=Path("artifacts/native_state_fit/assessment_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
