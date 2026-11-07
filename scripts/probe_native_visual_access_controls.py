#!/usr/bin/env python3
"""Prepare and run paired-question controls for the auxiliary visual reader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import save_json, sha
from ilm.image_edit.data import font_and_coverage, page
from ilm.image_edit.paired_data import qa_candidates


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    manifest = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    rows = [r for r in manifest if r["split"] == "validation" and r["task"] == "qa"]
    if len(rows) != 32:
        raise ValueError("the complete 32-case held-out QA cohort is required")
    source = {r["id"]: r for pair in qa_candidates(args.source) for r in pair}
    font, provenance = font_and_coverage([source[r["id"]] for r in rows])
    original_font = read(args.manifest.parent / "preparation.json")["font"]
    for key in ("font_file_sha256", "font_face_index", "font_size"):
        if provenance[key] != original_font[key]:
            raise ValueError("the original source font differs")
    train_pairs = {r.get("pair") for r in manifest if r["split"] == "train" and r["task"] == "qa"}
    groups = {}
    for r in rows:
        groups.setdefault(r["pair"], []).append(r)
    if len(groups) != 16 or set(groups) & train_pairs or any(len(v) != 2 for v in groups.values()):
        raise ValueError("sixteen complete held-out passage pairs are required")
    (args.output / "erased").mkdir()
    inputs, references, checks = [], [], []
    for index, (pair, members) in enumerate(groups.items()):
        erased_arrays = []
        seed, readout_seed = 20269021 + index, 20269121 + index
        for r in members:
            s = source[r["id"]]
            if hashlib.sha256(s["prompt"].encode()).hexdigest() != r["input_geometry"]["text_sha256"]:
                raise ValueError("reconstructed source wording differs")
            path = args.manifest.parent / r["input"]
            if sha(path) != r["input_sha256"]:
                raise ValueError("original input bytes changed")
            rendered, geometry = page(s["prompt"], (1024, 1024), font)
            original = np.asarray(Image.open(path).convert("RGB"))
            if geometry != r["input_geometry"] or not np.array_equal(original, np.asarray(rendered)):
                raise ValueError("reconstructed full-page pixels or geometry differ")
            before_question, separator, _ = s["prompt"].rpartition("\n问题：")
            if not separator:
                raise ValueError("explicit source question paragraph missing")
            _, context_geometry = page(before_question, (1024, 1024), font)
            top = 32 + 48 * context_geometry["lines"]
            if not 0 < top < 1024:
                raise ValueError("invalid question-band boundary")
            erased = original.copy()
            erased[top:] = 255
            if not np.array_equal(erased[:top], original[:top]) or np.any(erased[top:] != 255):
                raise ValueError("context-preservation check failed")
            erased_arrays.append(erased)
            inputs.append(dict(id=r["id"], pair=pair, condition="normal",
                input=os.path.relpath(path.resolve(), args.output.resolve()), input_sha256=sha(path),
                seed=seed, readout_seed=readout_seed))
            references.append(dict(id=r["id"], pair=pair, expected=r["reference_text"], language=r["language"]))
            checks.append(dict(id=r["id"], question_band_top=top,
                context_pixels_identical=True, source_geometry_verified=True))
        if not np.array_equal(*erased_arrays):
            raise ValueError("same-passage erased pages must be pixel-identical")
        if members[0]["reference_text"] == members[1]["reference_text"]:
            raise ValueError("paired questions must request different answers")
        name = f"erased/{index:02}.png"
        Image.fromarray(erased_arrays[0]).save(args.output / name)
        inputs.append(dict(id="erased/" + pair, pair=pair, condition="erased",
            input=name, input_sha256=sha(args.output / name), seed=seed, readout_seed=readout_seed))
    save_json(args.output / "inputs.json", inputs)
    save_json(args.output / "references.json", references)
    save_json(args.output / "preparation.json", dict(cases=48, normal_cases=32, erased_cases=16,
        complete_passage_pairs=16, manifest_sha256=sha(args.manifest), source_sha256=sha(args.source),
        inputs_sha256=sha(args.output / "inputs.json"), references_sha256=sha(args.output / "references.json"),
        implementation_sha256=sha(__file__), source_font=provenance, checks=checks,
        erasure="Blank from the question paragraph through the bottom, including the constant answer marker; all context pixels remain unchanged.",
        region_selected_from_answer=False, context_pixels_identical=True,
        paired_erased_pixels_identical=True, same_latent_and_readout_seed_per_passage=True,
        input_index_contains_answers=False, model_calls=0, training_overlap=0,
        interpretation="Compare two normal answers against one shared erased-page output per passage. The shared output is not two independent calls. A change alone does not establish correctness or language understanding.",
        general_ilm_goal_complete=False))
    print("Prepared 32 normal and 16 shared erased-page controls; no model calls.")


def evaluate(args):
    import torch
    from ilm.native_unified.showo2_source import load, image_latent, reading_prefix
    from ilm.native_unified.reading import add_reading_adapters, reader_gate, restore, cached_decisions
    from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION
    from train_native_visual_access import code_binding

    if args.checkpoint is None:
        raise ValueError("learned numeric adapters are required")
    preparation = read(args.inputs.parent / "preparation.json")
    if sha(args.inputs) != preparation["inputs_sha256"]:
        raise ValueError("prepared input-only index changed")
    rows = read(args.inputs)
    if len(rows) != 48 or sum(r["condition"] == "erased" for r in rows) != 16:
        raise ValueError("complete fixed 48-call paired-question cohort required")
    torch.set_num_threads(8)
    with gpu_slot(), torch.inference_mode():
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["metadata"]["implementation"] != code_binding() or saved["completed_updates"] != 4096:
            raise ValueError("complete checkpoint with matching implementation required")
        b = load(args.assets, dtype=torch.bfloat16)
        add_reading_adapters(b, 16)
        restore(b.model, saved["adapters"])
        b.vae = None
        torch.cuda.empty_cache()
        b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
        b.model.eval()
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        outputs = []
        with reader_gate(b.model):
            for r in rows:
                path = args.inputs.parent / r["input"]
                if sha(path) != r["input_sha256"]:
                    raise ValueError("prepared input pixels changed")
                z, _ = image_latent(b, path, r["seed"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                    generated = cached_decisions(b, prefix, mask, seed=r["readout_seed"], cap=96)
                result = {k: r[k] for k in ("id", "pair", "condition", "input_sha256", "seed", "readout_seed")}
                result.update(ids=generated, text=b.tokenizer.decode(generated, skip_special_tokens=True),
                    stopped=bool(generated and generated[-1] == b.tokenizer.eos_token_id), student_answer=False)
                outputs.append(result)
                save_json(args.output / "outputs.json", outputs)
                save_json(args.output / "status.json", dict(state="running", complete_calls=len(outputs), planned_calls=48))
                print(r["id"], r["condition"], repr(result["text"]), flush=True)
        save_json(args.output / "report.json", dict(cases=len(outputs),
            inputs_sha256=sha(args.inputs), checkpoint_sha256=sha(args.checkpoint),
            completed_updates=saved["completed_updates"], implementation=code_binding(),
            evaluator_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            auxiliary_unicode_head=True, teacher_answers_loaded=False, language_quality_qualified=False,
            general_ilm_goal_complete=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "evaluate"), required=True)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--source", type=Path, default=Path("data/visual_qa/cmrc2018_short/train.jsonl"))
    p.add_argument("--inputs", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", stage=args.stage))
    try:
        if args.stage == "prepare":
            prepare(args)
        else:
            if args.inputs is None:
                raise ValueError("an input-only prepared index is required")
            evaluate(args)
        save_json(args.output / "status.json", dict(state="completed", stage=args.stage))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
