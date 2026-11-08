#!/usr/bin/env python3
"""Register and execute a native/compiled/pruned image-only preflight.

prepare is CPU-only and checks complete actual source headers against real
meta-device component state dictionaries. run waits for the current QA owner,
then uses GPU0 once. Failures are preserved; no automatic quality rerolls.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.image_edit import visual_semantic as vs

FILES = ("scripts/preflight_visual_semantic.py", "ilm/image_edit/visual_semantic.py",
    "ilm/image_edit/compiled_visual_condition.py", "ilm/image_edit/baseline.py",
    "references/visual_semantic_native_preflight_protocol.md")
CONTROLS = ROOT / "artifacts/paired_language_controls_v1"
PAGES = ROOT / "artifacts/image_edit_learning/pages_v2"
CONDITION = ROOT / "artifacts/visual_task_condition_1024_v1/runtime"
QA_RUN = ROOT / "artifacts/native_paired_qa_learning_v2"
MANIFEST_SHA = "e2d6aec134a99e079a9bd42f217559a04e32727bdbc4b09d131b0fb22cf4dfa6"


def identity(*, clean):
    if clean and subprocess.check_output(["git", "status", "--porcelain", "--", *FILES], cwd=ROOT, text=True):
        raise ValueError("commit the exact preflight code and protocol first")
    return {n: sha(ROOT / n) for n in FILES}


def owner(pid):
    p = Path("/proc") / str(pid)
    try:
        return dict(pid=pid, command=(p / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
            start_ticks=(p / "stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def structure():
    import torch
    from safetensors import safe_open
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    module = vs.source_module()
    with torch.device("meta"):
        transformer = SD3Transformer2DModel.from_config(json.loads((vs.SNAPSHOT / "transformer/config.json").read_text()))
        vae = AutoencoderKL.from_config(json.loads((vs.SNAPSHOT / "vae/config.json").read_text()))
        vlm = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig.from_dict(json.loads((vs.SNAPSHOT / "vlm/config.json").read_text())))
        pipe = module.DeepGenPipeline(transformer=transformer, vae=vae,
            scheduler=module.FlowMatchEulerDiscreteScheduler.from_config(json.loads((vs.SNAPSHOT / "scheduler/scheduler_config.json").read_text())))
        pipe.lmm = vlm; pipe.connector_module = vs.connector(module, meta=True)
    vs.attach_body_aliases(vlm)
    components = dict(transformer=pipe.transformer, vae=pipe.vae, vlm=pipe.lmm, connector=pipe.connector_module)
    shards = dict(transformer=["transformer/diffusion_pytorch_model.safetensors"],
        vae=["vae/diffusion_pytorch_model.safetensors"], connector=["connector/model.safetensors"],
        vlm=["vlm/model-00001-of-00002.safetensors", "vlm/model-00002-of-00002.safetensors"])
    checks = {}
    for name, model in components.items():
        actual, mappings = {}, {}
        for file in shards[name]:
            with safe_open(str(vs.SNAPSHOT / file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    mapped = key
                    if name == "vlm":
                        if key.startswith("visual."):
                            mapped = "model." + key
                        elif key.startswith("model.") and not key.startswith("model.language_model."):
                            mapped = "model.language_model." + key[len("model."):]
                    if mapped in actual:
                        raise ValueError("duplicate stored component key")
                    actual[mapped] = tuple(handle.get_slice(key).get_shape())
                    if key != mapped: mappings[key] = mapped
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        missing = set(expected) - set(actual)
        if name == "vlm":
            missing -= {"lm_head.weight"}
        if missing or set(actual) - set(expected) or any(expected[k] != v for k, v in actual.items()):
            raise ValueError(f"stored {name} shapes/keys differ: missing={missing}, extra={set(actual)-set(expected)}")
        checks[name] = dict(stored_tensor_keys=len(actual), expected_state_keys=len(expected),
            source_keys_and_shapes_match=True, explicit_legacy_key_translations=len(mappings),
            unused_output_head_absent=(name == "vlm" and "lm_head.weight" not in actual),
            real_parameters_all_meta=all(p.is_meta for p in model.parameters()))
    aliases = dict(visual_same_object=vlm.visual is vlm.model.visual,
        language_body_same_object=vlm.language_model is vlm.model.language_model,
        no_duplicate_registered_modules=("visual" not in vlm._modules and "language_model" not in vlm._modules))
    if not all(aliases.values()):
        raise ValueError("compatibility aliases duplicate or replace a real body")
    removal = vs.remove_lexical_modules(pipe)
    return dict(component_checks=checks, compatibility_alias_checks=aliases, prototype_removal=removal,
        transformer_native_class_upgraded=type(pipe.transformer) is module.SD3Transformer2DModel,
        model_weight_tensors_loaded=False, model_forward_performed=False, cuda_initialized=torch.cuda.is_initialized(),
        parameter_counts_are_meta_prototypes=True, loaded_runtime_fit_or_quality_established=False)


def prepare(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("hide CUDA for CPU preparation")
    code = identity(clean=True)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", dict(phase="preparing_CPU", pid=os.getpid()))
    try:
        source = vs.verify_release()
        inspected = structure()
        save_json(output / "structure.json", inspected)
        from ilm.image_edit.compiled_visual_condition import CompiledVisualCondition
        fixed = CompiledVisualCondition.from_pretrained(CONDITION)
        if fixed.queries != 128 or fixed.prompt_embeds.shape != (2, 1082, 2048):
            raise ValueError("registered complete-page compiled geometry differs")
        control = json.loads((CONTROLS / "preparation.json").read_text())
        rows, source_indices = [], {}
        for variant in ("normal", "question_erased", "passage_erased"):
            path = CONTROLS / ("preflight_" + variant + "_inputs.json")
            if sha(path) != control["index_files_sha256"][path.name]:
                raise ValueError("original preflight control index changed")
            source_indices[path.name] = sha(path)
            for r in json.loads(path.read_text()):
                if set(r) != {"id", "input", "input_sha256", "seed"}:
                    raise ValueError("source index carries extra question or answer fields")
                target = (path.parent / r["input"]).resolve()
                rows.append(dict(id=variant + ":" + r["id"], input=os.path.relpath(target, output),
                    input_sha256=r["input_sha256"], seed=r["seed"]))
        if sha(PAGES / "pairs.jsonl") != MANIFEST_SHA:
            raise ValueError("prepared image manifest changed")
        manifest = [json.loads(x) for x in (PAGES / "pairs.jsonl").read_text().splitlines()]
        for language in ("zh", "en"):
            r = next(r for r in manifest if r["task"] == "copy" and r["split"] == "validation" and r["language"] == language)
            rows.append(dict(id=r["id"], input=os.path.relpath(PAGES / r["input"], output),
                input_sha256=r["input_sha256"], seed=20269101 + len(rows)))
        from PIL import Image
        if len(rows) != 10 or len({r["id"] for r in rows}) != 10:
            raise ValueError("complete ten-case cohort required")
        for r in rows:
            path = output / r["input"]
            if sha(path) != r["input_sha256"] or Image.open(path).size != (1024, 1024):
                raise ValueError("complete native input page differs")
        if rows[4]["input_sha256"] != rows[5]["input_sha256"] or rows[4]["seed"] != rows[5]["seed"]:
            raise ValueError("erased paired questions must share actual input bytes and seed")
        save_json(output / "inputs.json", rows)
        result = dict(code_sha256=code, source_sha256=source, revision=vs.REVISION,
            condition_folder=str(CONDITION.relative_to(ROOT)), condition_manifest_sha256=sha(CONDITION / "manifest.json"),
            condition_payload_sha256=sha(CONDITION / "condition.safetensors"), inputs_sha256=sha(output / "inputs.json"),
            complete_input_cases=10, actual_generation_calls_planned=12, steps=50, guidance=4.,
            width=1024, height=1024, source_index_sha256=source_indices,
            preparation_source_manifest_sha256=MANIFEST_SHA, structure_sha256=sha(output / "structure.json"),
            source_prep_has_answer_access=True, generation_index_has_answers=False,
            generation_index_fields=["id", "input", "input_sha256", "seed"],
            task_or_model_selected_from_output_quality=False, no_checkpoint_training_in_this_preflight=True,
            qa_run_must_complete=str(QA_RUN.relative_to(ROOT)), generation_performed=False,
            language_learning_established=False, general_ilm_goal_complete=False)
        if identity(clean=True) != code:
            raise ValueError("preflight source changed during CPU preparation")
        save_json(output / "preparation.json", result)
        save_json(output / "status.json", dict(phase="prepared", pid=None))
        print(json.dumps(dict(component_checks=inspected["component_checks"], prototype_removal=inspected["prototype_removal"],
            complete_input_cases=10, model_forward_performed=False), indent=2))
    except BaseException as error:
        save_json(output / "status.json", dict(phase="preparation_failed", pid=None, reason=str(error)))
        raise


def run(output, wait_pid):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("admit only GPU0")
    prepared = json.loads((output / "preparation.json").read_text())
    code = identity(clean=True)
    if prepared["code_sha256"] != code or sha(output / "inputs.json") != prepared["inputs_sha256"]:
        raise ValueError("registered source or input-only cohort changed")
    folder = output / "generation"; folder.mkdir(exist_ok=False)
    expected = owner(wait_pid) if wait_pid else None
    if wait_pid and (expected is None or "scripts/run_native_paired_qa.py" not in expected["command"] or QA_RUN.name not in expected["command"]):
        raise ValueError("specified wait owner is not the currently live registered QA coordinator")
    phase = "waiting_QA"; started = time.time()
    def status(**extra):
        save_json(folder / "runtime.json", dict(phase=phase, pid=None if phase in ("failed", "ready_for_raw_review") else os.getpid(),
            wait_owner=expected, started_unix=started, gpu=None if phase == "waiting_QA" else 0,
            novnc_url=None, code_sha256=code, general_ilm_goal_complete=False, **extra))
    def unchanged():
        if identity(clean=True) != code or sha(output / "inputs.json") != prepared["inputs_sha256"]:
            raise ValueError("bound preflight source or index changed")
    def interrupted(signum, frame): raise InterruptedError("preflight interrupted with signal " + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(signum, interrupted)
    try:
        status()
        while expected is not None and owner(wait_pid) == expected:
            unchanged(); time.sleep(15)
        terminal = json.loads((QA_RUN / "runtime.json").read_text())
        assessed = json.loads((QA_RUN / "assessment.json").read_text()) if (QA_RUN / "assessment.json").exists() else {}
        if (terminal["phase"] != "ready_for_raw_review" or terminal["child_pid"] is not None or
            assessed.get("actual_qa_assessment_calls") != 256 or assessed.get("actual_final_copying_calls") != 64):
            raise ValueError("complete current native QA assessment before a second model job")
        unchanged(); phase = "verifying_release"; status()
        if vs.verify_release() != prepared["source_sha256"]:
            raise ValueError("registered source weights changed")
        for file, key in (("manifest.json", "condition_manifest_sha256"), ("condition.safetensors", "condition_payload_sha256")):
            if sha(CONDITION / file) != prepared[key]: raise ValueError("compiled condition changed")
        rows = json.loads((output / "inputs.json").read_text())
        for r in rows:
            if set(r) != {"id", "input", "input_sha256", "seed"} or sha(output / r["input"]) != r["input_sha256"]:
                raise ValueError("input-only page index changed")
        with gpu_zero() as device:
            import torch
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            phase = "loading_native"; status()
            pipe, loader = vs.load_pipeline(device)
            save_json(folder / "loader.json", loader)
            outputs = []
            def call(row, name):
                unchanged()
                path = output / row["input"]
                if sha(path) != row["input_sha256"]: raise ValueError("live input page changed")
                t = time.perf_counter()
                value = vs.generate(pipe, path, seed=row["seed"], width=prepared["width"], height=prepared["height"],
                    steps=prepared["steps"], guidance=prepared["guidance"])
                saved = vs.save_generation(value, folder / name)
                saved["path"] = str(Path(saved["path"]).relative_to(folder))
                saved.update(id=row["id"], input_sha256=row["input_sha256"], seed=row["seed"], elapsed_seconds=time.perf_counter()-t)
                return value, saved
            phase = "native_replay"; status()
            native, n_saved = call(rows[0], "native_probe")
            vs.install_compiled(pipe, CONDITION)
            compiled, c_saved = call(rows[0], "compiled_probe")
            first = vs.exact_replay(native, compiled)
            save_json(folder / "native_compiled_replay.json", first)
            if not all(first["forward_fields_exact"].values()) or not first["decoded_tensor_exact"] or not first["png_pixels_exact"]:
                raise ValueError("compiled task differs from native real image generation")
            removal = vs.remove_lexical_modules(pipe)
            save_json(folder / "lexical_removal.json", removal)
            pruned, p_saved = call(rows[0], "000")
            second = vs.exact_replay(native, pruned)
            save_json(folder / "native_pruned_replay.json", second)
            if not all(second["forward_fields_exact"].values()) or not second["decoded_tensor_exact"] or not second["png_pixels_exact"]:
                raise ValueError("lexical module removal changes native image generation")
            outputs.append(p_saved); save_json(folder / "outputs.json", outputs)
            del native, compiled, pruned
            phase = "generating_pruned"
            for i, row in enumerate(rows[1:], 1):
                value, saved = call(row, f"{i:03}")
                outputs.append(saved); del value
                save_json(folder / "outputs.json", outputs); status(completed=len(outputs), total=10)
            left = torch.load(folder / "004/actual_tensors.pt", map_location="cpu", weights_only=True)["pixels"]
            right = torch.load(folder / "005/actual_tensors.pt", map_location="cpu", weights_only=True)["pixels"]
            if not torch.equal(left, right): raise ValueError("identical question-erased paired inputs/seeds generated different actual tensors")
            unchanged()
            result = dict(preparation_sha256=sha(output / "preparation.json"), complete_original_generations=12,
                complete_input_cases=10, complete_baseline_pages=10, native_probe=n_saved, compiled_probe=c_saved,
                native_compiled_replay=first, native_pruned_replay=second, lexical_removal=removal,
                question_erased_pair_actual_decoded_tensors_equal=True, parameters=vs.parameters(pipe),
                constant_condition_buffer_elements=sum(b.numel() for b in pipe._ilm_condition.buffers()),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 1024**3,
                elapsed_seconds=time.time()-started, source=prepared["source_sha256"], code_sha256=code,
                package_versions={n:version(n) for n in ("torch", "diffusers", "transformers", "safetensors")},
                runtime_lexical_table_or_head=False, runtime_tokenizer=False, external_teacher_or_OCR_used=False,
                changing_question_text_or_answer_loaded=False, output_repair=False,
                language_pretraining_retained=True, foundation_model_trained_here=False,
                raw_review_completed=False, language_quality_established=False, general_ilm_goal_complete=False)
            save_json(folder / "report.json", result)
        phase = "ready_for_raw_review"; status(completed=10, owned_GPU_job_finished=True)
    except BaseException as error:
        phase = "failed"; status(reason=str(error), exception=type(error).__name__); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-qa-pid", type=int)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    prepare(args.output.resolve()) if args.stage == "prepare" else run(args.output.resolve(), args.wait_for_qa_pid)
