#!/usr/bin/env python3
"""Train on final images from the complete native noise-to-image trajectory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, open_page, save_json, sha

INITIAL_CHECKPOINT = "23332544624497e12b1c9b18f21b65a0543bea025a44eb18fa933caa5a5723af"


def implementation():
    return {name: sha(ROOT / name) for name in ("ilm/image_edit/autonomous.py",
        "ilm/image_edit/recognition.py", "ilm/image_edit/paired.py",
        "scripts/train_autonomous_writer.py", "references/autonomous_writing_protocol.md",
        "references/autonomous_sampler_correction.md")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "train"), required=True)
    for name in ("snapshot", "condition", "teacher", "admission", "initial", "cache",
                 "manifest", "inputs", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    from ilm.image_edit.autonomous import encode_reference, final_image_loss, native_implementation, sample_page
    from ilm.image_edit.paired import (adapter_state, add_adapters, load_pair,
        load_pipeline, read_manifest, restore_adapters, save_answer, save_checkpoint, verify_source)
    from ilm.image_edit.recognition import RecognitionTeacher, verify_teacher

    source = verify_source(args.snapshot)
    if sha(args.initial) != INITIAL_CHECKPOINT:
        raise ValueError("the declared failed-copying checkpoint is required")
    saved_initial = torch.load(args.initial, map_location="cpu", weights_only=True)
    if (saved_initial["metadata"]["source"] != source or saved_initial["metadata"]["rank"] != 16
            or saved_initial["metadata"]["condition_sha256"] != sha(args.condition)):
        raise ValueError("initial source/condition/rank differs")
    admission = json.loads(args.admission.read_text())
    if (not admission["teacher_admitted"] or admission["source"] != verify_teacher(args.teacher)
            or admission["manifest_sha256"] != sha(args.manifest)
            or any(sha(ROOT / path) != digest for path, digest in admission["implementation"].items())):
        raise ValueError("complete unchanged reader admission required")
    training = [r for r in read_manifest(args.manifest) if r["task"] == "copy" and r["split"] == "train"][:32]
    if len(training) != 32:
        raise ValueError("same 32 training examples required")
    ids = [r["id"] for r in training]
    cached = json.loads((args.cache / "manifest.json").read_text())
    by_id = {r["id"]: r for r in cached["records"]}
    for row in training:
        if any(by_id[row["id"]][k + "_sha256"] != row[k + "_sha256"] for k in ("input", "target")):
            raise ValueError("training page/cache binding differs")
    inputs = {r["id"]: r for r in json.loads(args.inputs.read_text())}
    baseline = {r["id"]: r for r in json.loads(args.baseline.read_text())}
    fixed = condition_cache(args.condition)
    code = implementation()
    updates = 2 if args.stage == "preflight" else 32
    metadata = dict(source=source, implementation=code, rank=16, purpose=args.stage,
        updates=updates, condition_sha256=sha(args.condition), initial_sha256=INITIAL_CHECKPOINT,
        teacher_source=admission["source"], admission_sha256=sha(args.admission),
        training_cache_manifest_sha256=sha(args.cache / "manifest.json"),
        manifest_sha256=sha(args.manifest), inputs_sha256=sha(args.inputs),
        baseline_index_sha256=sha(args.baseline), training_ids=ids,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        learning_rate=1e-5, weight_decay=.01, gradient_clip=1., seed=20268401,
        sampler_seed=20268402, noise_seed_sampler_seed=20268403,
        objective="full native 50-step autonomous image: OCR CE + 10 balanced unclipped RGB L1",
        steps=50, guidance=1., output_canvas=[512, 256], trajectory_gradient_steps=50,
        target_in_sampler=False, teacher_at_inference=False, output_repair=False,
        question_specific_text_condition=False, external_pretraining=True,
        language_quality_qualified=False, general_ilm_goal_complete=False)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    started = time.perf_counter()
    try:
        with gpu_zero() as device:
            torch.manual_seed(20268401); torch.cuda.manual_seed_all(20268401)
            pipe = load_pipeline(args.snapshot, device)
            add_adapters(pipe.transformer, 16)
            restore_adapters(pipe.transformer, saved_initial["adapters"])
            del saved_initial
            metadata["native_implementation"] = native_implementation(pipe)
            if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in pipe.transformer.modules()):
                raise ValueError("source gradient replay must be deterministic")
            equivalence = []
            # Perform the actual native comparison before enabling optimization.
            if args.stage == "preflight":
                for row in training[:2]:
                    pair = load_pair(args.cache, by_id[row["id"]], device)
                    native_reference = encode_reference(pipe, row["input_path"])
                    feature_difference = float((native_reference - pair["input"]).abs().max())
                    del pair
                    retained = baseline[row["id"]]
                    if retained["input_sha256"] != row["input_sha256"]:
                        raise ValueError("retained native baseline input differs")
                    prior_file = args.baseline.parent / retained["path"]
                    prior_tensor = prior_file.parent / "pixels.pt"
                    if sha(prior_tensor) != retained["result"]["pixels_sha256"]:
                        raise ValueError("retained baseline tensor changed")
                    reference_trace = []
                    def observe(_, index, timestep, values):
                        reference_trace.append(values["latents"].detach().cpu().clone())
                        return values
                    pipe.transformer.eval()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        native = pipe(image=open_page(row["input_path"]),
                            prompt_embeds=fixed["positive"].to(device),
                            negative_prompt_embeds=fixed["negative"].to(device),
                            generator=torch.Generator(device=device).manual_seed(inputs[row["id"]]["seed"]),
                            width=512, height=256, num_inference_steps=50, guidance_scale=1.,
                            callback_on_step_end=observe, callback_on_step_end_tensor_inputs=["latents"],
                            output_type="pt").images
                        actual = sample_page(pipe, native_reference, fixed["positive"],
                            seed=inputs[row["id"]]["seed"], capture=True)
                    retained_pixels = torch.load(prior_tensor, map_location="cpu", weights_only=True)
                    checks = dict(id=row["id"], steps=50, historical_input_feature_max_difference=feature_difference,
                        all_step_tensors_exact=len(reference_trace) == len(actual["traces"]) == 50
                            and all(torch.equal(a, b) for a, b in zip(reference_trace, actual["traces"])),
                        final_pixels_exact=torch.equal(native, actual["rgb"]),
                        retained_native_pixels_exact=torch.equal(native.cpu(), retained_pixels))
                    save_json(args.output / ("equivalence_" + row["language"] + ".json"), checks)
                    torch.save(dict(native=reference_trace, custom=actual["traces"]),
                        args.output / ("trace_" + row["language"] + ".pt"))
                    save_answer(pipe, actual["rgb"].detach().cpu(), args.output / ("before_" + row["language"]))
                    if not all(checks[k] for k in ("all_step_tensors_exact", "final_pixels_exact", "retained_native_pixels_exact")):
                        raise AssertionError("differentiable sampler does not reproduce actual native generation")
                    equivalence.append(checks)
                    del native_reference, native, actual, retained_pixels, reference_trace
            reader = RecognitionTeacher(args.teacher, device)
            reader.model.train()
            pipe.transformer.train(); pipe.transformer.enable_gradient_checkpointing()
            parameters = [p for p in pipe.transformer.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
            sampler = torch.Generator().manual_seed(20268402)
            noise_sampler = torch.Generator().manual_seed(20268403)
            initial = adapter_state(pipe.transformer)
            first_step = 0
            if args.resume:
                saved = torch.load(args.resume, map_location="cpu", weights_only=True)
                comparison = dict(saved["metadata"], source_commit=metadata["source_commit"])
                if comparison != metadata or not 0 < saved["step"] < updates:
                    raise ValueError("resume requires unchanged source/code/data and unfinished budget")
                metadata = saved["metadata"]
                restore_adapters(pipe.transformer, saved["adapters"])
                optimizer.load_state_dict(saved["optimizer"])
                sampler.set_state(saved["sampler_state"]); noise_sampler.set_state(saved["generator_state"])
                torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
                first_step = saved["step"]
            for step in range(first_step, updates):
                tick = time.perf_counter()
                index = step if args.stage == "preflight" else int(torch.randint(32, (1,), generator=sampler))
                row = training[index]
                seed = inputs[row["id"]]["seed"] if args.stage == "preflight" else int(torch.randint(2**31 - 1, (1,), generator=noise_sampler))
                # Encode the input exactly as native inference does. No answer
                # image or target code is loaded into the sampler's trajectory.
                input_features = encode_reference(pipe, row["input_path"]).requires_grad_(args.stage == "preflight")
                optimizer.zero_grad(set_to_none=True)
                result = sample_page(pipe, input_features, fixed["positive"], seed=seed,
                    differentiable=True, capture=args.stage == "preflight")
                target = torch.from_numpy(np.array(Image.open(row["target_path"]).convert("RGB")).copy()).permute(2, 0, 1)[None].to(device).float() / 255
                loss, values = final_image_loss(result, target, row["reference_text"], reader)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite autonomous final-image loss")
                loss.backward()
                if args.stage == "preflight":
                    probes = {str(i): dict(norm=float(v.grad.float().norm()) if v.grad is not None else None,
                        finite_nonzero=v.grad is not None and bool(torch.isfinite(v.grad).all()) and bool(v.grad.abs().max() > 0))
                        for i, v in result["probes"]}
                    gradient = input_features.grad
                    values.update(trajectory_prediction_gradients=probes,
                        input_gradient_norm=float(gradient.float().norm()) if gradient is not None else None)
                    if (not all(g["finite_nonzero"] for g in probes.values()) or len(probes) != 3
                            or gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0)):
                        raise AssertionError("missing full-trajectory or visual-reference gradients")
                if any(p.grad is not None for model in (pipe.vae, reader.model) for p in model.parameters()) or any(
                        p.grad is not None for p in pipe.transformer.parameters() if not p.requires_grad):
                    raise AssertionError("frozen source/codec/reader weights received gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not bool(norm > 0):
                    raise AssertionError("zero autonomous student gradient")
                optimizer.step()
                output = save_answer(pipe, result["rgb"].detach().cpu(), args.output / f"update_{step + 1:03}_before")
                entry = dict(update=step + 1, id=row["id"], noise_seed=seed,
                    loss=float(loss.detach()), **values, gradient_norm=float(norm),
                    autonomous_page=output, elapsed_seconds=time.perf_counter() - tick)
                with (args.output / "updates.jsonl").open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                save_json(args.output / "status.json", dict(stage="training", update=step + 1, budget=updates,
                    loss=entry["loss"], pid=os.getpid(), elapsed_seconds=time.perf_counter() - started))
                if torch.cuda.max_memory_allocated() >= 22 * 1024**3:
                    raise RuntimeError("registered 22 GiB allocation limit exceeded")
                save_checkpoint(args.output / "checkpoint.pt", pipe, optimizer, step=step + 1,
                    generator=noise_sampler, sampler=sampler, metadata=metadata)
                print(json.dumps(entry), flush=True)
                del result, input_features, target, loss
            changed = sum(not torch.equal(v, initial[n]) for n, v in adapter_state(pipe.transformer).items())
            if not changed or implementation() != code:
                raise AssertionError("no changed adapters or live implementation changed")
            # Retain native after-update images without a teacher in that path.
            if args.stage == "preflight":
                with torch.no_grad():
                    for row in training[:2]:
                        features = encode_reference(pipe, row["input_path"])
                        observed = sample_page(pipe, features, fixed["positive"], seed=inputs[row["id"]]["seed"])
                        save_answer(pipe, observed["rgb"].cpu(), args.output / ("after_" + row["language"]))
                        del observed, features
            report = dict(stage=args.stage, updates=updates, metadata=metadata,
                checkpoint_sha256=sha(args.output / "checkpoint.pt"), equivalence=equivalence,
                changed_adapter_tensors=changed, trainable_parameters=sum(p.numel() for p in parameters),
                deployed_transformer_parameters=sum(p.numel() for p in pipe.transformer.parameters()),
                codec_parameters=sum(p.numel() for p in pipe.vae.parameters()),
                training_only_reader_parameters=sum(p.numel() for p in reader.model.parameters()),
                elapsed_seconds=time.perf_counter() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                quality_qualification=False, general_ilm_goal_complete=False)
            save_json(args.output / "report.json", report)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None, reason=str(error)))
        raise


if __name__ == "__main__":
    main()
