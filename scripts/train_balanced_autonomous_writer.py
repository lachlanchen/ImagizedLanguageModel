#!/usr/bin/env python3
"""Resumable fixed-budget final-image learning with complete balanced exposure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, save_json, sha

INITIAL = "75e192a265069bdebe0b1e07de5339ce814631ccdbef8751b3e1ea75c16f5e5d"


def implementation():
    return {name: sha(ROOT / name) for name in ("scripts/train_balanced_autonomous_writer.py",
        "references/balanced_autonomous_writing_protocol.md", "ilm/image_edit/autonomous.py",
        "ilm/image_edit/paired.py", "ilm/image_edit/recognition.py")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "condition", "teacher", "admission", "initial", "manifest", "inputs", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    from ilm.image_edit.autonomous import encode_reference, final_image_loss, native_implementation, sample_page
    from ilm.image_edit.paired import (adapter_state, add_adapters, answer, load_pipeline,
        read_manifest, restore_adapters, save_answer, verify_source)
    from ilm.image_edit.recognition import RecognitionTeacher, verify_teacher

    code = implementation()
    if subprocess.check_output(["git", "status", "--porcelain", "--", *code], cwd=ROOT, text=True):
        raise ValueError("commit the exact learning implementation and protocol first")
    source = verify_source(args.snapshot)
    if sha(args.initial) != INITIAL:
        raise ValueError("the complete final-image pilot checkpoint is required")
    initial = torch.load(args.initial, map_location="cpu", weights_only=True)
    prior = initial["metadata"]
    if (initial["step"] != 32 or prior["rank"] != 16 or prior["source"] != source
            or prior["condition_sha256"] != sha(args.condition)
            or prior["manifest_sha256"] != sha(args.manifest)
            or prior["inputs_sha256"] != sha(args.inputs)
            or any(sha(ROOT / path) != digest for path, digest in prior["implementation"].items())):
        raise ValueError("starting source, implementation or data differs")
    admission = json.loads(args.admission.read_text())
    if (not admission["teacher_admitted"] or admission["source"] != verify_teacher(args.teacher)
            or prior["admission_sha256"] != sha(args.admission)
            or any(sha(ROOT / path) != digest for path, digest in admission["implementation"].items())):
        raise ValueError("unchanged complete reader admission required")
    training = [r for r in read_manifest(args.manifest) if r["task"] == "copy" and r["split"] == "train"][:32]
    ids = [r["id"] for r in training]
    if ids != prior["training_ids"] or sum(r["language"] == "zh" for r in training) != 16:
        raise ValueError("same balanced 32 examples required")
    order = torch.Generator().manual_seed(20268502); seeds = torch.Generator().manual_seed(20268503)
    schedule = [dict(update=epoch * 32 + j + 1, epoch=epoch + 1, id=ids[index],
        noise_seed=int(torch.randint(2**31 - 1, (1,), generator=seeds)))
        for epoch in range(4) for j, index in enumerate(torch.randperm(32, generator=order).tolist())]
    schedule_hash = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    metadata = dict(source=source, rank=16, implementation=code, initial_sha256=INITIAL,
        condition_sha256=sha(args.condition), manifest_sha256=sha(args.manifest), inputs_sha256=sha(args.inputs),
        baseline_index_sha256=sha(args.baseline), admission_sha256=sha(args.admission), training_ids=ids,
        purpose="balanced_full_trajectory_continuation", updates=128, starting_optimizer_step=32,
        optimizer_restored=True, learning_rate=1e-5, weight_decay=.01, gradient_clip=1.,
        seed=20268501, schedule_seed=20268502, noise_seed_sampler_seed=20268503,
        schedule_sha256=schedule_hash, steps=50, guidance=1., output_canvas=[512, 256],
        objective="full native 50-step autonomous image: OCR CE + 10 balanced unclipped RGB L1",
        teacher_at_inference=False, target_in_sampler=False, question_specific_text_condition=False,
        external_pretraining=True, output_repair=False, general_ilm_goal_complete=False)
    if args.output.exists() and args.resume is None:
        raise ValueError("preserve existing run; resume only an explicit valid checkpoint")
    if args.resume:
        state = json.loads((args.output / "status.json").read_text())
        if state.get("pid"):
            try: os.kill(state["pid"], 0)
            except ProcessLookupError: pass
            else: raise RuntimeError("recorded owner PID still exists; inspect before resuming")
    args.output.mkdir(parents=True, exist_ok=True)
    fixed = condition_cache(args.condition)
    inputs = {r["id"]: r for r in json.loads(args.inputs.read_text())}
    baseline = {r["id"]: r for r in json.loads(args.baseline.read_text())}
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    started = time.perf_counter()
    try:
        with gpu_zero() as device:
            torch.manual_seed(20268501); torch.cuda.manual_seed_all(20268501)
            pipe = load_pipeline(args.snapshot, device); add_adapters(pipe.transformer, 16)
            restore_adapters(pipe.transformer, initial["adapters"])
            start_adapters = initial["adapters"]
            metadata["native_implementation"] = native_implementation(pipe)
            if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in pipe.transformer.modules()):
                raise ValueError("deterministic full-trajectory replay required")
            parameters = [v for v in pipe.transformer.parameters() if v.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
            optimizer.load_state_dict(initial["optimizer"])
            del initial
            if any(int(state["step"]) != 32 for state in optimizer.state.values()):
                raise ValueError("all restored optimizer steps must equal 32")
            first, prior_seconds = 0, 0.
            if args.resume:
                saved = torch.load(args.resume, map_location="cpu", weights_only=True)
                if saved["metadata"] != metadata or saved["schedule_sha256"] != schedule_hash or not 0 < saved["step"] < 128:
                    raise ValueError("resume requires the same unfinished schedule and implementation")
                first = saved["step"]; prior_seconds = saved["cumulative_invocation_seconds"]
                restore_adapters(pipe.transformer, saved["adapters"]); optimizer.load_state_dict(saved["optimizer"])
                torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
                journal = [json.loads(s) for s in (args.output / "updates.jsonl").read_text().splitlines() if s]
                if len(journal) == first - 1:
                    with (args.output / "updates.jsonl").open("a") as f: f.write(json.dumps(saved["last_entry"]) + "\n")
                elif len(journal) != first or journal[-1] != saved["last_entry"]:
                    raise ValueError("checkpoint and complete journal disagree; preserve and inspect")
                del saved
            else:
                pipe.transformer.eval(); checks = []
                for row in training[:2]:
                    retained = baseline[row["id"]]; tensor = args.baseline.parent / Path(retained["path"]).parent / "pixels.pt"
                    if retained["input_sha256"] != row["input_sha256"] or sha(tensor) != retained["result"]["pixels_sha256"]:
                        raise ValueError("retained pilot input/pixels differ")
                    raw = answer(pipe, row["input_path"], fixed, seed=inputs[row["id"]]["seed"], width=512, height=256)
                    if not torch.equal(raw, torch.load(tensor, map_location="cpu", weights_only=True)):
                        raise AssertionError("initial pilot pixels fail native replay")
                    checks.append(dict(id=row["id"], native_pixels_bit_exact=True,
                        output=save_answer(pipe, raw, args.output / ("starting_" + row["language"]))))
                save_json(args.output / "initial_replays.json", checks)
                save_json(args.output / "schedule.json", schedule)
            if json.loads((args.output / "schedule.json").read_text()) != schedule:
                raise ValueError("saved schedule changed")
            reader = RecognitionTeacher(args.teacher, device); reader.model.train()
            pipe.transformer.train(); pipe.transformer.enable_gradient_checkpointing()
            for item in schedule[first:]:
                tick = time.perf_counter(); row = training[ids.index(item["id"])]
                if item["update"] == 1:
                    pipe.transformer.eval()
                    native_pixels = answer(pipe, row["input_path"], fixed, seed=item["noise_seed"], width=512, height=256)
                    pipe.transformer.train()
                input_features = encode_reference(pipe, row["input_path"])
                optimizer.zero_grad(set_to_none=True)
                result = sample_page(pipe, input_features, fixed["positive"], seed=item["noise_seed"],
                    differentiable=True, capture=item["update"] == 1)
                values = {}
                if item["update"] == 1:
                    if not torch.equal(native_pixels, result["rgb"].detach().cpu()):
                        raise AssertionError("actual differentiable forward differs from native pixels")
                    values["actual_differentiable_forward_native_pixel_exact"] = True
                    del native_pixels
                # Targets enter only after the actual noise-to-image procedure.
                target = torch.from_numpy(np.array(Image.open(row["target_path"]).convert("RGB")).copy()).permute(2, 0, 1)[None].to(device).float() / 255
                loss, components = final_image_loss(result, target, row["reference_text"], reader)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite final-image loss")
                loss.backward()
                if item["update"] == 1:
                    probes = {str(i): float(v.grad.float().norm()) if v.grad is not None and bool(torch.isfinite(v.grad).all()) else 0.
                        for i, v in result["probes"]}
                    if len(probes) != 3 or not all(v > 0 for v in probes.values()):
                        raise AssertionError("missing early/middle/late trajectory gradients")
                    values["trajectory_prediction_gradient_norms"] = probes
                if any(v.grad is not None for model in (pipe.vae, reader.model) for v in model.parameters()) or any(
                        v.grad is not None for v in pipe.transformer.parameters() if not v.requires_grad):
                    raise AssertionError("frozen weights received gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not bool(norm > 0): raise AssertionError("zero adapter gradient")
                folder = args.output / f"update_{item['update']:03}_before"
                if folder.exists():
                    old = torch.load(folder / "pixels.pt", map_location="cpu", weights_only=True)
                    if not torch.equal(old, result["rgb"].detach().cpu()):
                        raise ValueError("uncheckpointed saved page differs on exact resume")
                    output = dict(png_sha256=sha(folder / "answer.png"), pixels_sha256=sha(folder / "pixels.pt"), shape=list(old.shape), output_repair=False)
                else: output = save_answer(pipe, result["rgb"].detach().cpu(), folder)
                optimizer.step()
                if any(int(state["step"]) != 32 + item["update"] for state in optimizer.state.values()):
                    raise ValueError("optimizer continuation step differs")
                if torch.cuda.max_memory_allocated() >= 22 * 1024**3:
                    raise RuntimeError("registered allocation limit exceeded")
                entry = dict(item, loss=float(loss.detach()), **components, **values,
                    gradient_norm=float(norm), autonomous_page=output, elapsed_seconds=time.perf_counter() - tick)
                temporary = args.output / "checkpoint.tmp"
                torch.save(dict(adapters=adapter_state(pipe.transformer), optimizer=optimizer.state_dict(),
                    step=item["update"], schedule_sha256=schedule_hash, metadata=metadata, last_entry=entry,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    cumulative_invocation_seconds=prior_seconds + time.perf_counter() - started), temporary)
                temporary.replace(args.output / "checkpoint.pt")
                with (args.output / "updates.jsonl").open("a") as f: f.write(json.dumps(entry) + "\n")
                save_json(args.output / "status.json", dict(stage="training", update=item["update"], budget=128, pid=os.getpid(),
                    elapsed_seconds=prior_seconds + time.perf_counter() - started))
                print(json.dumps(entry), flush=True)
                del result, input_features, target, loss
            changed = sum(not torch.equal(v, start_adapters[name]) for name, v in adapter_state(pipe.transformer).items())
            if not changed or implementation() != code:
                raise AssertionError("unchanged source and changed adapters required")
            report = dict(stage="completed", updates=128, metadata=metadata, checkpoint_sha256=sha(args.output / "checkpoint.pt"),
                changed_adapter_tensors=changed, elapsed_seconds=prior_seconds + time.perf_counter() - started,
                peak_allocated_vram_gib_this_invocation=torch.cuda.max_memory_allocated() / 1024**3,
                trainable_parameters=sum(v.numel() for v in parameters),
                deployed_transformer_parameters=sum(v.numel() for v in pipe.transformer.parameters()),
                codec_parameters=sum(v.numel() for v in pipe.vae.parameters()),
                training_only_reader_parameters=sum(v.numel() for v in reader.model.parameters()),
                writing_qualification=False, general_ilm_goal_complete=False)
            save_json(args.output / "report.json", report); save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2), flush=True)
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None, reason=str(error))); raise


if __name__ == "__main__":
    main()
