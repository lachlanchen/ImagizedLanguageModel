#!/usr/bin/env python3
"""Continue the old assisted objective with the pilot's exposure and optimizer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, open_page, save_json, sha

INITIAL = "23332544624497e12b1c9b18f21b65a0543bea025a44eb18fa933caa5a5723af"
PROTOCOL = "references/autonomous_writing_control_protocol.md"


def implementation():
    names = ("scripts/train_autonomous_writing_control.py", PROTOCOL,
        "scripts/train_recognition_guided_writer.py", "ilm/image_edit/autonomous.py",
        "ilm/image_edit/paired.py", "ilm/image_edit/recognition.py")
    return {name: sha(ROOT / name) for name in names}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "condition", "teacher", "admission", "initial", "cache",
                 "manifest", "inputs", "baseline", "pilot", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    import torch
    from ilm.image_edit.autonomous import encode_reference, native_implementation
    from ilm.image_edit.paired import (adapter_state, add_adapters, answer, load_pair,
        load_pipeline, read_manifest, restore_adapters, save_answer, save_checkpoint, verify_source)
    from ilm.image_edit.recognition import RecognitionTeacher, verify_teacher
    from scripts.train_recognition_guided_writer import loss_with_reader

    code = implementation()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *code], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("commit the control implementation and protocol before execution")
    source = verify_source(args.snapshot)
    if sha(args.initial) != INITIAL:
        raise ValueError("the unchanged declared starting checkpoint is required")
    initial = torch.load(args.initial, map_location="cpu", weights_only=True)
    if (initial["metadata"]["source"] != source or initial["metadata"]["rank"] != 16
            or initial["metadata"]["condition_sha256"] != sha(args.condition)):
        raise ValueError("starting source, condition or rank differs")
    admission = json.loads(args.admission.read_text())
    if (not admission["teacher_admitted"] or admission["source"] != verify_teacher(args.teacher)
            or admission["manifest_sha256"] != sha(args.manifest)
            or any(sha(ROOT / path) != digest for path, digest in admission["implementation"].items())):
        raise ValueError("unchanged complete training-reader admission required")
    pilot = json.loads((args.pilot / "report.json").read_text())
    meta = pilot["metadata"]
    if (pilot["updates"] != 32 or meta["initial_sha256"] != INITIAL
            or meta["source"] != source or meta["condition_sha256"] != sha(args.condition)
            or meta["manifest_sha256"] != sha(args.manifest)
            or meta["inputs_sha256"] != sha(args.inputs)
            or meta["baseline_index_sha256"] != sha(args.baseline)
            or meta["admission_sha256"] != sha(args.admission)
            or any(sha(ROOT / path) != digest for path, digest in meta["implementation"].items())):
        raise ValueError("complete unchanged pilot required")
    for key, value in dict(seed=20268401, sampler_seed=20268402, noise_seed_sampler_seed=20268403,
            learning_rate=1e-5, weight_decay=.01, gradient_clip=1.).items():
        if meta[key] != value:
            raise ValueError("pilot optimizer or RNG declaration differs: " + key)
    training = [r for r in read_manifest(args.manifest) if r["task"] == "copy" and r["split"] == "train"][:32]
    ids = [r["id"] for r in training]
    if ids != meta["training_ids"] or len(ids) != 32:
        raise ValueError("same 32 copying examples required")
    cached = json.loads((args.cache / "manifest.json").read_text())
    if sha(args.cache / "manifest.json") != meta["training_cache_manifest_sha256"]:
        raise ValueError("original target-feature cache changed")
    by_id = {r["id"]: r for r in cached["records"]}
    for row in training:
        if any(by_id[row["id"]][k + "_sha256"] != row[k + "_sha256"] for k in ("input", "target")):
            raise ValueError("original cached page binding differs")
    journal = [json.loads(line) for line in (args.pilot / "updates.jsonl").read_text().splitlines() if line]
    sampler = torch.Generator().manual_seed(20268402)
    seeds = torch.Generator().manual_seed(20268403)
    if len(journal) != 32:
        raise ValueError("complete pilot journal required")
    for i, item in enumerate(journal):
        if (item["update"] != i + 1 or item["id"] != ids[int(torch.randint(32, (1,), generator=sampler))]
                or item["noise_seed"] != int(torch.randint(2**31 - 1, (1,), generator=seeds))):
            raise ValueError("pilot's registered example/seed trajectory does not reproduce")
    inputs = {r["id"]: r for r in json.loads(args.inputs.read_text())}
    baseline = {r["id"]: r for r in json.loads(args.baseline.read_text())}
    fixed = condition_cache(args.condition)
    metadata = dict(source=source, rank=16, implementation=code, initial_sha256=INITIAL,
        condition_sha256=sha(args.condition), manifest_sha256=sha(args.manifest),
        inputs_sha256=sha(args.inputs), baseline_index_sha256=sha(args.baseline),
        admission_sha256=sha(args.admission), teacher_source=admission["source"],
        training_cache_manifest_sha256=sha(args.cache / "manifest.json"), training_ids=ids,
        pilot_report_sha256=sha(args.pilot / "report.json"), pilot_journal_sha256=sha(args.pilot / "updates.jsonl"),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        purpose="additional_old_objective_control", updates=32, seed=20268401,
        sampler_seed=20268402, noise_seed_sampler_seed=20268403,
        learning_rate=1e-5, weight_decay=.01, gradient_clip=1., fresh_optimizer=True,
        objective="unchanged uniform-sigma velocity MSE + .05 OCR CE at sigma <= .5",
        native_autocast_input_encoding=True, original_target_feature_cache=True,
        matched_update_exposure=True, matched_compute=False, matched_realized_noise=False,
        retrospectively_specified_control=True, teacher_at_inference=False,
        output_repair=False, external_pretraining=True, general_ilm_goal_complete=False)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    started = time.perf_counter()
    try:
        with gpu_zero() as device:
            torch.manual_seed(20268401); torch.cuda.manual_seed_all(20268401)
            pipe = load_pipeline(args.snapshot, device)
            add_adapters(pipe.transformer, 16)
            restore_adapters(pipe.transformer, initial["adapters"])
            del initial
            metadata["native_implementation"] = native_implementation(pipe)
            pipe.transformer.eval()
            replay_started = time.perf_counter(); replays = []
            for row in training[:2]:
                retained = baseline[row["id"]]
                tensor = args.baseline.parent / Path(retained["path"]).parent / "pixels.pt"
                if (retained["input_sha256"] != row["input_sha256"]
                        or sha(tensor) != retained["result"]["pixels_sha256"]):
                    raise ValueError("retained actual starting pixels changed")
                raw = answer(pipe, row["input_path"], fixed, seed=inputs[row["id"]]["seed"], width=512, height=256)
                if not torch.equal(raw, torch.load(tensor, map_location="cpu", weights_only=True)):
                    raise AssertionError("starting native output does not reproduce")
                observed = save_answer(pipe, raw, args.output / ("starting_" + row["language"]))
                replays.append(dict(id=row["id"], native_pixels_bit_exact=True, **observed))
            replay_seconds = time.perf_counter() - replay_started
            reader = RecognitionTeacher(args.teacher, device)
            reader.model.train(); pipe.transformer.train(); pipe.transformer.enable_gradient_checkpointing()
            parameters = [v for v in pipe.transformer.parameters() if v.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
            old = adapter_state(pipe.transformer)
            optimization_started = time.perf_counter(); active = 0
            for item in journal:
                tick = time.perf_counter(); row = training[ids.index(item["id"])]
                pair = load_pair(args.cache, by_id[row["id"]], device)
                native = encode_reference(pipe, row["input_path"])
                feature_difference = float((pair["input"] - native).abs().max())
                pair["input"] = native.requires_grad_(item["update"] == 1)
                optimizer.zero_grad(set_to_none=True)
                generator = torch.Generator(device=device).manual_seed(item["noise_seed"])
                loss, observed = loss_with_reader(pipe, pair, fixed["positive"], generator, reader,
                    row["reference_text"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite control objective")
                loss.backward()
                if item["update"] == 1:
                    gradient = pair["input"].grad
                    if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
                        raise AssertionError("missing finite nonzero input gradient")
                    observed["input_gradient_norm"] = float(gradient.float().norm())
                if any(v.grad is not None for model in (pipe.vae, reader.model) for v in model.parameters()) or any(
                        v.grad is not None for v in pipe.transformer.parameters() if not v.requires_grad):
                    raise AssertionError("frozen weights received gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not bool(norm > 0):
                    raise AssertionError("zero control student gradient")
                optimizer.step()
                active += observed["recognition_loss"] is not None
                entry = dict(update=item["update"], id=item["id"], noise_seed=item["noise_seed"],
                    loss=float(loss.detach()), **observed, historical_input_feature_max_difference=feature_difference,
                    gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - tick)
                with (args.output / "updates.jsonl").open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                save_json(args.output / "status.json", dict(stage="training", update=item["update"], budget=32, pid=os.getpid()))
                if torch.cuda.max_memory_allocated() >= 22 * 1024**3:
                    raise RuntimeError("control allocated-memory limit exceeded")
                save_checkpoint(args.output / "checkpoint.pt", pipe, optimizer, step=item["update"],
                    generator=generator, sampler=sampler, metadata=metadata)
                print(json.dumps(entry), flush=True)
                del pair, native, loss
            optimization_seconds = time.perf_counter() - optimization_started
            changed = sum(not torch.equal(value, old[name]) for name, value in adapter_state(pipe.transformer).items())
            if not changed or not active or implementation() != code:
                raise AssertionError("unchanged implementation, changed adapters and recognition updates required")
            report = dict(stage="completed", updates=32, metadata=metadata, initial_replays=replays,
                initial_replay_seconds=replay_seconds, optimization_seconds=optimization_seconds,
                elapsed_seconds=time.perf_counter() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                checkpoint_sha256=sha(args.output / "checkpoint.pt"), changed_adapter_tensors=changed,
                recognition_updates=active, trainable_parameters=sum(v.numel() for v in parameters),
                deployed_transformer_parameters=sum(v.numel() for v in pipe.transformer.parameters()),
                codec_parameters=sum(v.numel() for v in pipe.vae.parameters()),
                training_only_reader_parameters=sum(v.numel() for v in reader.model.parameters()),
                writing_qualification=False, general_ilm_goal_complete=False)
            save_json(args.output / "report.json", report)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2), flush=True)
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None, reason=str(error)))
        raise


if __name__ == "__main__":
    main()
