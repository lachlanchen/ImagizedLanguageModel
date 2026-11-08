#!/usr/bin/env python3
"""Controlled paired-image flow training with an admitted frozen OCR loss.

Deployment uses the existing image-only infer_paired_image_cohort.py. This
training script, its labels and the OCR teacher are not inference dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, save_json, sha


def implementation():
    names = ("ilm/image_edit/paired.py", "ilm/image_edit/recognition.py",
        "scripts/train_recognition_guided_writer.py", "scripts/calibrate_writing_reader.py",
        "references/recognition_guided_writing_protocol.md")
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        files={name: sha(ROOT / name) for name in names})


def loss_with_reader(pipe, pair, fixed, generator, teacher, text, *, capture=False):
    import torch
    from torch.utils.checkpoint import checkpoint
    from ilm.image_edit.recognition import decode_features
    x, y = pair["input"], pair["target"]
    noise = torch.randn(y.shape, device=y.device, dtype=torch.float32, generator=generator)
    sigma = torch.rand((len(y),), device=y.device, generator=generator)
    mixed = (1 - sigma[:, None, None, None]) * y.float() + sigma[:, None, None, None] * noise
    packed_y = pipe._pack_latents(mixed.to(y.dtype))
    packed_x = pipe._pack_latents(x)
    target_ids = pipe._prepare_latent_ids(y).to(y.device)
    ids = torch.cat((target_ids, pipe._prepare_image_ids([x]).to(y.device)), dim=1)
    c = fixed.to(y.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pipe.transformer(hidden_states=torch.cat((packed_y, packed_x), dim=1),
            encoder_hidden_states=c, timestep=sigma,
            img_ids=ids, txt_ids=pipe._prepare_text_ids(c).to(y.device), guidance=None,
            return_dict=False)[0][:, :packed_y.shape[1]]
    velocity = pipe._pack_latents(noise - y.float())
    flow = (prediction.float() - velocity).square().mean()
    recognition = None
    if teacher is not None and float(sigma[0]) <= .5:
        clean = pipe._pack_latents(mixed) - sigma[:, None, None] * prediction.float()
        features = pipe._unpack_latents_with_ids(clean, target_ids,
            height=y.shape[-2], width=y.shape[-1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rgb = checkpoint(lambda z: decode_features(pipe, z), features, use_reentrant=False)
            recognition = teacher.loss(rgb, text)
    loss = flow if recognition is None else flow + .05 * recognition
    observation = dict(sigma=float(sigma[0]), target_patches=packed_y.shape[1],
        input_patches=packed_x.shape[1], flow_loss=float(flow.detach()),
        recognition_loss=float(recognition.detach()) if recognition is not None else None)
    if capture and recognition is not None:
        gradient = torch.autograd.grad(recognition, prediction, retain_graph=True)[0]
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
            raise AssertionError("recognition loss did not reach the native velocity prediction")
        observation["recognition_velocity_gradient_norm"] = float(gradient.float().norm())
    return loss, observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "condition", "teacher", "admission", "manifest", "cache", "output", "control"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stage", choices=("preflight", "train"), required=True)
    p.add_argument("--checkpoint", type=Path, help="resume only the same registered run")
    args = p.parse_args()
    import torch
    from ilm.image_edit.paired import (adapter_state, add_adapters, flow_loss, load_pair,
        load_pipeline, read_manifest, restore_adapters, save_checkpoint, verify_source)
    from ilm.image_edit.recognition import RecognitionTeacher, verify_teacher
    from scripts.calibrate_writing_reader import __file__ as calibration_file
    source = verify_source(args.snapshot)
    teacher_source = verify_teacher(args.teacher)
    admission = json.loads(args.admission.read_text())
    if (not admission["teacher_admitted"] or admission["source"] != teacher_source
            or admission["manifest_sha256"] != sha(args.manifest)
            or admission["implementation"]["scripts/calibrate_writing_reader.py"] != sha(calibration_file)
            or any(sha(ROOT / key) != digest for key, digest in admission["implementation"].items())):
        raise ValueError("unchanged complete teacher admission must precede student optimization")
    all_rows = read_manifest(args.manifest)
    training = [r for r in all_rows if r["task"] == "copy" and r["split"] == "train"][:32]
    labels = {r["id"]: r["reference_text"] for r in training}
    cached = json.loads((args.cache / "manifest.json").read_text())
    records = [r for r in cached["records"] if r["id"] in labels]
    if [r["id"] for r in records] != [r["id"] for r in training] or len(records) != 32:
        raise ValueError("same first 32 copying examples required")
    originals = {r["id"]: r for r in training}
    for r in records:
        original = originals[r["id"]]
        if any(r[k + "_sha256"] != original[k + "_sha256"] for k in ("input", "target")):
            raise ValueError("numeric training cache differs from original page pair")
    fixed = condition_cache(args.condition)
    control = [json.loads(line) for line in args.control.read_text().splitlines() if line]
    if len(control) != 512:
        raise ValueError("full original 512-update comparison journal required")
    code = implementation()
    updates = 2 if args.stage == "preflight" else 512
    metadata = dict(source=source, implementation=code, rank=16, seed=20268321,
        updates=updates, learning_rate=1e-4, weight_decay=.01, accumulation=1,
        condition_sha256=sha(args.condition), training_ids=list(labels),
        training_cache_manifest_sha256=sha(args.cache / "manifest.json"),
        training_manifest_sha256=sha(args.manifest), teacher_source=teacher_source,
        admission_sha256=sha(args.admission), control_journal_sha256=sha(args.control),
        objective="native uniform-sigma velocity MSE + .05 OCR CE at sigma <= .5",
        teacher_at_inference=False, question_specific_text_condition=False,
        external_pretraining=True, language_quality_qualified=False, general_ilm_goal_complete=False)
    args.output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    try:
        with gpu_zero() as device:
            torch.manual_seed(20268321); torch.cuda.manual_seed_all(20268321)
            pipe = load_pipeline(args.snapshot, device)
            add_adapters(pipe.transformer, 16)
            teacher = RecognitionTeacher(args.teacher, device)
            teacher.model.train()
            pipe.transformer.enable_gradient_checkpointing(); pipe.transformer.train()
            parameters = [p for p in pipe.transformer.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
            generator = torch.Generator(device=device).manual_seed(20268322)
            sampler = torch.Generator().manual_seed(20268323)
            initial = adapter_state(pipe.transformer)
            step = 0
            if args.checkpoint:
                saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                if saved["metadata"] != metadata or not 0 < saved["step"] < updates:
                    raise ValueError("resume requires the exact registered unfinished budget")
                restore_adapters(pipe.transformer, saved["adapters"])
                optimizer.load_state_dict(saved["optimizer"])
                generator.set_state(saved["generator_state"]); sampler.set_state(saved["sampler_state"])
                torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
                step = saved["step"]
            equivalence = None
            if args.stage == "preflight":
                pair = load_pair(args.cache, records[0], device)
                rng = torch.Generator(device=device).manual_seed(20268331)
                state = rng.get_state()
                with torch.no_grad():
                    original, _ = flow_loss(pipe, pair, fixed["positive"], rng)
                    rng.set_state(state)
                    actual, _ = loss_with_reader(pipe, pair, fixed["positive"], rng, None, "")
                if not torch.equal(original, actual):
                    raise AssertionError("native flow term changed before adding recognition")
                equivalence = dict(native_flow_bit_exact=True, loss=float(actual))
                del pair, original, actual
            recognition_updates = 0
            for step in range(step, updates):
                optimizer.zero_grad(set_to_none=True)
                index = int(torch.randint(len(records), (1,), generator=sampler))
                record = records[index]; pair = load_pair(args.cache, record, device)
                if args.stage == "preflight":
                    pair["input"].requires_grad_(True)
                loss, observation = loss_with_reader(pipe, pair, fixed["positive"], generator,
                    teacher, labels[record["id"]], capture=args.stage == "preflight")
                expected = control[step]
                if expected["ids"] != [record["id"]] or expected["observations"][0]["sigma"] != observation["sigma"]:
                    raise AssertionError("example/noise-time trajectory differs from original control")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite writing loss")
                loss.backward()
                if args.stage == "preflight":
                    gradient = pair["input"].grad
                    if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
                        raise AssertionError("missing finite gradient to image reference")
                    observation["input_gradient_norm"] = float(gradient.norm())
                if any(p.grad is not None for model in (teacher.model, pipe.vae)
                        for p in model.parameters()):
                    raise AssertionError("frozen training reader or codec received parameter gradients")
                if any(p.grad is not None for p in pipe.transformer.parameters() if not p.requires_grad):
                    raise AssertionError("frozen source body received parameter gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not bool(norm > 0):
                    raise AssertionError("zero student gradient")
                optimizer.step()
                recognition_updates += observation["recognition_loss"] is not None
                row = dict(update=step + 1, loss=float(loss.detach()), ids=[record["id"]],
                    observations=[observation], gradient_norm=float(norm),
                    elapsed_seconds=time.perf_counter() - tick)
                with (args.output / "updates.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                save_json(args.output / "status.json", dict(stage="training", pid=os.getpid(), update=step + 1, budget=updates))
                if torch.cuda.max_memory_allocated() >= 22 * 1024**3:
                    raise RuntimeError("registered 22 GiB allocation bound exceeded")
                if (step + 1) % 32 == 0 or step + 1 == updates:
                    save_checkpoint(args.output / "checkpoint.pt", pipe, optimizer,
                        step=step + 1, generator=generator, sampler=sampler, metadata=metadata)
                if step < 2 or (step + 1) % 16 == 0:
                    print(json.dumps(row), flush=True)
                del pair, loss
            changed = sum(not torch.equal(v, initial[k]) for k, v in adapter_state(pipe.transformer).items())
            if changed == 0 or recognition_updates == 0:
                raise AssertionError("no changed student weights or recognition-supervised update")
            if implementation() != code:
                raise ValueError("learning implementation changed while live")
            report = dict(stage=args.stage, updates=updates, recognition_updates=recognition_updates,
                changed_adapter_tensors=changed, metadata=metadata, native_flow_equivalence=equivalence,
                checkpoint_sha256=sha(args.output / "checkpoint.pt"),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                elapsed_seconds=time.perf_counter() - tick,
                trainable_parameters=sum(p.numel() for p in parameters),
                deployed_transformer_parameters=sum(p.numel() for p in pipe.transformer.parameters()),
                codec_parameters=sum(p.numel() for p in pipe.vae.parameters()),
                training_only_teacher_parameters=sum(p.numel() for p in teacher.model.parameters()),
                quality_qualification=False, general_ilm_goal_complete=False)
            save_json(args.output / "report.json", report)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None, reason=str(error)))
        raise


if __name__ == "__main__":
    main()
