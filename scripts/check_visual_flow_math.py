#!/usr/bin/env python3
"""Verify training algebra against pinned source and the actual scheduler.

Uses existing real answer images as numerical data, not a neural latent or
learned result. The source ASTs test interpolation/loss/gradient exactly. An
oracle velocity tests scheduler direction; no toy neural model is created.
"""
import argparse
import ast
import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
import types
import urllib.request

from ilm.image_edit.baseline import ROOT, save_json, sha

SOURCE = ROOT / "artifacts/external/visual_semantic_training_source/src/models/sd3_kontext/qwen2_5_vl_sd3_hf_dynamic_fusion.py"
SOURCE_SHA = "a9f5188fe9c3570ccb0fd32d89870add575b0845b99255c57d3c0388cafb2148"
SOURCE_URL = "https://raw.githubusercontent.com/deepgenteam/deepgen/fbaaaf4ae1c79450d6a34b30beb4f59840568c69/src/models/sd3_kontext/qwen2_5_vl_sd3_hf_dynamic_fusion.py"
SNAPSHOT = ROOT / "artifacts/external/visual_semantic_source/hub/models--deepgenteam--DeepGen-1.0-diffusers/snapshots/85c2ed257b9406d8531965c68e140dc7fdb81382"


def native_algebra(torch):
    if not SOURCE.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            body = response.read(300001)
        if len(body) > 300000 or hashlib.sha256(body).hexdigest() != SOURCE_SHA:
            raise ValueError("downloaded training source differs from pinned checksum")
        SOURCE.parent.mkdir(parents=True, exist_ok=True); SOURCE.write_bytes(body)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("pinned training source differs")
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2p5VLStableDiffusion3HF")
    loss = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "diff_loss")
    names = {"noisy_model_input", "target", "loss"}
    statements = [n for n in loss.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    if len(statements) != 4:
        raise ValueError("native loss statements changed")
    sigmas = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_sigmas")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[sigmas], type_ignores=[]), str(SOURCE), "exec"), scope)
    return compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), "exec"), scope["get_sigmas"]


def main(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("hide CUDA for numerical CPU verification")
    import numpy as np
    from PIL import Image
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from ilm.image_edit.visual_flow_learning import interpolate, sample_flow, velocity_loss
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / "artifacts/image_edit_learning/pages_v2/0160_target.png",
        ROOT / "artifacts/image_edit_learning/pages_v2/0162_target.png"]
    clean = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1 for p in inputs])
    native, get_sigmas = native_algebra(torch)
    fresh = lambda: FlowMatchEulerDiscreteScheduler.from_pretrained(str(SNAPSHOT / "scheduler"), local_files_only=True)
    scheduler = fresh()
    reports = []
    for dtype in (torch.float32, torch.bfloat16):
        data = clean.to(dtype)
        time_rng = torch.Generator(device="cpu").manual_seed(20269201)
        noise_rng = torch.Generator(device="cpu").manual_seed(20269202)
        sample = sample_flow(data, scheduler, time_generator=time_rng, noise_generator=noise_rng)
        repeat = sample_flow(data, scheduler,
            time_generator=torch.Generator(device="cpu").manual_seed(20269201),
            noise_generator=torch.Generator(device="cpu").manual_seed(20269202))
        # Recreate the exact sampled Gaussian for executing source expressions.
        noise = torch.randn(data.shape, dtype=dtype, generator=torch.Generator().manual_seed(20269202))
        oracle_sigmas = get_sigmas(types.SimpleNamespace(train_scheduler=scheduler, device=torch.device("cpu"), dtype=dtype), sample.timesteps)
        if not torch.equal(oracle_sigmas, sample.sigmas): raise AssertionError("source get_sigmas differs")
        # Two non-neural numerical probes: the actual target and a perturbed
        # velocity. Neither is described as a learned prediction.
        probes = []
        for offset in (0., .25):
            pred = (sample.velocity + offset).detach().requires_grad_(True)
            separate = pred.detach().clone().requires_grad_(True)
            scope = dict(sigmas=list(sample.sigmas), model_input=list(data), noise=list(noise),
                weighting=[torch.ones_like(s) for s in sample.sigmas], model_pred=list(separate))
            exec(native, {}, scope)
            actual = velocity_loss(pred, sample.velocity)
            noisy = torch.stack(scope["noisy_model_input"])
            target = torch.stack(scope["target"])
            actual.backward(); scope["loss"].backward()
            fields = dict(noisy_input=torch.equal(noisy, sample.noisy_latents),
                velocity_target=torch.equal(target, sample.velocity), scalar_loss=torch.equal(actual, scope["loss"]),
                velocity_gradient=torch.equal(pred.grad, separate.grad))
            if not all(fields.values()): raise AssertionError("native flow algebra or gradient differs")
            probes.append(dict(numeric_offset=offset, exact_fields=fields, loss=float(actual.detach())))
        if not torch.equal(sample.noisy_latents, repeat.noisy_latents) or not torch.equal(sample.timesteps, repeat.timesteps):
            raise AssertionError("explicit RNG replay differs")
        reports.append(dict(dtype=str(dtype), indices=sample.indices.tolist(), timesteps=sample.timesteps.tolist(),
            sigmas=sample.sigmas.flatten().float().tolist(), source_get_sigmas_exact=True, explicit_rng_replay_exact=True, probes=probes))
    scheduler = fresh(); scheduler.set_timesteps(50, device="cpu")
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(20269203))
    velocity = noise - clean
    state = (1-scheduler.sigmas[0]) * clean + scheduler.sigmas[0] * noise
    for t in scheduler.timesteps:
        state = scheduler.step(velocity, t, state, return_dict=False)[0]
    error = float((state-clean).abs().max())
    wrong = fresh(); wrong.set_timesteps(50, device="cpu")
    backwards = (1-wrong.sigmas[0]) * clean + wrong.sigmas[0] * noise
    for t in wrong.timesteps:
        backwards = wrong.step(-velocity, t, backwards, return_dict=False)[0]
    wrong_error = float((backwards-clean).abs().max())
    if error > 2e-5 or wrong_error < 1:
        raise AssertionError("native reverse sampler direction is inconsistent")
    try:
        sample_flow(clean, scheduler, time_generator=torch.Generator(), noise_generator=torch.Generator())
    except ValueError: rejects_mutated=True
    else: raise AssertionError("inference-mutated scheduler admitted to training")
    import ilm.image_edit.visual_flow_learning as implementation
    report = dict(source_url=SOURCE_URL, source_sha256=SOURCE_SHA, code_sha256=sha(__file__),
        training_module_sha256=sha(ROOT / "ilm/image_edit/visual_flow_learning.py"),
        scheduler_config_sha256=sha(SNAPSHOT / "scheduler/scheduler_config.json"),
        installed_scheduler_class_source_sha256=hashlib.sha256(inspect.getsource(type(scheduler)).encode()).hexdigest(),
        diffusers_version=version("diffusers"), actual_existing_answer_image_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        data_are_actual_RGB_pages_not_VAE_latents=True, native_source_AST_executed=True,
        native_algebra_dtype_checks=reports, native_reverse_oracle_steps=50,
        native_reverse_oracle_dtype="torch.float32", native_reverse_oracle_max_abs_error=error,
        deliberately_wrong_velocity_sign_max_abs_error=wrong_error, rejects_inference_mutated_training_schedule=rejects_mutated,
        training_denoiser_forward_helper_present=callable(implementation.denoising_loss),
        training_denoiser_forward_helper_exercised=False, neural_model_loaded=False, neural_forward_performed=False,
        cuda_initialized=torch.cuda.is_initialized(), optimizer_updates=0,
        language_learning_or_training_speed_established=False, general_ilm_goal_complete=False)
    save_json(output / "report.json", report)
    print(json.dumps({k:report[k] for k in ("native_reverse_oracle_max_abs_error", "deliberately_wrong_velocity_sign_max_abs_error",
        "rejects_inference_mutated_training_schedule", "neural_forward_performed", "optimizer_updates")}, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();main(args.output.resolve())
