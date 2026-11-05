#!/usr/bin/env python3
"""Matched narrow/full-dimensional flow control on one real Chinese image.

This learns a single fixed image distribution. It is an architectural
denoising control, not a prompt-conditioned language experiment.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.full_rank_writing_flow import FullRankWritingFlow, full_rank_configuration
from ilm.visual_lm.semantic_writing_model import JointWritingFlow, SemanticWritingConfig
from scripts.cache_semantic_images_v72 import (
    ROOT, TRAIN, TRAIN_SHA, CODEC, atomic_json, committed_identity, pixels_sha, sha,
)
from scripts.train_semantic_ilm_v72 import setup


UPDATES, BATCH, EVALUATIONS = 4000, 64, 64
SOURCES = ("scripts/probe_glyph_denoising_v73.py", "ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py", "ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/complete_writing_data.py", "scripts/cache_semantic_images_v72.py",
    "scripts/train_semantic_ilm_v72.py", "references/full_rank_flow_v73_control_protocol.md")


def source_case(run):
    training = json.loads((run/"training/training_report.json").read_text())
    cache = run/"cache"
    inventory = json.loads((cache/"report.json").read_text())
    recipe = json.loads((cache/"recipe.json").read_text())
    if (sha(cache/"report.json") != training["identity"]["cache_report_sha256"]
            or not inventory["complete"] or inventory["completed"] != 12976
            or sha(cache/"recipe.json") != inventory["recipe_sha256"]):
        raise ValueError("completed authenticated source cache required")
    position = next(i for i, name in enumerate(recipe["names"]) if name.startswith("cmrc/"))
    relative = f"records/{position:05}.pt"
    path = cache/relative
    if sha(path) != inventory["files_sha256"][relative]:
        raise ValueError("selected cached record changed")
    row = torch.load(path, map_location="cpu", weights_only=True)
    data = CompleteWritingImageData(TRAIN, purpose="train", expected_report_sha256=TRAIN_SHA)
    pixels, boundary = data.get(row["name"])
    target = pixels[..., boundary*64:-128]
    if (row["name"] != recipe["names"][position] or pixels_sha(target) != row["target_sha256"]
            or row["source_sha256"] != data.report["files_sha256"][row["name"]]):
        raise ValueError("original target and cached codes are not paired")
    reference = target[..., :64].clone()
    if not bool((reference < 0).any()):
        raise ValueError("predeclared first Chinese block is blank; do not select another")
    return reference, row["codes"][:1].clone(), dict(cache_record=relative,
        cache_record_sha256=sha(path), source=row["name"], source_sha256=row["source_sha256"],
        reference_sha256=pixels_sha(reference), cache_report_sha256=sha(cache/"report.json"))


def scores(images, reference, supplied):
    ink, truth = images < 0, reference < 0
    overlap = (ink & truth).flatten(1).sum(1)
    f1 = (2*overlap/(ink.flatten(1).sum(1)+truth.flatten(1).sum(1)).clamp_min(1)).double()
    exact = (images == reference).flatten(1).all(1)
    codec_exact = (images == supplied).flatten(1).all(1)
    return dict(mean_foreground_f1=float(f1.mean()), minimum_foreground_f1=float(f1.min()),
        original_exact_fraction=float(exact.double().mean()),
        supplied_render_exact_fraction=float(codec_exact.double().mean()))


def quality_gates(measurements):
    return dict(mean_original_f1=measurements["mean_foreground_f1"] >= .995,
        every_original_f1=measurements["minimum_foreground_f1"] >= .99,
        repeatable_codec_image=measurements["supplied_render_exact_fraction"] >= .9)


def save_png(path, images):
    value = ((images+1)*127.5).to(torch.uint8)
    if len(value) == 1:
        array = value[0, 0].numpy()
    else:
        array = value[:, 0].reshape(8, 8, 32, 64).permute(0, 2, 1, 3).reshape(256, 512).numpy()
    Image.fromarray(array).save(path)


@torch.no_grad()
def generate(flow, writer, device):
    generator = torch.Generator(device=device).manual_seed(20267302)
    noise = torch.randn((EVALUATIONS, 1536), device=device, generator=generator)
    value = noise.clone()
    condition = torch.zeros((EVALUATIONS, 512), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(32):
            velocity = flow(value, torch.full((EVALUATIONS,), i/32, device=device), condition)
            midpoint = value+velocity/64
            value = value+flow(midpoint, torch.full((EVALUATIONS,), (i+.5)/32, device=device), condition)/32
    codes = F.layer_norm(value.reshape(-1, 768), (768,),
        eps=writer.codec.config.layer_norm_epsilon).reshape(EVALUATIONS, 2, 768)
    soft = writer.predict(codes)
    pixels = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
    return dict(initial_noise=noise.cpu(), endpoints=value.cpu(), coordinates=codes.cpu(), pixels=pixels.cpu())


def checkpoint(path, flow, optimizer, generator, step, identity, elapsed):
    temporary = path.with_suffix(".partial")
    torch.save(dict(flow=flow.state_dict(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
        update=step, identity=identity, active_elapsed_seconds=elapsed), temporary)
    temporary.replace(path)


def train_arm(name, out, target_codes, reference, supplied, writer, identity, device):
    out.mkdir(exist_ok=True)
    torch.manual_seed(20267300)
    torch.cuda.manual_seed_all(20267300)
    if name == "narrow":
        config = SemanticWritingConfig()
        flow = JointWritingFlow(config)
        torch.nn.init.eye_(flow.noisy.weight)
        torch.nn.init.zeros_(flow.noisy.bias)
    else:
        config = full_rank_configuration()
        flow = FullRankWritingFlow(config)
    flow = flow.to(device).train()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=5e-4, betas=(.9, .95),
        weight_decay=.01, foreach=False)
    generator = torch.Generator(device=device).manual_seed(20267301)
    latest, step, accumulated = out/"checkpoint_latest.pt", 0, 0.
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("control resume identity changed")
        flow.load_state_dict(saved["flow"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        step, accumulated = saved["update"], saved["active_elapsed_seconds"]
        del saved
    begun = time.perf_counter()
    target = target_codes.to(device).flatten(1).expand(BATCH, -1)
    condition = torch.zeros((BATCH, 512), device=device)
    with (out/"metrics.jsonl").open("a") as log:
        while step < UPDATES:
            lr = 5e-4*min((step+1)/100, 1.)*(.1+.9*.5*(1+math.cos(math.pi*step/UPDATES)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            noise = torch.randn((BATCH, 1536), device=device, generator=generator)
            time_value = torch.rand((BATCH,), device=device, generator=generator)
            noisy = (1-time_value[:, None])*noise+time_value[:, None]*target
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                velocity = flow(noisy, time_value, condition)
                loss = (velocity-(target-noise)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite denoising control loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                row = dict(arm=name, update=step, velocity_loss=float(loss.detach()),
                    gradient_norm=float(norm), lr=lr, elapsed_seconds=accumulated+time.perf_counter()-begun)
                log.write(json.dumps(row)+"\n")
                log.flush()
                atomic_json(out.parent/"status.json", dict(stage="training_control", pid=os.getpid(), **row))
                print(json.dumps(row), flush=True)
            if step % 500 == 0:
                checkpoint(latest, flow, optimizer, generator, step, identity,
                    accumulated+time.perf_counter()-begun)
    flow.eval()
    result = generate(flow, writer, device)
    result.update(reference=reference, supplied_render=supplied, target_codes=target_codes)
    torch.save(result, out/"evaluation.pt")
    save_png(out/"samples.png", result["pixels"])
    measurements = scores(result["pixels"], reference, supplied)
    gates = quality_gates(measurements)
    report = dict(identity=identity, arm=name, flow_width=config.flow_width,
        parameters=sum(p.numel() for p in flow.parameters()), updates=step, evaluation_seeds=EVALUATIONS,
        measurements=measurements, gates=gates, clean_glyph_control_pass=all(gates.values()),
        checkpoint_sha256=sha(latest), evaluation_sha256=sha(out/"evaluation.pt"),
        sample_png_sha256=sha(out/"samples.png"), language_quality_claim=False,
        elapsed_seconds=accumulated+time.perf_counter()-begun,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json(out/"report.json", report)
    del optimizer, flow
    torch.cuda.empty_cache()
    return report


def control(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"report.json").exists():
        raise ValueError("completed controls are immutable; verify or supply a fresh directory")
    reference, target_codes, selected = source_case(args.run)
    identity = dict(**committed_identity(SOURCES), **selected, updates=UPDATES, batch=BATCH,
        evaluation_seeds=EVALUATIONS, model_seed=20267300, training_seed=20267301,
        evaluation_seed=20267302, condition="constant zero 512-vector", language_quality_claim=False)
    identity_path = args.out/"identity.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text())
        if {k:v for k,v in prior.items() if k != "source_commit"} != {
                k:v for k,v in identity.items() if k != "source_commit"}:
            raise ValueError("existing control identity changed")
        identity = prior
    else:
        atomic_json(identity_path, identity)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup()
        writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, device))
        encoded = writer.encode_targets(reference.to(device)).cpu()
        encoding_difference = float((encoded-target_codes).abs().max())
        if encoding_difference > 1e-4:
            raise ValueError("cached code differs from original image encoding")
        soft = writer.predict(target_codes.to(device)).cpu()
        supplied = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
        supplied_scores = scores(supplied, reference, supplied)
        if supplied_scores["minimum_foreground_f1"] < .99:
            raise ValueError("fixed selected block fails the supplied-code control; do not change selection")
        save_png(args.out/"reference.png", reference)
        save_png(args.out/"supplied_code.png", supplied)
        reports = {}
        for name in ("narrow", "full"):
            path = args.out/name/"report.json"
            if path.exists():
                report = json.loads(path.read_text())
                if (report["identity"] != identity
                        or sha(args.out/name/"checkpoint_latest.pt") != report["checkpoint_sha256"]
                        or sha(args.out/name/"evaluation.pt") != report["evaluation_sha256"]):
                    raise ValueError("completed prerequisite control changed")
                reports[name] = report
            else:
                reports[name] = train_arm(name, args.out/name, target_codes, reference,
                    supplied, writer, identity, device)
        atomic_json(args.out/"report.json", dict(identity=identity, supplied_code=supplied_scores,
            cached_code_maximum_difference=encoding_difference, arms=reports,
            clean_glyph_control_pass=reports["full"]["clean_glyph_control_pass"],
            language_quality_claim=False, general_ilm_goal_complete=False))
        atomic_json(args.out/"status.json", dict(stage="control_complete_awaiting_verification", pid=None))


def verify(args):
    from scripts.verify_codec_language_v68 import physical_scores
    report = json.loads((args.out/"report.json").read_text())
    reference, target_codes, selected = source_case(args.run)
    for name, digest in report["identity"]["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("control implementation changed")
    for key, value in selected.items():
        if report["identity"][key] != value:
            raise ValueError("selected source changed")
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cpu")))
    results = {}
    for name, arm in report["arms"].items():
        folder = args.out/name
        if sha(folder/"evaluation.pt") != arm["evaluation_sha256"] or sha(folder/"checkpoint_latest.pt") != arm["checkpoint_sha256"]:
            raise ValueError("retained control changed")
        saved = torch.load(folder/"evaluation.pt", map_location="cpu", weights_only=True)
        if not torch.equal(saved["reference"], reference) or not torch.equal(saved["target_codes"], target_codes):
            raise ValueError("physical reference/codes changed")
        pixels = saved["pixels"]
        if pixels.shape != (64, 1, 32, 64) or not bool(((pixels == -1) | (pixels == 1)).all()):
            raise ValueError("invalid physical samples")
        predicted = writer.predict(saved["coordinates"]).detach()
        replay = torch.where(predicted < 0, -torch.ones_like(predicted), torch.ones_like(predicted))
        bit_mismatches = int((replay != pixels).sum())
        if bit_mismatches:
            raise ValueError("independent CPU decoder pixel replay differs")
        ref = ((reference[0, 0]+1)*127.5).to(torch.uint8)
        independent = [physical_scores(image[0].numpy() < 0, [ref], True) for image in pixels]
        measured = dict(mean_foreground_f1=float(np.mean([v["foreground_f1"] for v in independent])),
            minimum_foreground_f1=float(min(v["foreground_f1"] for v in independent)),
            original_exact_fraction=float(np.mean([v["exact"] for v in independent])),
            supplied_render_exact_fraction=float(np.mean([
                np.array_equal(x.numpy(), saved["supplied_render"][0].numpy()) for x in pixels])))
        for key, value in measured.items():
            if abs(value-arm["measurements"][key]) > 1e-7:
                raise ValueError("independent physical score differs: "+key)
        gates = quality_gates(measured)
        if gates != arm["gates"]:
            raise ValueError("quality gate changed")
        # Authenticate that the preview is only a rearrangement of raw pixels.
        expected = ((pixels+1)*127.5).to(torch.uint8)[:, 0].reshape(8, 8, 32, 64).permute(
            0, 2, 1, 3).reshape(256, 512).numpy()
        if not np.array_equal(np.asarray(Image.open(folder/"samples.png")), expected):
            raise ValueError("preview differs from retained output pixels")
        results[name] = dict(measurements=measured, gates=gates, clean_glyph_control_pass=all(gates.values()),
            independent_decoder_bit_mismatches=bit_mismatches)
    atomic_json(args.out/"verification.json", dict(report_sha256=sha(args.out/"report.json"),
        arms=results, independent_physical_scoring=True, cpu_decoder_replayed=True,
        full_neural_sampling_replayed=False, language_quality_claim=False, general_ilm_goal_complete=False))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"artifacts/semantic_ilm_v72_20261105")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        torch.set_num_threads(4)
        verify(arguments)
    else:
        control(arguments)
