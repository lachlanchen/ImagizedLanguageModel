#!/usr/bin/env python3
"""Train and independently verify clean-code generation of a fixed real image."""

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.clean_writing_flow import CleanWritingConfig, CleanWritingFlow
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from scripts.cache_semantic_images_v72 import ROOT, CODEC, atomic_json, committed_identity, sha
from scripts.probe_flow_precision_v73 import preview
from scripts.probe_glyph_denoising_v73 import source_case, scores, quality_gates, checkpoint
from scripts.train_semantic_ilm_v72 import setup


SOURCES = ("scripts/probe_clean_endpoint_v74.py", "ilm/visual_lm/clean_writing_flow.py",
    "ilm/visual_lm/full_rank_writing_flow.py", "ilm/visual_lm/semantic_writing_model.py",
    "ilm/visual_lm/codec_block_writer.py", "scripts/probe_glyph_denoising_v73.py",
    "scripts/probe_flow_precision_v73.py", "references/clean_endpoint_v74_protocol.md")


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"report.json").exists():
        raise ValueError("completed controls are immutable")
    reference, target_codes, selected = source_case(args.run)
    config = CleanWritingConfig()
    identity = dict(**committed_identity(SOURCES), **selected, model_config=asdict(config),
        updates=4000, batch=64, evaluation_count=256, model_seed=20267400,
        training_seed=20267401, evaluation_seed=20267402, condition="constant zero 512-vector",
        language_quality_claim=False)
    if (args.out/"identity.json").exists():
        prior = json.loads((args.out/"identity.json").read_text())
        if {k:v for k,v in prior.items() if k != "source_commit"} != {
                k:v for k,v in identity.items() if k != "source_commit"}:
            raise ValueError("control identity changed")
        identity = prior
    else:
        atomic_json(args.out/"identity.json", identity)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup()
        torch.manual_seed(identity["model_seed"])
        torch.cuda.manual_seed_all(identity["model_seed"])
        flow = CleanWritingFlow(config).to(device).train()
        writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, device))
        if float((writer.encode_targets(reference.to(device)).cpu()-target_codes).abs().max()) > 1e-4:
            raise ValueError("original pixels and encoded target differ")
        with torch.no_grad():
            soft = writer.predict(target_codes.to(device)).cpu()
        supplied = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
        if scores(supplied, reference, supplied)["minimum_foreground_f1"] < .99:
            raise ValueError("fixed selected image cannot be reconstructed")
        optimizer = torch.optim.AdamW(flow.parameters(), lr=5e-4, betas=(.9, .95),
            weight_decay=.01, foreach=False)
        generator = torch.Generator(device=device).manual_seed(identity["training_seed"])
        latest, step, accumulated = args.out/"checkpoint_latest.pt", 0, 0.
        if latest.exists():
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            if saved["identity"] != identity:
                raise ValueError("resume identity differs")
            flow.load_state_dict(saved["flow"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            step, accumulated = saved["update"], saved["active_elapsed_seconds"]
            del saved
        target = target_codes.to(device).flatten(1).expand(64, -1)
        condition = torch.zeros((64, 512), device=device)
        begun = time.perf_counter()
        with (args.out/"metrics.jsonl").open("a") as log:
            while step < 4000:
                lr = 5e-4*min((step+1)/100, 1.)*(.1+.9*.5*(1+np.cos(np.pi*step/4000)))
                for group in optimizer.param_groups:
                    group["lr"] = float(lr)
                noise = torch.randn((64, 1536), device=device, generator=generator)
                time_value = torch.rand((64,), device=device, generator=generator)
                noisy = (1-time_value[:, None])*noise+time_value[:, None]*target
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    clean = flow(noisy, time_value, condition)
                    error = (clean-target).square().mean(-1)
                    loss = (error/(1-time_value).clamp_min(.05).square()).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite clean prediction loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step == 1 or step % 100 == 0:
                    row = dict(update=step, weighted_loss=float(loss.detach()),
                        clean_mse=float(error.mean().detach()), gradient_norm=float(norm),
                        elapsed_seconds=accumulated+time.perf_counter()-begun)
                    log.write(json.dumps(row)+"\n")
                    log.flush()
                    atomic_json(args.out/"status.json", dict(stage="training_control", pid=os.getpid(), **row))
                    print(json.dumps(row), flush=True)
                if step % 500 == 0:
                    checkpoint(latest, flow, optimizer, generator, step, identity,
                        accumulated+time.perf_counter()-begun)
        flow.eval()
        condition = torch.zeros((256, 512), device=device)
        evaluation_generator = torch.Generator(device=device).manual_seed(identity["evaluation_seed"])
        with torch.inference_mode():
            noise, endpoint = flow.sample_endpoint(condition, generator=evaluation_generator, steps=32)
            coordinates = F.layer_norm(endpoint.reshape(-1, 768), (768,),
                eps=writer.codec.config.layer_norm_epsilon).reshape(256, 2, 768)
            soft = writer.predict(coordinates)
            pixels = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
        raw = dict(initial_noise=noise.cpu(), endpoints=endpoint.cpu(), coordinates=coordinates.cpu(),
            pixels=pixels.cpu(), reference=reference, supplied_render=supplied, target_codes=target_codes)
        torch.save(raw, args.out/"evaluation.pt")
        Image.fromarray(preview(raw["pixels"])).save(args.out/"samples.png")
        Image.fromarray(((reference[0, 0]+1)*127.5).to(torch.uint8).numpy()).save(args.out/"reference.png")
        measured = scores(raw["pixels"], reference, supplied)
        gates = quality_gates(measured)
        atomic_json(args.out/"report.json", dict(identity=identity, parameters=sum(p.numel() for p in flow.parameters()),
            measurements=measured, gates=gates, clean_glyph_control_pass=all(gates.values()),
            checkpoint_sha256=sha(latest), evaluation_sha256=sha(args.out/"evaluation.pt"),
            png_sha256=sha(args.out/"samples.png"), elapsed_seconds=accumulated+time.perf_counter()-begun,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False, general_ilm_goal_complete=False))
        atomic_json(args.out/"status.json", dict(stage="control_complete_awaiting_verification", pid=None))
        print(json.dumps(dict(measurements=measured, gates=gates)), flush=True)


def verify(args):
    from scripts.verify_codec_language_v68 import physical_scores
    report = json.loads((args.out/"report.json").read_text())
    for name, digest in report["identity"]["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("source implementation changed")
    reference, codes, selected = source_case(args.run)
    if any(report["identity"][k] != v for k, v in selected.items()):
        raise ValueError("selected source changed")
    if (sha(args.out/"evaluation.pt") != report["evaluation_sha256"]
            or sha(args.out/"checkpoint_latest.pt") != report["checkpoint_sha256"]
            or sha(args.out/"samples.png") != report["png_sha256"]):
        raise ValueError("retained output changed")
    raw = torch.load(args.out/"evaluation.pt", map_location="cpu", weights_only=True)
    pixels = raw["pixels"]
    if (pixels.shape != (256, 1, 32, 64) or not bool(((pixels == -1) | (pixels == 1)).all())
            or not torch.equal(raw["reference"], reference) or not torch.equal(raw["target_codes"], codes)):
        raise ValueError("invalid original/generated images")
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cpu")))
    decoded = writer.predict(raw["coordinates"]).detach()
    replay = torch.where(decoded < 0, -torch.ones_like(decoded), torch.ones_like(decoded))
    if not torch.equal(replay, pixels):
        raise ValueError("independent CPU fixed decoder replay differs")
    ref = ((reference[0, 0]+1)*127.5).to(torch.uint8)
    physical = [physical_scores(x[0].numpy() < 0, [ref], True) for x in pixels]
    measured = dict(mean_foreground_f1=float(np.mean([x["foreground_f1"] for x in physical])),
        minimum_foreground_f1=float(min(x["foreground_f1"] for x in physical)),
        original_exact_fraction=float(np.mean([x["exact"] for x in physical])),
        supplied_render_exact_fraction=float(np.mean([
            np.array_equal(x.numpy(), raw["supplied_render"][0].numpy()) for x in pixels])))
    if any(abs(v-report["measurements"][k]) > 1e-7 for k, v in measured.items()):
        raise ValueError("independent scores differ")
    gates = quality_gates(measured)
    if gates != report["gates"] or all(gates.values()) != report["clean_glyph_control_pass"]:
        raise ValueError("independent quality decision differs")
    if not np.array_equal(np.asarray(Image.open(args.out/"samples.png")), preview(pixels)):
        raise ValueError("preview pixels changed")
    atomic_json(args.out/"verification.json", dict(report_sha256=sha(args.out/"report.json"),
        measurements=measured, gates=gates, clean_glyph_control_pass=all(gates.values()),
        independent_cpu_physical_scoring=True, cpu_fixed_decoder_replayed=True,
        full_neural_sampler_replayed=False, language_quality_claim=False, general_ilm_goal_complete=False))
    print(json.dumps(dict(measurements=measured, gates=gates), indent=2))


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
        run(arguments)
