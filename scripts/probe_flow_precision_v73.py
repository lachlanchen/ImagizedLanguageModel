#!/usr/bin/env python3
"""Post-hoc integration diagnosis followed by a fresh-noise confirmation."""

import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.full_rank_writing_flow import FullRankWritingFlow
from scripts.cache_semantic_images_v72 import ROOT, CODEC, atomic_json, committed_identity, sha
from scripts.probe_glyph_denoising_v73 import source_case, scores, quality_gates
from scripts.train_semantic_ilm_v72 import setup


SETTINGS = (("bf16", 32), ("fp32", 32), ("bf16", 64), ("fp32", 64))
SOURCES = ("scripts/probe_flow_precision_v73.py", "scripts/probe_glyph_denoising_v73.py",
    "ilm/visual_lm/full_rank_writing_flow.py", "ilm/visual_lm/semantic_writing_model.py",
    "ilm/visual_lm/codec_block_writer.py", "references/flow_precision_v73_protocol.md")


@torch.inference_mode()
def sample(flow, writer, device, *, precision, steps, count, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((count, 1536), device=device, generator=generator)
    value = noise.clone()
    condition = torch.zeros((count, 512), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        for i in range(steps):
            velocity = flow(value, torch.full((count,), i/steps, device=device), condition)
            midpoint = value+velocity/(2*steps)
            value = value+flow(midpoint, torch.full((count,), (i+.5)/steps, device=device), condition)/steps
    coordinates = F.layer_norm(value.reshape(-1, 768), (768,),
        eps=writer.codec.config.layer_norm_epsilon).reshape(count, 2, 768)
    soft = writer.predict(coordinates)
    pixels = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
    return dict(initial_noise=noise.cpu(), endpoints=value.cpu(),
        coordinates=coordinates.cpu(), pixels=pixels.cpu())


def preview(pixels):
    count = len(pixels)
    return ((pixels+1)*127.5).to(torch.uint8)[:, 0].reshape(count//8, 8, 32, 64).permute(
        0, 2, 1, 3).reshape((count//8)*32, 512).numpy()


def run(args):
    if args.out.exists():
        raise ValueError("retain earlier sampling diagnoses; use a fresh directory")
    primary = json.loads((args.control/"report.json").read_text())
    verified = json.loads((args.control/"verification.json").read_text())
    if verified["report_sha256"] != sha(args.control/"report.json"):
        raise ValueError("source physical verification changed")
    for name, digest in primary["identity"]["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("source control changed")
    checkpoint = args.control/"full/checkpoint_latest.pt"
    if sha(checkpoint) != primary["arms"]["full"]["checkpoint_sha256"]:
        raise ValueError("trained flow changed")
    reference, _, _ = source_case(args.run)
    if sha(args.control/"full/evaluation.pt") != primary["arms"]["full"]["evaluation_sha256"]:
        raise ValueError("source physical output changed")
    original = torch.load(args.control/"full/evaluation.pt", map_location="cpu", weights_only=True)
    if not torch.equal(original["reference"], reference):
        raise ValueError("source physical reference changed")
    identity = dict(**committed_identity(SOURCES),
        source_control_report_sha256=sha(args.control/"report.json"),
        source_control_verification_sha256=sha(args.control/"verification.json"),
        checkpoint_sha256=sha(checkpoint), calibration_seed=20267302, calibration_count=64,
        confirmation_seed=20267303, confirmation_count=128,
        settings=[dict(precision=p, steps=s) for p, s in SETTINGS],
        selection="first passing calibration setting in fixed order; one fresh-noise confirmation only",
        original_control_remains_failed=not primary["clean_glyph_control_pass"],
        language_quality_claim=False)
    args.out.mkdir(parents=True)
    atomic_json(args.out/"identity.json", identity)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup()
        flow = FullRankWritingFlow().to(device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        flow.load_state_dict(state["flow"], strict=True)
        del state
        writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, device))
        results, selected = [], None

        def evaluate(precision, steps, count, seed, name):
            torch.cuda.synchronize()
            begun = time.perf_counter()
            result = sample(flow, writer, device, precision=precision, steps=steps, count=count, seed=seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begun
            torch.save(result, args.out/(name+".pt"))
            Image.fromarray(preview(result["pixels"])).save(args.out/(name+".png"))
            measured = scores(result["pixels"], reference, original["supplied_render"])
            gates = quality_gates(measured)
            value = dict(name=name, precision=precision, steps=steps, count=count, seed=seed,
                measurements=measured, gates=gates, clean_glyph_control_pass=all(gates.values()),
                raw_sha256=sha(args.out/(name+".pt")), png_sha256=sha(args.out/(name+".png")),
                elapsed_seconds=elapsed)
            print(json.dumps(value), flush=True)
            return value, result

        for precision, steps in SETTINGS:
            row, raw = evaluate(precision, steps, 64, 20267302, f"calibration_{precision}_{steps}")
            if (precision, steps) == ("bf16", 32):
                if any(not torch.equal(raw[key], original[key]) for key in (
                        "initial_noise", "endpoints", "coordinates", "pixels")):
                    raise ValueError("original sampler did not replay exactly")
            results.append(row)
            if selected is None and row["clean_glyph_control_pass"]:
                selected = (precision, steps)
        confirmation = None
        if selected is not None:
            confirmation, _ = evaluate(*selected, 128, 20267303, "confirmation")
        atomic_json(args.out/"report.json", dict(identity=identity, calibration=results,
            original_sampler_replayed_exactly=True, confirmation=confirmation,
            confirmed_clean_glyph_generation=bool(confirmation and confirmation["clean_glyph_control_pass"]),
            original_control_result_replaced=False, language_quality_claim=False,
            general_ilm_goal_complete=False, peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3))


def verify(args):
    from scripts.verify_codec_language_v68 import physical_scores
    report = json.loads((args.out/"report.json").read_text())
    for name, digest in report["identity"]["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("sampling diagnosis implementation changed")
    identity = report["identity"]
    if (sha(args.control/"report.json") != identity["source_control_report_sha256"]
            or sha(args.control/"verification.json") != identity["source_control_verification_sha256"]
            or sha(args.control/"full/checkpoint_latest.pt") != identity["checkpoint_sha256"]):
        raise ValueError("source control identity changed")
    primary = json.loads((args.control/"report.json").read_text())
    if sha(args.control/"full/evaluation.pt") != primary["arms"]["full"]["evaluation_sha256"]:
        raise ValueError("source physical output changed")
    reference, _, _ = source_case(args.run)
    original = torch.load(args.control/"full/evaluation.pt", map_location="cpu", weights_only=True)
    if not torch.equal(original["reference"], reference):
        raise ValueError("source physical reference changed")
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cpu")))
    ref = ((reference[0, 0]+1)*127.5).to(torch.uint8)
    verified = []
    if [(row["precision"], row["steps"], row["count"], row["seed"])
            for row in report["calibration"]] != [(p, s, 64, 20267302) for p, s in SETTINGS]:
        raise ValueError("fixed calibration settings changed")
    rows = report["calibration"]+([report["confirmation"]] if report["confirmation"] else [])
    for row in rows:
        path = args.out/(row["name"]+".pt")
        if sha(path) != row["raw_sha256"]:
            raise ValueError("retained sampler output changed")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if row["name"] == "calibration_bf16_32" and any(
                not torch.equal(raw[key], original[key]) for key in (
                    "initial_noise", "endpoints", "coordinates", "pixels")):
            raise ValueError("retained original sampler replay differs")
        pixels = raw["pixels"]
        if pixels.shape != (row["count"], 1, 32, 64) or not bool(((pixels == -1) | (pixels == 1)).all()):
            raise ValueError("invalid physical samples")
        decoded = writer.predict(raw["coordinates"]).detach()
        replay = torch.where(decoded < 0, -torch.ones_like(decoded), torch.ones_like(decoded))
        if not torch.equal(replay, pixels):
            raise ValueError("independent fixed decoder replay differs")
        physical = [physical_scores(x[0].numpy() < 0, [ref], True) for x in pixels]
        measured = dict(mean_foreground_f1=float(np.mean([x["foreground_f1"] for x in physical])),
            minimum_foreground_f1=float(min(x["foreground_f1"] for x in physical)),
            original_exact_fraction=float(np.mean([x["exact"] for x in physical])),
            supplied_render_exact_fraction=float(np.mean([
                np.array_equal(x.numpy(), original["supplied_render"][0].numpy()) for x in pixels])))
        if any(abs(value-row["measurements"][key]) > 1e-7 for key, value in measured.items()):
            raise ValueError("independent physical scoring differs")
        if quality_gates(measured) != row["gates"]:
            raise ValueError("sampling quality gate differs")
        png = args.out/(row["name"]+".png")
        if sha(png) != row["png_sha256"] or not np.array_equal(np.asarray(Image.open(png)), preview(pixels)):
            raise ValueError("preview pixels differ")
        verified.append(dict(name=row["name"], measurements=measured, gates=quality_gates(measured)))
    selected = next((row for row in report["calibration"] if all(row["gates"].values())), None)
    confirmation = report["confirmation"]
    if (selected is None) != (confirmation is None):
        raise ValueError("confirmation selection rule changed")
    if confirmation and (confirmation["precision"], confirmation["steps"], confirmation["count"],
            confirmation["seed"]) != (selected["precision"], selected["steps"], 128, 20267303):
        raise ValueError("fresh-noise confirmation changed")
    confirmed = bool(confirmation and all(verified[-1]["gates"].values()))
    if confirmed != report["confirmed_clean_glyph_generation"]:
        raise ValueError("independent confirmation outcome differs")
    atomic_json(args.out/"verification.json", dict(report_sha256=sha(args.out/"report.json"),
        verified=verified, independent_cpu_physical_scoring=True, cpu_fixed_decoder_replayed=True,
        full_sampler_independently_replayed=False,
        confirmed_clean_glyph_generation=confirmed,
        language_quality_claim=False, general_ilm_goal_complete=False))
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"artifacts/semantic_ilm_v72_20261105")
    parser.add_argument("--control", type=Path, default=ROOT/"artifacts/full_rank_flow_v73_20261106")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        torch.set_num_threads(4)
        verify(arguments)
    else:
        run(arguments)
