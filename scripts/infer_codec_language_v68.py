#!/usr/bin/env python3
"""Read a prompt image and generate writing with the final, audited V68 actor."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from ilm.visual_lm.codec_image_interface import (
    json_file, prepare_image_prompt, save_generation, save_prepared_prompt, sha,
)

ROOT = Path(__file__).resolve().parents[1]


def audited_identity(run):
    # Completion and audit admission happen before CUDA or model loading.
    from ilm.visual_lm.codec_language_runtime import completed_identity
    run = Path(run)
    if not (run / "training_report.json").exists():
        raise RuntimeError("V68 training is unfinished; preserve the same trainer and wait")
    training = completed_identity(run)
    receipt = run / "development/independent_verification.json"
    if not receipt.exists():
        raise RuntimeError("wait for the existing autonomous audit and independent verification")
    verified = json.loads(receipt.read_text())
    report_path = run / "development/report.json"
    report = json.loads(report_path.read_text())
    identity = report["identity"]
    if (not verified["all_verification_checks_pass"] or verified["raw_generations_verified"] != 384
            or verified["report_sha256"] != sha(report_path)
            or identity["training_report_sha256"] != sha(run / "training_report.json")
            or identity["checkpoint_sha256"] != training["final_trainable_sha256"]):
        raise ValueError("final adaptation and independently verified audit must match")
    return dict(training_report_sha256=sha(run / "training_report.json"),
        checkpoint_sha256=training["final_trainable_sha256"],
        audit_report_sha256=sha(report_path), audit_verification_sha256=sha(receipt),
        bounded_chinese_development_pass=verified["bounded_chinese_development_pass"],
        standalone_export_verified=False, general_language_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="one observed prompt image")
    parser.add_argument("--layout", choices=("page", "strip"), default="page")
    parser.add_argument("--strip-page-width", type=int, default=1024)
    parser.add_argument("--maximum-blocks", type=int, default=128)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    parser.add_argument("--out", type=Path, required=True, help="new immutable output directory")
    parser.add_argument("--prepare-only", action="store_true", help="CPU input validation; no model call")
    args = parser.parse_args()
    if not 1 <= args.maximum_blocks <= 128:
        raise ValueError("capacity must be between 1 and 128 physical blocks")
    if args.out.exists():
        raise FileExistsError("choose a new output directory; previous evidence is immutable")
    torch.set_num_threads(4)
    prepared = prepare_image_prompt(args.image, layout=args.layout, maximum_width=args.strip_page_width)
    if args.prepare_only:
        receipt = save_prepared_prompt(prepared, args.out)
        json_file(args.out / "report.json", dict(input=receipt, preparation_only=True,
            model_calls=0, gpu_allocations=0, language_quality_claim=False))
        print(json.dumps(dict(stage="input_prepared", shape=list(prepared.pixels.shape),
            out=str(args.out), model_calls=0)))
        return
    identity = audited_identity(args.run)
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("physical GPU 0 and deterministic cuBLAS must be configured before loading")
    from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
    check_host_memory()
    if gpu_memory() > 256:
        raise RuntimeError("GPU 0 is occupied; preserve the existing project job")
    receipt = save_prepared_prompt(prepared, args.out)
    started = time.perf_counter()
    try:
        torch.manual_seed(20266832)
        torch.cuda.manual_seed_all(20266832)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        from ilm.visual_lm.codec_language_runtime import load_completed_actor
        actor, _ = load_completed_actor(args.run, root=ROOT, device=torch.device("cuda:0"))
        setup_seconds = time.perf_counter() - started
        torch.cuda.synchronize()
        before = time.perf_counter()
        # Only an owning image tensor enters the actor; no receipt or filename.
        result = actor.generate(prepared.pixels, maximum_blocks=args.maximum_blocks)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - before
        output = save_generation(result, args.out / "generation", maximum_blocks=args.maximum_blocks)
        json_file(args.out / "report.json", dict(identity=identity, input=receipt, output=output,
            setup_seconds=setup_seconds, generation_seconds=seconds,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            model_calls=1, actor_teacher_calls=0, actor_ocr_calls=0,
            standalone_export_verified=False, general_language_qualified=False,
            language_quality_claim=False))
        print(json.dumps(dict(stage="generated", blocks=output["blocks"],
            stopped=output["stopped"], capped=output["capped"], out=str(args.out))))
    except BaseException as error:
        json_file(args.out / "failure.json", dict(error=repr(error), identity=identity))
        raise


if __name__ == "__main__":
    main()
