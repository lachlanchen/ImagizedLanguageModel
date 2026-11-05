#!/usr/bin/env python3
"""Evaluate complete V68 answer-image streams, or validate the audit on CPU."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.codec_language_audit import (
    AUDIT_DATA_SHA, CodecLanguageAuditData, output_strip, reference_scores, sha, summarize,
)
from ilm.visual_lm.codec_language_runtime import completed_identity, load_completed_actor
from scripts.check_grounded_image_actor import ROOT, atomic_json

SOURCES = ("scripts/eval_codec_language_v68.py", "ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/codec_language_runtime.py", "references/codec_language_v68_autonomous_audit.md")
DATA = ROOT / "data/general_instructions/codec_language_audit_v68_20261105"


def committed_sources():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {name:sha(ROOT / name) for name in SOURCES}
    for name, digest in sources.items():
        original = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
        if hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("commit evaluator sources before execution")
    return revision, sources


def data_preflight(data, out, revision, sources):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only evaluator preflight required")
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    for index in range(192):
        value = data.payload(index)
        for blank in (False, True):
            prompt = data.prompt(index, blank=blank)
            if (prompt.dtype != torch.float32 or prompt.device.type != "cpu" or prompt._base is not None
                    or prompt.storage_offset() != 0
                    or prompt.untyped_storage().nbytes() != prompt.numel() * prompt.element_size()):
                raise ValueError("only owning prompt-pixel storage may enter generation")
        reference = value["references"][0]
        exact = reference_scores(reference, (reference,), stopped=True)
        wrong_stop = reference_scores(reference, (reference,), stopped=False)
        extra_white = torch.cat((reference, torch.full((32, 64), 255, dtype=torch.uint8)), -1)
        wrong_width = reference_scores(extra_white, (reference,), stopped=True)
        empty = reference_scores(torch.empty((32, 0), dtype=torch.uint8), (reference,), stopped=True)
        if (not exact["exact"] or exact["foreground_f1"] != 1 or wrong_stop["exact"]
                or wrong_width["exact"] or wrong_width["foreground_f1"] != 1
                or empty["exact"] or empty["foreground_f1"] != 0):
            raise AssertionError("physical reference/stop/width metric invariants failed")
    checks["all_192_source_bound_cases"] = True
    checks["all_384_prompt_storage_paths_owning"] = True
    checks["reference_stop_width_empty_distinctions"] = True
    # Test the registered aggregation with measured physical reference scores,
    # not a mock actor or a claim about generated language.
    ideal = {}
    for index, row in enumerate(data.cases):
        reference = data.references(index)[0]
        ideal[row["source"] + "/" + row["id"]] = dict(
            normal=reference_scores(reference, (reference,), stopped=True),
            blank=reference_scores(torch.empty((32, 0), dtype=torch.uint8), (reference,), stopped=True))
    aggregate = summarize(data, ideal)
    checks["article_bootstrap_reference_limit"] = (
        aggregate["question_effect"]["mean"] == 1
        and aggregate["question_effect"]["ci95"] == [1., 1.]
        and aggregate["paired_both_correct"] == 1
        and all(aggregate["gates"].values()))
    no_question = {key:dict(normal=value["normal"], blank=value["normal"]) for key, value in ideal.items()}
    negative = summarize(data, no_question)
    checks["zero_question_effect_rejected"] = (
        negative["question_effect"]["mean"] == 0
        and not negative["gates"]["question_gain"]
        and not negative["gates"]["question_gain_ci_positive"])
    if not all(checks.values()):
        raise AssertionError(checks)
    report = dict(checks=checks, source_commit=revision, sources=sources,
        prepared_data_sha256=AUDIT_DATA_SHA, actual_questions=192,
        model_calls=0, gpu_allocations=0, metric_validation_only=True,
        language_quality_claim=False, general_ilm_goal_complete=False)
    atomic_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-data-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    revision, sources = committed_sources()
    data = CodecLanguageAuditData(DATA)
    if args.validate_data_only:
        data_preflight(data, args.out or args.run / "audit_preflight/attempt01", revision, sources)
        return
    # Inspect completion before creating an audit directory or CUDA context.
    training = completed_identity(args.run)
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("physical GPU 0 and deterministic cuBLAS required")
    preflight = json.loads((args.run / "audit_preflight/attempt01/report.json").read_text())
    if preflight["sources"] != sources or not all(preflight["checks"].values()) or preflight["model_calls"] != 0:
        raise ValueError("source-matched evaluator preflight required")
    check_host_memory()
    if gpu_memory() > 256:
        raise RuntimeError("GPU 0 occupied; wait for the existing process to release it")
    out = args.out or args.run / "development"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError("completed autonomous evaluation is immutable")
    lock = (out / "operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    begun = time.perf_counter()
    identity = dict(training_report_sha256=sha(args.run / "training_report.json"),
        checkpoint_sha256=training["final_trainable_sha256"], prepared_data_sha256=AUDIT_DATA_SHA,
        evaluator_preflight_sha256=sha(args.run / "audit_preflight/attempt01/report.json"),
        sources=sources, maximum_blocks=128)
    if (out / "identity.json").exists():
        if json.loads((out / "identity.json").read_text()) != identity:
            raise ValueError("partial audit source/data/checkpoint identity changed")
    else:
        atomic_json(out / "identity.json", identity)

    def status(stage, **fields):
        value = dict(stage=stage, pid=os.getpid(), physical_gpu=0, **fields)
        atomic_json(out / "status.json", value)
        print(json.dumps(value), flush=True)

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
        actor, _ = load_completed_actor(args.run, root=ROOT, device=torch.device("cuda:0"),
            progress=lambda loaded, total:status("loading_final_actor", loaded=loaded, total=total))
        setup_seconds = time.perf_counter() - begun
        scores, raw_files = {}, {}
        for index, row in enumerate(data.cases):
            key = row["source"] + "/" + row["id"]
            scores[key] = {}
            for route in ("normal", "blank"):
                prompt = data.prompt(index, blank=route == "blank")
                digest = hashlib.sha256(prompt.numpy().tobytes()).hexdigest()
                header = dict(id=row["id"], source=row["source"], route=route,
                    prompt_pixels_sha256=digest, checkpoint_sha256=identity["checkpoint_sha256"])
                relative = Path("predictions") / route / (row["source"] + "_" + row["id"] + ".pt")
                path = out / relative
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["header"] != header:
                        raise ValueError("saved raw generation has another input or checkpoint")
                else:
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    # This call has no case metadata, references, lengths or text.
                    result = actor.generate(prompt, maximum_blocks=128)
                    torch.cuda.synchronize()
                    saved = dict(header=header, result=result, generation_seconds=time.perf_counter() - before)
                    output_strip(result)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".partial")
                    torch.save(saved, temporary)
                    temporary.replace(path)
                strip = output_strip(saved["result"])
                score = reference_scores(strip, data.references(index), stopped=saved["result"]["stopped"])
                scores[key][route] = score
                raw_files[str(relative)] = sha(path)
                if strip.shape[1]:
                    image_path = path.with_suffix(".png")
                    if not image_path.exists():
                        Image.fromarray(strip.numpy()).save(image_path)
                    with Image.open(image_path) as picture:
                        if not np.array_equal(np.asarray(picture.convert("L")), strip.numpy()):
                            raise ValueError("raw PNG differs from generated physical pixels")
                    raw_files[str(image_path.relative_to(out))] = sha(image_path)
                status("generating_answers", completed_generations=2 * index + int(route == "blank") + 1,
                    total_generations=384, id=row["id"], route=route)
            if index % 16 == 0:
                check_host_memory()
        if sources != {name:sha(ROOT / name) for name in SOURCES}:
            raise ValueError("frozen evaluator sources changed")
        summary = summarize(data, scores)
        atomic_json(out / "cases.json", scores)
        report = dict(**summary, identity=identity, source_commit=revision,
            cases_sha256=sha(out / "cases.json"), raw_files_sha256=raw_files,
            actual_generations=384, setup_seconds=setup_seconds,
            elapsed_seconds=time.perf_counter() - begun,
            summed_generation_seconds=sum(torch.load(out / name, map_location="cpu", weights_only=True)["generation_seconds"]
                for name in raw_files if name.endswith(".pt")),
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            frozen_renderer=True, actor_teacher_calls=0, actor_ocr_calls=0,
            standalone_export_verified=False)
        atomic_json(out / "report.json", report)
        status("complete", bounded_chinese_development_pass=report["bounded_chinese_development_pass"],
            general_ilm_goal_complete=False)
    except BaseException as error:
        atomic_json(out / ("failure_" + str(time.time_ns()) + ".json"), dict(error=repr(error), identity=identity))
        status("failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
