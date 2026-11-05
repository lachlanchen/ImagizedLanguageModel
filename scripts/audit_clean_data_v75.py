#!/usr/bin/env python3
"""Check exact training/audit raster overlap without loading a neural model."""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.codec_language_audit import AUDIT_DATA_SHA, CodecLanguageAuditData
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, sha, atomic_json, committed_identity

SOURCES = ("scripts/audit_clean_data_v75.py", "ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/complete_writing_data.py")


def fingerprint(packed, width):
    """Shape plus row-major binary pixels; no font/text interpretation."""
    if packed.shape != (32, width//8) or width % 64 or packed.dtype != np.uint8:
        raise ValueError("complete binary strip required")
    return hashlib.sha256(np.asarray([32, width], dtype="<i8").tobytes()+packed.tobytes()).hexdigest()


def pixel_fingerprint(image):
    return fingerprint(np.packbits(image.numpy() != 0, axis=1), image.shape[1])


def distribution(values):
    value = np.asarray(values)
    return dict(count=len(values), minimum=int(value.min()), median=float(np.median(value)),
        p90=float(np.quantile(value, .9)), maximum=int(value.max()), mean=float(value.mean()))


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("this data-only audit must hide CUDA")
    if args.out.exists():
        raise ValueError("retain completed evidence; choose a fresh report")
    torch.set_num_threads(2)
    started = time.perf_counter()
    identity = committed_identity(SOURCES)
    if sha(TRAIN/"report.json") != TRAIN_SHA:
        raise ValueError("admitted training inventory changed")
    training = json.loads((TRAIN/"report.json").read_text())
    if not training["training_only"] or not training["all_retained_rows_lossless"]:
        raise ValueError("complete training-only images required")
    inventory = training["files_sha256"]
    if len(inventory) != 12976:
        raise ValueError("full admitted corpus required")
    summaries, identifiers = {}, set()
    prompts, targets = Counter(), Counter()
    for source in ("cmrc", "dolly"):
        manifest = TRAIN/(source+"_accepted.jsonl")
        if sha(manifest) != training["summaries"][source]["accepted_manifest_sha256"]:
            raise ValueError("accepted manifest changed")
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if {row["file"] for row in rows} != {name for name in inventory if name.startswith(source+"/")}:
            raise ValueError("accepted manifest and physical inventory differ")
        prompt_lengths, answer_lengths = [], []
        categories = Counter()
        source_targets = set()
        for row in rows:
            path = Path(row["file"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe image path")
            raw = (TRAIN/path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != inventory[str(path)]:
                raise ValueError("source image changed")
            saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            width, boundary, bits = saved["width"], saved["prompt_blocks"], saved["bits"]
            if (type(width) is not int or type(boundary) is not int or width % 64
                    or not 1 <= boundary <= 240 or not 1 <= width//64-boundary-2 <= 128
                    or bits.dtype != torch.uint8 or bits.ndim != 1 or bits.numel() != 32*width//8):
                raise ValueError("invalid admitted physical image bounds")
            packed = bits.numpy().reshape(32, width//8)
            if not np.all(packed[:, -16:] == 255):
                raise ValueError("explicit white termination blocks changed")
            prompt_key = fingerprint(packed[:, :boundary*8], boundary*64)
            answer_blocks = width//64-boundary-2
            target_key = fingerprint(packed[:, boundary*8:-16], answer_blocks*64)
            prompts[prompt_key] += 1
            targets[target_key] += 1
            source_targets.add(target_key)
            identifiers.add((source, row["id"]))
            prompt_lengths.append(boundary)
            answer_lengths.append(answer_blocks)
            categories[row["category"]] += 1
        summaries[source] = dict(examples=len(rows), categories=dict(categories),
            unique_answer_rasters=len(source_targets), prompt_blocks=distribution(prompt_lengths),
            answer_blocks=distribution(answer_lengths))
    audit = CodecLanguageAuditData(ROOT/"data/general_instructions/codec_language_audit_v68_20261105")
    shared_ids, shared_prompts, repeated_targets = [], [], []
    audit_sources = Counter()
    for i, case in enumerate(audit.cases):
        payload = audit.payload(i)
        key = case["source"]+"/"+case["id"]
        audit_sources[case["source"]] += 1
        if (case["source"], case["id"]) in identifiers:
            shared_ids.append(key)
        if pixel_fingerprint(payload["prompt"]) in prompts:
            shared_prompts.append(key)
        if any(pixel_fingerprint(reference) in targets for reference in payload["references"]):
            repeated_targets.append(key)
    if identity["sources"] != {name:sha(ROOT/name) for name in SOURCES}:
        raise ValueError("audit implementation changed during execution")
    result = dict(identity=identity, training_report_sha256=TRAIN_SHA, audit_report_sha256=AUDIT_DATA_SHA,
        training_images_authenticated=len(inventory), audit_images_authenticated=len(audit.cases),
        training=summaries, audit_source_counts=dict(audit_sources),
        unique_training_prompt_rasters=len(prompts), unique_training_answer_rasters=len(targets),
        exact_record_id_overlap=shared_ids, exact_prompt_raster_overlap=shared_prompts,
        cases_with_a_reference_raster_seen_in_training=repeated_targets,
        no_exact_prompt_or_record_overlap=not shared_ids and not shared_prompts,
        model_calls=0, gpu_allocations=0, sealed_examples_opened=0,
        external_foundation_contamination_assessed=False, near_duplicate_passages_assessed=False,
        language_quality_claim=False, elapsed_seconds=time.perf_counter()-started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(json.dumps({k:result[k] for k in ("training_images_authenticated", "audit_images_authenticated",
        "exact_record_id_overlap", "exact_prompt_raster_overlap", "no_exact_prompt_or_record_overlap",
        "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
