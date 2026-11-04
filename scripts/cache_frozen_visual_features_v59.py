#!/usr/bin/env python3
"""Cache frozen pixel-language features without exposing answer annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from ilm.visual_lm.frozen_visual_foundation import extract_features, make_projectors
from ilm.visual_lm.visual_answer_blocks import model_from_checkpoint
from ilm.visual_lm.visual_qa_data import VisualQARenderer

REVISION = "c5d5bcfbad2f290296fd83810558f396c6a6333a"
WEIGHT_SHA = "be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8"
STUDENT = (
    "artifacts/visual_answer_blocks_v58_20261105/standalone/ilm_reading_standalone.pt"
)
STUDENT_SHA = "357c2fec17cf5977f956c8529189b2fde46ed727d4c02bb48355fd4db12a281b"
DATA = "data/visual_qa/cmrc2018_short"
DATA_HASHES = {
    "train.jsonl": "742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c",
    "development.jsonl": "8bef093075ac0715d5878a1abe61aa38092893ade8753b935cdf657551ebc6a7",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--snapshot",
        default=str(
            Path.home()
            / ".cache/huggingface/hub/models--TYTTYTTYT--llama-pixar-477m-p32-stage2/snapshots"
            / REVISION
        ),
    )
    p.add_argument("--student", default=STUDENT)
    p.add_argument("--data", default=DATA)
    p.add_argument(
        "--out", default="artifacts/frozen_visual_foundation_v59_20261105/features"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError("completed feature cache exists")
    if digest(a.student) != STUDENT_SHA:
        raise ValueError("student export changed")
    snap = Path(a.snapshot)
    if snap.name != REVISION or digest(snap / "model.safetensors") != WEIGHT_SHA:
        raise ValueError("author revision/weights changed")
    for name, sha in DATA_HASHES.items():
        if digest(Path(a.data) / name) != sha:
            raise ValueError("question data changed")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(a.device)
    if device.type != "cuda" or device.index not in (0, None):
        raise ValueError("feature preparation uses physical GPU 0 only")
    projectors = make_projectors()
    identity = {
        "student_sha256": STUDENT_SHA,
        "author_sha256": WEIGHT_SHA,
        "author_revision": REVISION,
        "data_sha256": DATA_HASHES,
        "smoke_only": a.smoke,
        "projector_sha256": {
            name: hashlib.sha256(t.numpy().tobytes()).hexdigest()
            for name, t in projectors.items()
        },
        "sources": {
            name: digest(name)
            for name in [
                "scripts/cache_frozen_visual_features_v59.py",
                "ilm/visual_lm/frozen_visual_foundation.py",
                "references/frozen_visual_foundation_v59_protocol.md",
                "ilm/visual_lm/visual_answer_blocks.py",
                "ilm/visual_lm/visual_copy_language.py",
                "ilm/visual_lm/hierarchical_raster_language.py",
                "ilm/visual_lm/canonical_glyph_language.py",
                "ilm/visual_lm/visual_cell_stream.py",
            ]
        },
        "external_code_sha256": {
            name: digest(snap / name)
            for name in [
                "modeling_llama_pixar.py",
                "configuration_llama_pixar.py",
                "utils_llama_pixar.py",
                "config.json",
            ]
        },
        "feature_dim": 514,
        "encoder_autocast": "bf16",
        "projection": "fp32 highest",
        "cache_dtype": "float16",
    }
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("partial feature cache identity changed")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    atomic_save(projectors, out / "projectors.pt")
    begun = time.perf_counter()
    from transformers import AutoModelForCausalLM

    author = (
        AutoModelForCausalLM.from_pretrained(
            str(snap),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        .to(device)
        .requires_grad_(False)
        .eval()
    )
    student = model_from_checkpoint(
        torch.load(a.student, map_location="cpu", weights_only=False), device
    )
    renderer = VisualQARenderer()
    torch.cuda.reset_peak_memory_stats(device)
    counts = {}
    reused = 0
    maximum_error = 0.0
    source_hashes = {}
    for split, file, blank in [
        ("train", "train.jsonl", False),
        ("development", "development.jsonl", False),
        ("development_blank", "development.jsonl", True),
    ]:
        # References are never read, filtered, rendered or sent to an encoder.
        records = [
            json.loads(line) for line in (Path(a.data) / file).read_text().splitlines()
        ]
        if a.smoke:
            records = records[:4]
        folder = out / split
        folder.mkdir(exist_ok=True)
        counts[split] = len(records)
        for index, record in enumerate(records):
            path = folder / (record["id"] + ".pt")
            if path.exists():
                reused += 1
                continue
            pixels = renderer(record["prompt"]).to(device)
            if blank:
                start = 6 + len(record["context"])
                assert (
                    record["prompt"][start : start + len(record["question"])]
                    == record["question"]
                )
                pixels[start : start + len(record["question"])] = 0
            values = extract_features(student, author, pixels, projectors)
            if any(
                value.shape != (len(pixels), 514)
                or not bool(torch.isfinite(value).all())
                for value in values.values()
            ):
                raise FloatingPointError("invalid continuous features")
            if index == 0:
                again = extract_features(student, author, pixels, projectors)
                maximum_error = max(
                    maximum_error,
                    max(float((values[k] - again[k]).abs().max()) for k in values),
                )
                if maximum_error != 0:
                    raise RuntimeError(
                        "feature replay is not bit-identical after quantization"
                    )
            atomic_save(
                {
                    "id": record["id"],
                    "prompt_cells": len(pixels),
                    "features": values,
                    "prompt_sha256": hashlib.sha256(
                        record["prompt"].encode()
                    ).hexdigest(),
                    "question_blanked": blank,
                },
                path,
            )
            source_hashes[str(path.relative_to(out))] = digest(path)
            if index % 50 == 0:
                print(
                    json.dumps(
                        {"split": split, "completed": index + 1, "total": len(records)}
                    ),
                    flush=True,
                )
            if (
                torch.cuda.max_memory_allocated(device) >= 16 * 1024**3
                or time.perf_counter() - begun > 4 * 3600
            ):
                raise RuntimeError(
                    "feature preparation resource limit reached; partial cache preserved"
                )
    report = {
        **identity,
        "counts": counts,
        "records_reused": reused,
        "recomputed_feature_max_error": maximum_error,
        "elapsed_seconds_this_process": time.perf_counter() - begun,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "framework": {"torch": torch.__version__},
        "answer_annotations_used": False,
        "cache_files_sha256": {
            str(p.relative_to(out)): digest(p)
            for split in counts
            for p in sorted((out / split).glob("*.pt"))
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "cache_files_sha256"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
