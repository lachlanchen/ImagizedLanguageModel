#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.folio import FolioRetina, folio_config_from_payload
from ilm.visual_lm.folio_data import (
    FolioRenderConfig,
    folio_tensor_to_image,
    load_teacher_cache,
    render_folio,
    semantic_residual_fields,
)
from ilm.visual_lm.folio_memory import FolioMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate independent image-to-image folio retrieval.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--paraphrases", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="artifacts/visual_folio_evaluation")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def chunks(values: Sequence[Any], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def image_to_ink(path: Path, *, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image.size != (width, height):
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (width, height), 255)
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        image = canvas
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - array)[None]


def metric_template() -> dict[str, float]:
    return {
        "examples": 0.0,
        "top1": 0.0,
        "top5": 0.0,
        "reciprocal_rank": 0.0,
        "score": 0.0,
        "margin": 0.0,
        "teacher_cosine": 0.0,
        "teacher_examples": 0.0,
    }


def finalize(values: dict[str, float]) -> dict[str, float | int | None]:
    count = max(1.0, values["examples"])
    teacher_count = values["teacher_examples"]
    return {
        "examples": int(values["examples"]),
        "top1_accuracy": values["top1"] / count,
        "top5_accuracy": values["top5"] / count,
        "mean_reciprocal_rank": values["reciprocal_rank"] / count,
        "mean_best_score": values["score"] / count,
        "mean_score_margin": values["margin"] / count,
        "mean_teacher_cosine": values["teacher_cosine"] / teacher_count if teacher_count else None,
    }


def teacher_fields(path: str | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    cache = load_teacher_cache(path)
    residuals, _ = semantic_residual_fields(cache)
    output = {}
    for index, document in enumerate(cache["documents"]):
        if document["kind"] == "prompt":
            output[str(document["record_identifier"])] = residuals[index]
    return output


def append_paraphrase_targets(
    targets: list[tuple[int, dict[str, Any], Path, str]],
    *,
    paraphrase_path: str | None,
    entries: Sequence[dict[str, Any]],
    render_config: FolioRenderConfig,
    output: Path,
) -> None:
    if paraphrase_path is None:
        return
    by_identifier = {str(entry.get("identifier")): index for index, entry in enumerate(entries)}
    query_root = output / "paraphrase_queries"
    query_root.mkdir(parents=True, exist_ok=True)
    for line_index, line in enumerate(Path(paraphrase_path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = str(record.get("identifier", ""))
        entry_index = by_identifier.get(identifier)
        paraphrase = record.get("paraphrase")
        if entry_index is None or not isinstance(paraphrase, str) or not paraphrase.strip():
            continue
        digest = hashlib.sha256(f"{identifier}\0{paraphrase}".encode("utf-8")).hexdigest()
        variant = int(digest[:8], 16)
        image = render_folio(paraphrase, config=render_config, variant=variant, augment=True)
        path = query_root / f"{line_index:05d}_{digest[:12]}.png"
        folio_tensor_to_image(image).save(path, optimize=True)
        targets.append((entry_index, entries[entry_index], path, "validated_paraphrase"))


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = folio_config_from_payload(checkpoint["model_config"])
    render_payload = dict(checkpoint["render_config"])
    render_payload["augment"] = False
    render_config = FolioRenderConfig(**render_payload)
    model = FolioRetina(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    memory = FolioMemory.load(args.memory)
    targets = []
    for entry_index, entry in enumerate(memory.entries):
        for relative in entry.get("evaluation_images", []):
            path = Path(str(relative))
            targets.append(
                (
                    entry_index,
                    entry,
                    path if path.is_absolute() else memory.root / path,
                    "held_out_render",
                )
            )
    append_paraphrase_targets(
        targets,
        paraphrase_path=args.paraphrases,
        entries=memory.entries,
        render_config=render_config,
        output=output,
    )
    if args.limit is not None:
        targets = targets[: args.limit]
    teacher = teacher_fields(args.teacher_cache)
    groups: dict[str, dict[str, float]] = defaultdict(metric_template)
    predictions = []
    encode_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in chunks(targets, max(1, args.batch_size)):
        images = torch.stack(
            [image_to_ink(path, height=config.image_height, width=config.image_width) for _, _, path, _ in batch]
        ).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            fields = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_seconds += time.perf_counter() - started
        hit_batches = memory.search(fields, top_k=max(5, args.top_k))
        for row, ((expected, entry, path, evaluation_policy), hits) in enumerate(zip(batch, hit_batches)):
            ranks = [rank for rank, hit in enumerate(hits, 1) if hit.entry_index == expected]
            rank = ranks[0] if ranks else None
            best_score = hits[0].score if hits else -1.0
            margin = best_score - hits[1].score if len(hits) > 1 else 0.0
            identifier = str(entry.get("identifier", ""))
            expected_teacher = teacher.get(identifier)
            teacher_cosine = None
            if expected_teacher is not None:
                teacher_cosine = float(torch.dot(fields[row].float().cpu(), expected_teacher))
            group_names = (
                "all",
                str(entry.get("kind", "unknown")),
                str(entry.get("encoder_training_status", "unknown")),
                str(entry.get("language", "unknown")),
                evaluation_policy,
            )
            for group_name in group_names:
                values = groups[group_name]
                values["examples"] += 1
                values["top1"] += float(rank == 1)
                values["top5"] += float(rank is not None and rank <= 5)
                values["reciprocal_rank"] += 0.0 if rank is None else 1.0 / rank
                values["score"] += best_score
                values["margin"] += margin
                if teacher_cosine is not None:
                    values["teacher_cosine"] += teacher_cosine
                    values["teacher_examples"] += 1
            predictions.append(
                {
                    "expected_entry": expected,
                    "identifier": identifier,
                    "kind": entry.get("kind"),
                    "evaluation_image": str(path),
                    "evaluation_policy": evaluation_policy,
                    "rank": rank,
                    "best_score": best_score,
                    "score_margin": margin,
                    "teacher_cosine": teacher_cosine,
                    "retrieved_entries": [hit.entry_index for hit in hits],
                }
            )

    report = {
        "architecture": "visual-folio-memory-v1",
        "checkpoint": args.checkpoint,
        "memory": args.memory,
        "memory_entries": len(memory.entries),
        "memory_keys": int(memory.keys.shape[0]),
        "evaluation_images": len(targets),
        "groups": {name: finalize(values) for name, values in sorted(groups.items())},
        "encode_seconds_total": encode_seconds,
        "images_per_second": len(targets) / max(encode_seconds, 1e-9),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "key_bytes": memory.keys.numel() * memory.keys.element_size(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "student_received_text": False,
        "student_used_tokens": False,
        "student_used_ocr": False,
        "student_called_external_model": False,
        "limitations": [
            "Retrieval demonstrates visual semantic addressing, not unrestricted answer generation.",
            "Instruction evaluation uses unseen renderings of known wording unless a paraphrase suite is supplied.",
            "Historical evaluation uses held-out multilingual wording and exact evidence-bearing answer images.",
        ],
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
