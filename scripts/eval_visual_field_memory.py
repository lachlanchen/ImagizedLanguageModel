#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from ilm.visual_lm.dataset import pil_to_tensor
from ilm.visual_lm.retinal_memory import (
    VisualAssociativeReader,
    VisualEpisodeMemory,
    retinal_config_from_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-only visual memory on separately rendered query images."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--out", default="artifacts/visual_field_evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def chunks(values: Sequence[Any], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def metric_template() -> dict[str, float]:
    return {"examples": 0.0, "top1": 0.0, "top5": 0.0, "reciprocal_rank": 0.0, "score": 0.0, "margin": 0.0}


def finalize(values: dict[str, float]) -> dict[str, float | int]:
    count = max(1.0, values["examples"])
    return {
        "examples": int(values["examples"]),
        "top1_accuracy": values["top1"] / count,
        "top5_accuracy": values["top5"] / count,
        "mean_reciprocal_rank": values["reciprocal_rank"] / count,
        "mean_best_score": values["score"] / count,
        "mean_score_margin": values["margin"] / count,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = retinal_config_from_payload(checkpoint["retinal_config"])
    model = VisualAssociativeReader(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    memory = VisualEpisodeMemory.load(args.memory)
    selected = list(enumerate(memory.entries))
    if args.limit is not None:
        selected = selected[: args.limit]

    groups: dict[str, dict[str, float]] = defaultdict(metric_template)
    records: list[dict[str, Any]] = []
    total_encode_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in chunks(selected, max(1, args.batch_size)):
        images = []
        for _, entry in batch:
            path = Path(str(entry["evaluation_image"]))
            if not path.is_absolute():
                path = memory.root / path
            images.append(pil_to_tensor(Image.open(path).convert("RGB")))
        image_tensor = torch.stack(images).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            query = model.encode_query(image_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_encode_seconds += time.perf_counter() - started
        hit_batches = memory.search(query, top_k=5)
        for (expected_index, entry), hits in zip(batch, hit_batches):
            ranks = [rank for rank, hit in enumerate(hits, 1) if hit.entry_index == expected_index]
            rank = ranks[0] if ranks else None
            best_score = hits[0].score if hits else -1.0
            margin = best_score - hits[1].score if len(hits) > 1 else 0.0
            group_names = (
                "all",
                str(entry.get("kind", "unknown")),
                str(entry.get("encoder_training_status", "unknown")),
                str(entry.get("evaluation_policy", "unknown")),
            )
            for group_name in group_names:
                values = groups[group_name]
                values["examples"] += 1
                values["top1"] += float(rank == 1)
                values["top5"] += float(rank is not None and rank <= 5)
                values["reciprocal_rank"] += 0.0 if rank is None else 1.0 / rank
                values["score"] += best_score
                values["margin"] += margin
            records.append(
                {
                    "expected_entry": expected_index,
                    "identifier": entry.get("identifier"),
                    "kind": entry.get("kind"),
                    "encoder_training_status": entry.get("encoder_training_status"),
                    "rank": rank,
                    "best_score": best_score,
                    "score_margin": margin,
                    "retrieved_entries": [hit.entry_index for hit in hits],
                }
            )

    report = {
        "architecture": checkpoint.get("architecture"),
        "checkpoint": str(args.checkpoint),
        "memory": str(args.memory),
        "evaluation_input": "separately rendered and capture-damaged images",
        "student_used_text_tokens": False,
        "student_used_ocr": False,
        "student_used_external_llm": False,
        "groups": {name: finalize(values) for name, values in sorted(groups.items())},
        "encode_seconds_total": total_encode_seconds,
        "images_per_second": len(selected) / max(total_encode_seconds, 1e-9),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "memory_key_bytes": memory.keys.numel() * memory.keys.element_size(),
        "limitations": [
            "Retrieval accuracy proves visual addressing, not open-ended generative reasoning.",
            "Instruction queries retain their wording but use a held-out render and damage process.",
            "Historical queries use held-out wording and render; answer glyphs are exact evidence copies.",
        ],
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
