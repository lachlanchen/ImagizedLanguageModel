#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.folio import FolioRetina, folio_config_from_payload
from ilm.visual_lm.folio_data import (
    FolioRenderConfig,
    folio_tensor_to_image,
    load_teacher_cache,
    render_folio,
    render_folio_pages,
    stable_fraction,
)
from ilm.visual_lm.folio_memory import FolioMemory
from ilm.visual_lm.rendering import GlyphCorpus
from ilm.visual_lm.teacher import load_teacher_manifest
from ilm.visual_lm.visual_episodes import historical_episode_specs, render_episode_answer


DEFAULT_HISTORICAL_CHARS = "言,中,水,日,月,人,山,火,木,口,学,車,车,王,雨,田,金"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-keyed, image-valued runtime folio with no serialized text."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--out", default="artifacts/visual_folio_memory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--key-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--glyph-root", default=None)
    parser.add_argument("--teacher-manifest", default="data/teacher/historical_qwen8b_v2.jsonl")
    parser.add_argument("--historical-chars", default=DEFAULT_HISTORICAL_CHARS)
    parser.add_argument("--disable-historical", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(path: Path, device: torch.device) -> tuple[FolioRetina, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "visual-folio-retina-v1":
        raise ValueError("checkpoint is not a visual folio retina")
    model = FolioRetina(folio_config_from_payload(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def render_config_from_checkpoint(checkpoint: dict[str, Any]) -> FolioRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = False
    return FolioRenderConfig(**payload)


def paired_documents(cache: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for document in cache["documents"]:
        identifier = str(document["record_identifier"])
        if identifier not in grouped:
            grouped[identifier] = {}
            order.append(identifier)
        grouped[identifier][str(document["kind"])] = document
    pairs = []
    for identifier in order:
        group = grouped[identifier]
        if "prompt" in group and "response" in group:
            pairs.append(
                {
                    "identifier": identifier,
                    "prompt": str(group["prompt"]["text"]),
                    "response": str(group["response"]["text"]),
                    "language": str(group["prompt"]["language"]),
                    "source": str(group["prompt"]["source"]),
                }
            )
    return pairs


@torch.no_grad()
def encode_images(
    model: FolioRetina,
    images: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    fields = []
    for offset in range(0, len(images), batch_size):
        batch = torch.stack(images[offset : offset + batch_size]).to(device, non_blocking=True)
        fields.append(model(batch).cpu())
    return torch.cat(fields, dim=0)


def write_instruction_entry(
    pair: dict[str, Any],
    *,
    entry_index: int,
    root: Path,
    render_config: FolioRenderConfig,
    validation_fraction: float,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    seed = stable_seed(pair["identifier"])
    stem = f"{entry_index:06d}_{hashlib.sha256(pair['identifier'].encode()).hexdigest()[:12]}"
    pages = render_folio_pages(pair["response"], config=render_config, variant=seed + 1000)
    answer_paths = []
    for page_index, page in enumerate(pages):
        relative = Path("answers") / f"{stem}_p{page_index + 1:02d}.png"
        folio_tensor_to_image(page).save(root / relative, optimize=True)
        answer_paths.append(str(relative))
    evaluation = render_folio(
        pair["prompt"],
        config=render_config,
        variant=seed + 90_000,
        augment=True,
    )
    evaluation_relative = Path("evaluation") / f"{stem}.png"
    folio_tensor_to_image(evaluation).save(root / evaluation_relative, optimize=True)
    training_status = (
        "encoder_holdout"
        if stable_fraction(pair["identifier"]) < validation_fraction
        else "encoder_training"
    )
    entry = {
        "identifier": pair["identifier"],
        "source": pair["source"],
        "language": pair["language"],
        "kind": "instruction",
        "answer_images": answer_paths,
        "evaluation_images": [str(evaluation_relative)],
        "encoder_training_status": training_status,
        "output_origin": "retrieved_exact_image_pages",
        "serialized_text": False,
    }
    return entry, [evaluation]


class InstructionMemoryRenderDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[dict[str, Any]],
        *,
        root: Path,
        render_config: FolioRenderConfig,
        validation_fraction: float,
        key_views: int,
    ):
        self.pairs = list(pairs)
        self.root = root
        self.render_config = render_config
        self.validation_fraction = float(validation_fraction)
        self.key_views = int(key_views)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        entry, _ = write_instruction_entry(
            pair,
            entry_index=index,
            root=self.root,
            render_config=self.render_config,
            validation_fraction=self.validation_fraction,
        )
        seed = stable_seed(pair["identifier"])
        keys = torch.stack(
            [
                render_folio(
                    pair["prompt"],
                    config=self.render_config,
                    variant=seed + view * 10_007,
                    augment=True,
                )
                for view in range(self.key_views)
            ]
        )
        return {"entry_index": index, "entry": entry, "key_images": keys}


def instruction_memory_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entry_indices": torch.tensor([item["entry_index"] for item in batch], dtype=torch.long),
        "entries": [item["entry"] for item in batch],
        "key_images": torch.stack([item["key_images"] for item in batch]),
    }


def historical_entries(
    args: argparse.Namespace,
    *,
    root: Path,
    render_config: FolioRenderConfig,
    start_index: int,
) -> list[tuple[dict[str, Any], list[torch.Tensor]]]:
    if args.disable_historical:
        return []
    try:
        seed_corpus = GlyphCorpus(args.glyph_root)
    except FileNotFoundError:
        return []
    requested = [value.strip() for value in args.historical_chars.split(",") if value.strip()]
    available = [character for character in requested if (seed_corpus.root / character).exists()]
    if not available:
        return []
    corpus = GlyphCorpus(seed_corpus.root, characters=available)
    episodes = historical_episode_specs(
        corpus,
        available,
        teacher_records=load_teacher_manifest(args.teacher_manifest),
    )
    output = []
    for offset, episode in enumerate(episodes):
        entry_index = start_index + offset
        seed = stable_seed(episode.identifier)
        stem = f"{entry_index:06d}_historical_{ord(episode.historical_char or ' '):x}"
        answer = render_episode_answer(episode, image_size=768, variant=seed + 1000, augment=False)
        answer_relative = Path("answers") / f"{stem}.png"
        answer.save(root / answer_relative, optimize=True)
        key_images = [
            render_folio(query, config=render_config, variant=seed + index * 113, augment=True)
            for index, query in enumerate(episode.query_variants)
        ]
        evaluation_paths = []
        for query_index, query in enumerate(episode.evaluation_queries):
            image = render_folio(
                query,
                config=render_config,
                variant=seed + 90_000 + query_index * 137,
                augment=True,
            )
            relative = Path("evaluation") / f"{stem}_q{query_index + 1:02d}.png"
            folio_tensor_to_image(image).save(root / relative, optimize=True)
            evaluation_paths.append(str(relative))
        metadata = episode.metadata()
        metadata.update(
            {
                "answer_images": [str(answer_relative)],
                "evaluation_images": evaluation_paths,
                "encoder_training_status": "post_training_insertion",
                "output_origin": "copied_attested_glyph_pixels_on_retrieved_page",
                "serialized_text": False,
            }
        )
        output.append((metadata, key_images))
    return output


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    render_config = render_config_from_checkpoint(checkpoint)
    cache = load_teacher_cache(args.teacher_cache)
    pairs = paired_documents(cache)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    validation_fraction = (
        float(args.validation_fraction)
        if args.validation_fraction is not None
        else float(checkpoint.get("arguments", {}).get("validation_fraction", 0.05))
    )
    root = Path(args.out)
    (root / "answers").mkdir(parents=True, exist_ok=True)
    (root / "evaluation").mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    keys: list[torch.Tensor] = []
    entry_indices: list[int] = []
    started = time.perf_counter()

    render_dataset = InstructionMemoryRenderDataset(
        pairs,
        root=root,
        render_config=render_config,
        validation_fraction=validation_fraction,
        key_views=args.key_views,
    )
    render_loader = DataLoader(
        render_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=instruction_memory_collate,
    )
    for batch in render_loader:
        expected_indices = list(range(len(entries), len(entries) + len(batch["entries"])))
        if batch["entry_indices"].tolist() != expected_indices:
            raise RuntimeError("parallel renderer changed deterministic entry order")
        entries.extend(batch["entries"])
        image_tensor = batch["key_images"].flatten(0, 1)
        encoded_parts = []
        with torch.inference_mode():
            for offset in range(0, image_tensor.shape[0], args.batch_size):
                encoded_parts.append(model(image_tensor[offset : offset + args.batch_size].to(device)).cpu())
        encoded = torch.cat(encoded_parts, dim=0)
        keys.extend(encoded.unbind(dim=0))
        entry_indices.extend(batch["entry_indices"].repeat_interleave(args.key_views).tolist())
        print(
            json.dumps(
                {
                    "stage": "instruction_memory",
                    "entries": len(entries),
                    "keys": len(keys),
                    "total": len(pairs),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    for entry, key_images in historical_entries(
        args,
        root=root,
        render_config=render_config,
        start_index=len(entries),
    ):
        entry_index = len(entries)
        entries.append(entry)
        encoded = encode_images(model, key_images, batch_size=args.batch_size, device=device)
        keys.extend(encoded.unbind(dim=0))
        entry_indices.extend([entry_index] * encoded.shape[0])

    key_tensor = F.normalize(torch.stack(keys).float(), dim=-1)
    index_tensor = torch.tensor(entry_indices, dtype=torch.long)
    atomic_torch_save(
        {"keys": key_tensor.half(), "entry_indices": index_tensor},
        root / FolioMemory.INDEX_NAME,
    )
    with (root / FolioMemory.MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "architecture": "visual-folio-memory-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "entries": len(entries),
        "keys": int(key_tensor.shape[0]),
        "dimensions": int(key_tensor.shape[1]),
        "key_bytes": int(key_tensor.half().numel() * key_tensor.half().element_size()),
        "student_model_calls_received_images_only": True,
        "manifest_contains_prompt_or_answer_text": False,
        "external_model_required_at_inference": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (root / FolioMemory.METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
