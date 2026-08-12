#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ilm.visual_lm.ink_jepa_data import load_visual_grammar_manifest
from ilm.visual_lm.visual_relation_circuit import (
    RELATION_AWARE_ROUTE,
    VisualCanonicalizer,
    VisualRelationCircuit,
    relation_circuit_config_from_payload,
)
from ilm.visual_lm.visual_relation_data import (
    PARTITION_SALT,
    VisualRelationEpisodeConfig,
    VisualRelationEpisodeDataset,
    build_relation_character_bank,
    relation_partition_receipt,
    split_relation_characters,
    visual_relation_collate,
)
from scripts.eval_visual_relation_circuit_development_v23 import (
    AUDIT_ARCHITECTURE,
)
from scripts.train_visual_relation_circuit_v23 import (
    EXPECTED_CANONICALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    load_relation_state,
    validate_canonicalizer_checkpoint,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


ARCHITECTURE = "visual-relation-circuit-v23-blinded-review-pack"
EXPECTED_PAIRED_AUDIT_SHA256 = (
    "8ee56109212b2a5f48ca4ab6b09dabaec4da6fc00db05e26dd262bd2c3dfdf17"
)
REVIEW_SEED = 23_260_833
REVIEW_CASES = 48
REVIEW_HELDOUT_CASES = 12
REVIEW_SEEN_CASES = REVIEW_CASES - REVIEW_HELDOUT_CASES
REVIEW_BATCH_SIZE = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the sealed V23 48-case blinded visual review pack."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--paired-audit", required=True)
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--canonicalizer-checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument(
        "--out", default="artifacts/visual_relation_circuit_v23_blinded_review"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    return parser.parse_args()


def _opaque_identifier(dataset_index: int) -> str:
    payload = f"{REVIEW_SEED}\0{dataset_index}".encode()
    return f"R-{hashlib.sha256(payload).hexdigest()[:10].upper()}"


def select_review_items(
    dataset: VisualRelationEpisodeDataset,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    heldout = 0
    seen = 0
    for index in range(len(dataset)):
        item = dataset[index]
        is_heldout = bool(item["metadata"]["heldout_combination"])
        if is_heldout and heldout < REVIEW_HELDOUT_CASES:
            selected.append((index, item))
            heldout += 1
        elif not is_heldout and seen < REVIEW_SEEN_CASES:
            selected.append((index, item))
            seen += 1
        if len(selected) == REVIEW_CASES:
            break
    if heldout != REVIEW_HELDOUT_CASES or seen != REVIEW_SEEN_CASES:
        raise ValueError("V23 could not construct the fixed blinded review balance")
    return selected


def _ink_image(tensor: torch.Tensor, scale: int = 2) -> Image.Image:
    array = (
        (1.0 - tensor.detach().float().cpu().clamp(0, 1)[0]).numpy() * 255.0
    ).round().astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.NEAREST,
        )
    return image


def render_review_card(
    review_id: str,
    prompt: torch.Tensor,
    generated: torch.Tensor,
) -> Image.Image:
    tile = 64
    gap = 8
    top = 28
    width = 7 * tile + 8 * gap
    height = top + tile + 16
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    draw.text((gap, 7), review_id, fill="#24373f", font=font)
    source_marks = {1: ("A", "#007f8b"), 3: ("B", "#b23a48")}
    for frame in range(6):
        x = gap + frame * (tile + gap)
        card.paste(_ink_image(prompt[frame]), (x, top))
        color = source_marks.get(frame, ("", "#b8c4c8"))[1]
        draw.rectangle((x - 1, top - 1, x + tile, top + tile), outline=color, width=2)
        if frame in source_marks:
            draw.text(
                (x + tile // 2 - 3, 8),
                source_marks[frame][0],
                fill=color,
                font=font,
            )
    answer_x = gap + 6 * (tile + gap)
    draw.line((answer_x - gap // 2, top - 4, answer_x - gap // 2, top + tile + 4), fill="#61767e", width=2)
    card.paste(_ink_image(generated), (answer_x, top))
    draw.rectangle(
        (answer_x - 1, top - 1, answer_x + tile, top + tile),
        outline="#24373f",
        width=2,
    )
    return card


def save_review_pages(cards: list[Image.Image], root: Path) -> None:
    cards_per_page = 12
    columns = 2
    rows = cards_per_page // columns
    margin = 16
    vertical_gap = 8
    for page_index in range((len(cards) + cards_per_page - 1) // cards_per_page):
        page_cards = cards[
            page_index * cards_per_page : (page_index + 1) * cards_per_page
        ]
        width = columns * page_cards[0].width + (columns + 1) * margin
        height = rows * page_cards[0].height + (rows + 1) * vertical_gap
        page = Image.new("RGB", (width, height), "#e9eff1")
        for index, card in enumerate(page_cards):
            x = margin + (index % columns) * (card.width + margin)
            y = vertical_gap + (index // columns) * (card.height + vertical_gap)
            page.paste(card, (x, y))
        page.save(root / f"review_page_{page_index + 1:02d}.png")


def main() -> None:
    args = parse_args()
    if args.precision != "bf16":
        raise ValueError("V23 blinded review requires BF16")
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty V23 review output: {output_dir}")

    paired_sha256 = file_sha256(args.paired_audit)
    if paired_sha256 != EXPECTED_PAIRED_AUDIT_SHA256:
        raise ValueError("V23 paired-audit report hash differs")
    paired = json.loads(Path(args.paired_audit).read_text(encoding="utf-8"))
    if paired.get("architecture") != AUDIT_ARCHITECTURE:
        raise ValueError("V23 blinded review requires the paired-audit report")
    if not paired.get("paired_gate_passed") or not paired.get(
        "blinded_review_permitted"
    ):
        raise ValueError("V23 paired gate did not permit blinded review")
    if paired.get("frozen_images_instantiated") != 0:
        raise ValueError("V23 paired audit broke the frozen seal")

    candidate_sha256 = file_sha256(args.candidate)
    if paired["checkpoint_sha256"][RELATION_AWARE_ROUTE] != candidate_sha256:
        raise ValueError("V23 candidate differs from the paired audit")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if candidate.get("route_mode") != RELATION_AWARE_ROUTE:
        raise ValueError("V23 review candidate has the wrong route")
    if candidate.get("smoke_only") or candidate.get("best_development") is None:
        raise ValueError("V23 review candidate is not selected evidence")

    if file_sha256(args.pvf_checkpoint) != EXPECTED_PVF_SHA256:
        raise ValueError("V23 review PVF file hash differs")
    if file_sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("V23 review manifest file hash differs")
    canonicalizer_sha256 = file_sha256(args.canonicalizer_checkpoint)
    if canonicalizer_sha256 != EXPECTED_CANONICALIZER_SHA256:
        raise ValueError("V23 review canonicalizer file hash differs")
    canonicalizer_checkpoint = torch.load(
        args.canonicalizer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_canonicalizer_checkpoint(
        canonicalizer_checkpoint,
        checkpoint_sha256=canonicalizer_sha256,
    )

    seed_everything(FIXED_OPTIMIZATION_ARGUMENTS["seed"])
    device = choose_device(args.device)
    pvf, _ = load_pvf(args.pvf_checkpoint, device)
    canonicalizer = VisualCanonicalizer()
    canonicalizer.load_state_dict(canonicalizer_checkpoint["canonicalizer"])
    config = relation_circuit_config_from_payload(candidate["model_config"])
    model = VisualRelationCircuit(config, pvf.retina, canonicalizer).to(device).eval()
    load_relation_state(model, candidate["relation"])
    del pvf

    records = load_visual_grammar_manifest(args.manifest)
    bank = build_relation_character_bank(
        records, bank_size=FIXED_MODEL_ARGUMENTS["bank_size"]
    )
    partitions = split_relation_characters(bank, salt=PARTITION_SALT)
    partition = relation_partition_receipt(partitions, salt=PARTITION_SALT)
    for key, expected in EXPECTED_PARTITION.items():
        if partition.get(key) != expected:
            raise ValueError(f"V23 review partition differs for {key}")
    dataset = VisualRelationEpisodeDataset(
        partitions["development"],
        split="development",
        length=4_096,
        config=VisualRelationEpisodeConfig(),
        seed=REVIEW_SEED,
    )
    selected = select_review_items(dataset)
    collated = visual_relation_collate([item for _, item in selected])
    prompts = collated["prompt"].to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        generated = model(prompts)[:, 0].cpu()

    cards_dir = output_dir / "cards"
    pages_dir = output_dir / "pages"
    sealed_dir = output_dir / "sealed"
    cards_dir.mkdir(parents=True)
    pages_dir.mkdir(parents=True)
    sealed_dir.mkdir(parents=True)
    cards: list[Image.Image] = []
    answer_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, str | None]] = []
    for position, ((dataset_index, item), output) in enumerate(
        zip(selected, generated)
    ):
        review_id = _opaque_identifier(dataset_index)
        card = render_review_card(review_id, item["prompt"], output)
        card.save(cards_dir / f"{position + 1:02d}_{review_id}.png")
        cards.append(card)
        answer_rows.append(
            {
                "review_id": review_id,
                "correct_source": "A" if item["metadata"]["target_index"] == 0 else "B",
                "heldout_combination": bool(
                    item["metadata"]["heldout_combination"]
                ),
                "dataset_index": dataset_index,
            }
        )
        review_rows.append({"review_id": review_id, "response": None})
    save_review_pages(cards, pages_dir)

    answer_path = sealed_dir / "answer_key.json"
    answer_path.write_text(
        json.dumps(answer_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    answer_path.chmod(0o600)
    response_path = output_dir / "responses.json"
    response_path.write_text(
        json.dumps(review_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    receipt = {
        "architecture": ARCHITECTURE,
        "review_seed": REVIEW_SEED,
        "review_cases": REVIEW_CASES,
        "heldout_cases": REVIEW_HELDOUT_CASES,
        "seen_cases": REVIEW_SEEN_CASES,
        "candidate_sha256": candidate_sha256,
        "paired_audit_sha256": paired_sha256,
        "canonicalizer_sha256": canonicalizer_sha256,
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "answer_key_sha256": file_sha256(answer_path),
        "response_template_sha256": file_sha256(response_path),
        "cards": len(cards),
        "pages": len(list(pages_dir.glob("*.png"))),
        "contains_text_transcriptions": False,
        "contains_target_images": False,
        "contains_correctness_labels": False,
        "frozen_images_instantiated": 0,
        "review_scored": False,
        "frozen_evaluation_permitted": False,
    }
    receipt_path = output_dir / "review_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
