#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ilm.visual_lm.ink_jepa_data import load_visual_grammar_manifest
from ilm.visual_lm.visual_packet_data import (
    PARTITION_SALT,
    VisualPacketEpisodeConfig,
    VisualPacketEpisodeDataset,
    build_packet_character_bank,
    packet_partition_receipt,
    split_packet_characters,
    visual_packet_collate,
)
from ilm.visual_lm.visual_packet_stream import PACKET_AWARE_ROUTE
from scripts.eval_visual_packet_stream_development_v24 import (
    AUDIT_ARCHITECTURE,
    _load_model,
    validate_selected_arm,
)
from scripts.train_visual_packet_stream_v24 import (
    DEFAULT_CANONICALIZER_CHECKPOINT,
    DEFAULT_PVF_CHECKPOINT,
    DEFAULT_RELATION_CHECKPOINT,
    EXPECTED_CANONICALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    EXPECTED_RELATION_SHA256,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    validate_canonicalizer_checkpoint,
    validate_relation_checkpoint,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


ARCHITECTURE = "visual-packet-reread-stream-v24-opaque-review-pack"
EXPECTED_PAIRED_AUDIT_SHA256 = (
    "0b2cb99228539e2655270fbb9ff28ed0dd29ffe95b8d041a26a08a0c82c722e9"
)
REVIEW_SEED = FIXED_OPTIMIZATION_ARGUMENTS["dataset_seed"] + 3_000_006
REVIEW_CASES = 48
REVIEW_HELDOUT_LENGTH_CASES = 12
REVIEW_SEEN_LENGTH_CASES = REVIEW_CASES - REVIEW_HELDOUT_LENGTH_CASES
REVIEW_DATASET_LENGTH = 4_096
REVIEW_BATCH_SIZE = REVIEW_CASES
DEFAULT_EVIDENCE_ROOT = Path("artifacts/visual_packet_stream_v24_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the sealed V24 48-case opaque visual review pack."
    )
    parser.add_argument(
        "--candidate",
        default=str(
            DEFAULT_EVIDENCE_ROOT
            / PACKET_AWARE_ROUTE
            / "checkpoint_selected_development.pt"
        ),
    )
    parser.add_argument(
        "--paired-audit",
        default=(
            "artifacts/visual_packet_stream_v24_paired_audit/"
            "paired_development_audit.json"
        ),
    )
    parser.add_argument("--pvf-checkpoint", default=DEFAULT_PVF_CHECKPOINT)
    parser.add_argument(
        "--canonicalizer-checkpoint", default=DEFAULT_CANONICALIZER_CHECKPOINT
    )
    parser.add_argument("--relation-checkpoint", default=DEFAULT_RELATION_CHECKPOINT)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument(
        "--out", default="artifacts/visual_packet_stream_v24_opaque_review"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def _opaque_identifier(dataset_index: int) -> str:
    payload = f"{REVIEW_SEED}\0{dataset_index}".encode()
    return f"P-{hashlib.sha256(payload).hexdigest()[:10].upper()}"


def select_review_items(
    dataset: VisualPacketEpisodeDataset,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    heldout = 0
    seen = 0
    for index in range(len(dataset)):
        item = dataset[index]
        is_heldout = bool(item["metadata"]["heldout_length"])
        if is_heldout and heldout < REVIEW_HELDOUT_LENGTH_CASES:
            selected.append((index, item))
            heldout += 1
        elif not is_heldout and seen < REVIEW_SEEN_LENGTH_CASES:
            selected.append((index, item))
            seen += 1
        if len(selected) == REVIEW_CASES:
            break
    if heldout != REVIEW_HELDOUT_LENGTH_CASES or seen != REVIEW_SEEN_LENGTH_CASES:
        raise ValueError("V24 could not construct the fixed opaque-review balance")
    return selected


def _ink_image(tensor: torch.Tensor, size: int) -> Image.Image:
    array = (
        ((1.0 - tensor.detach().float().cpu().clamp(0, 1)[0]).numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    return (
        Image.fromarray(array)
        .convert("RGB")
        .resize((size, size), Image.Resampling.NEAREST)
    )


def _review_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _pair_packet_indices(metadata: dict[str, Any]) -> tuple[int, int]:
    indices = tuple(
        index for index, kind in enumerate(metadata["packet_kinds"]) if kind == "pair"
    )
    if len(indices) != 2:
        raise ValueError("V24 review item does not contain exactly two pair packets")
    return indices


def _correct_choice(item: dict[str, Any], pair_indices: Sequence[int]) -> str:
    target_source = item["oracle_reference"]
    matches = [
        torch.equal(item["prompt"][packet_index * 3 + 2], target_source)
        for packet_index in pair_indices
    ]
    if matches.count(True) != 1:
        raise ValueError("V24 review target does not match one visible source glyph")
    return "A" if matches[0] else "B"


def render_review_card(
    review_id: str,
    prompt: torch.Tensor,
    generated: torch.Tensor,
    *,
    pair_packet_indices: Sequence[int],
) -> Image.Image:
    if prompt.ndim != 4 or prompt.shape[0] % 3:
        raise ValueError("V24 review prompt must be a packet-aligned image stream")
    if tuple(generated.shape) != (2, 1, 32, 32):
        raise ValueError("V24 review output must contain exactly two image frames")
    if len(pair_packet_indices) != 2:
        raise ValueError("V24 review card requires two pair choices")

    packet_count = prompt.shape[0] // 3
    tile = 40
    frame_gap = 3
    packet_gap = 10
    margin = 12
    sequence_top = 35
    packet_width = 3 * tile + 2 * frame_gap
    sequence_width = packet_count * packet_width + (packet_count - 1) * packet_gap
    choice_tile = 58
    choice_top = sequence_top + tile + 24
    choice_width = 2 * (2 * choice_tile + 44) + 2 * (choice_tile + 28) + 52
    width = max(sequence_width, choice_width) + 2 * margin
    height = choice_top + choice_tile + 18
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    small_font = _review_font(13)
    marker_font = _review_font(17)
    draw.text((margin, 8), review_id, fill="#24373f", font=small_font)

    choice_colors = ("#007f8b", "#b23a48")
    packet_choice = {
        int(packet_index): choice
        for choice, packet_index in zip(("A", "B"), pair_packet_indices, strict=True)
    }
    for packet_index in range(packet_count):
        packet_x = margin + packet_index * (packet_width + packet_gap)
        choice = packet_choice.get(packet_index)
        outline = (
            choice_colors[0]
            if choice == "A"
            else choice_colors[1]
            if choice == "B"
            else "#aab8bd"
        )
        for offset in range(3):
            x = packet_x + offset * (tile + frame_gap)
            card.paste(
                _ink_image(prompt[packet_index * 3 + offset], tile), (x, sequence_top)
            )
            draw.rectangle(
                (x - 1, sequence_top - 1, x + tile, sequence_top + tile),
                outline=outline,
                width=2 if choice is not None else 1,
            )
        if choice is not None:
            draw.text(
                (packet_x + packet_width // 2 - 5, sequence_top - 23),
                choice,
                fill=outline,
                font=marker_font,
            )

    x = margin
    for choice, packet_index, color in zip(
        ("A", "B"), pair_packet_indices, choice_colors, strict=True
    ):
        draw.text((x, choice_top + 18), choice, fill=color, font=marker_font)
        x += 24
        for offset in (1, 2):
            card.paste(
                _ink_image(prompt[packet_index * 3 + offset], choice_tile),
                (x, choice_top),
            )
            draw.rectangle(
                (x - 1, choice_top - 1, x + choice_tile, choice_top + choice_tile),
                outline=color,
                width=2,
            )
            x += choice_tile + 6
        x += 18

    draw.line(
        (x, choice_top - 5, x, choice_top + choice_tile + 5),
        fill="#61767e",
        width=2,
    )
    x += 22
    for frame_index in range(2):
        draw.text(
            (x + choice_tile // 2 - 4, choice_top - 23),
            str(frame_index + 1),
            fill="#24373f",
            font=marker_font,
        )
        card.paste(_ink_image(generated[frame_index], choice_tile), (x, choice_top))
        draw.rectangle(
            (x - 1, choice_top - 1, x + choice_tile, choice_top + choice_tile),
            outline="#24373f",
            width=2,
        )
        x += choice_tile + 16
    return card


def save_review_pages(cards: Sequence[Image.Image], root: Path) -> None:
    cards_per_page = 12
    columns = 2
    rows = cards_per_page // columns
    margin = 14
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
        raise ValueError("V24 opaque review requires BF16")
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty V24 review output: {output_dir}")

    paired_sha256 = file_sha256(args.paired_audit)
    if paired_sha256 != EXPECTED_PAIRED_AUDIT_SHA256:
        raise ValueError("V24 paired-audit report hash differs")
    paired = json.loads(Path(args.paired_audit).read_text(encoding="utf-8"))
    if paired.get("architecture") != AUDIT_ARCHITECTURE:
        raise ValueError("V24 opaque review requires the paired-audit report")
    if not paired.get("paired_gate_passed") or not paired.get(
        "opaque_review_permitted"
    ):
        raise ValueError("V24 paired gate did not permit opaque review")
    if paired.get("frozen_images_instantiated") != 0:
        raise ValueError("V24 paired audit broke the frozen seal")

    candidate_sha256 = file_sha256(args.candidate)
    if paired["checkpoint_sha256"][PACKET_AWARE_ROUTE] != candidate_sha256:
        raise ValueError("V24 candidate differs from the paired audit")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    validate_selected_arm(candidate, route_mode=PACKET_AWARE_ROUTE)

    input_hashes = {
        "pvf": file_sha256(args.pvf_checkpoint),
        "canonicalizer": file_sha256(args.canonicalizer_checkpoint),
        "relation": file_sha256(args.relation_checkpoint),
        "manifest": file_sha256(args.manifest),
    }
    expected_hashes = {
        "pvf": EXPECTED_PVF_SHA256,
        "canonicalizer": EXPECTED_CANONICALIZER_SHA256,
        "relation": EXPECTED_RELATION_SHA256,
        "manifest": EXPECTED_MANIFEST_SHA256,
    }
    for name, expected in expected_hashes.items():
        if input_hashes[name] != expected:
            raise ValueError(f"V24 review {name} file hash differs")
    canonicalizer_checkpoint = torch.load(
        args.canonicalizer_checkpoint, map_location="cpu", weights_only=False
    )
    validate_canonicalizer_checkpoint(
        canonicalizer_checkpoint,
        checkpoint_sha256=input_hashes["canonicalizer"],
    )
    relation_checkpoint = torch.load(
        args.relation_checkpoint, map_location="cpu", weights_only=False
    )
    validate_relation_checkpoint(
        relation_checkpoint, checkpoint_sha256=input_hashes["relation"]
    )

    seed_everything(FIXED_OPTIMIZATION_ARGUMENTS["seed"])
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    pvf, _ = load_pvf(args.pvf_checkpoint, device)
    model = _load_model(
        candidate,
        retina=pvf.retina,
        canonicalizer_state=canonicalizer_checkpoint["canonicalizer"],
        relation_checkpoint=relation_checkpoint,
        device=device,
    )
    del pvf

    records = load_visual_grammar_manifest(args.manifest)
    bank = build_packet_character_bank(
        records, bank_size=FIXED_MODEL_ARGUMENTS["bank_size"]
    )
    partitions = split_packet_characters(bank, salt=PARTITION_SALT)
    partition = packet_partition_receipt(partitions, salt=PARTITION_SALT)
    for key, expected in EXPECTED_PARTITION.items():
        if partition.get(key) != expected:
            raise ValueError(f"V24 review partition differs for {key}")
    dataset = VisualPacketEpisodeDataset(
        partitions["development"],
        split="development",
        length=REVIEW_DATASET_LENGTH,
        config=VisualPacketEpisodeConfig(),
        seed=REVIEW_SEED,
    )
    selected = select_review_items(dataset)
    collated = visual_packet_collate([item for _, item in selected])
    prompts = collated["prompt"].to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        generated = model(prompts).cpu()

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
        zip(selected, generated, strict=True)
    ):
        review_id = _opaque_identifier(dataset_index)
        pair_indices = _pair_packet_indices(item["metadata"])
        card = render_review_card(
            review_id,
            item["prompt"],
            output,
            pair_packet_indices=pair_indices,
        )
        card.save(cards_dir / f"{position + 1:02d}_{review_id}.png")
        cards.append(card)
        correct_choice = _correct_choice(item, pair_indices)
        answer_rows.append(
            {
                "review_id": review_id,
                "correct_frame1_source": correct_choice,
                "correct_frame2_label": correct_choice,
                "heldout_length": bool(item["metadata"]["heldout_length"]),
                "dataset_index": dataset_index,
            }
        )
        review_rows.append(
            {
                "review_id": review_id,
                "frame1_response": None,
                "frame2_response": None,
            }
        )
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
        "heldout_length_cases": REVIEW_HELDOUT_LENGTH_CASES,
        "seen_length_cases": REVIEW_SEEN_LENGTH_CASES,
        "answer_frames": 2,
        "candidate_sha256": candidate_sha256,
        "paired_audit_sha256": paired_sha256,
        "canonicalizer_sha256": input_hashes["canonicalizer"],
        "relation_sha256": input_hashes["relation"],
        "pvf_sha256": input_hashes["pvf"],
        "manifest_sha256": input_hashes["manifest"],
        "answer_key_sha256": file_sha256(answer_path),
        "response_template_sha256": file_sha256(response_path),
        "cards": len(cards),
        "pages": len(list(pages_dir.glob("*.png"))),
        "contains_text_transcriptions": False,
        "contains_target_images": False,
        "contains_correctness_labels": False,
        "contains_pair_choice_markers": True,
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
