#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import RetinalRenderConfig, load_visual_grammar_manifest
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    render_glyph_fovea,
    visual_saccade_collate,
)
from ilm.visual_lm.saccade_lm import (
    VisualSaccadeLM,
    visual_saccade_config_from_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-native next-glyph language prediction against a visual glyph bank."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/visual_grammar/chinese_wikisource_public_domain.jsonl")
    parser.add_argument("--out", default="artifacts/visual_saccade_evaluation")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--prototype-views", type=int, default=4)
    parser.add_argument("--evaluation-samples", type=int, default=4096)
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_han(character: str) -> bool:
    value = ord(character)
    return any(
        lower <= value <= upper
        for lower, upper in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x323AF),
        )
    )


def visible_writing(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def language_statistics(
    records: Sequence[Any],
    bank_size: int,
) -> tuple[list[str], Counter[str], dict[str, str]]:
    unigram: Counter[str] = Counter()
    bigram: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        writing = visible_writing(record.text)
        for character in writing:
            if is_han(character):
                unigram[character] += 1
        for previous, target in zip(writing, writing[1:]):
            if is_han(target):
                bigram[previous][target] += 1
    bank = [character for character, _ in unigram.most_common(bank_size)]
    bank_set = set(bank)
    bigram_best: dict[str, str] = {}
    for previous, counts in bigram.items():
        next_character = next(
            (character for character, _ in counts.most_common() if character in bank_set),
            None,
        )
        if next_character is not None:
            bigram_best[previous] = next_character
    if len(bank) != bank_size:
        raise ValueError(f"requested {bank_size} Han glyphs, but corpus supplied only {len(bank)}")
    return bank, unigram, bigram_best


def render_config_from_checkpoint(checkpoint: dict[str, Any]) -> RetinalRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = True
    return RetinalRenderConfig(**payload)


def load_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    random_weights: bool,
    seed: int,
) -> VisualSaccadeLM:
    if random_weights:
        torch.manual_seed(seed + 91_337)
    model = VisualSaccadeLM(visual_saccade_config_from_payload(checkpoint["model_config"]))
    if not random_weights:
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def prototype_bank(
    model: VisualSaccadeLM,
    characters: Sequence[str],
    *,
    render_config: RetinalRenderConfig,
    views: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    seed: int,
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    owners: list[int] = []
    encoded: list[torch.Tensor] = []
    encoded_owners: list[int] = []

    def flush() -> None:
        if not images:
            return
        batch = torch.stack(images).to(device, non_blocking=True)
        with autocast_context(device, precision):
            features = model.target_retina(batch).float().cpu()
        encoded.append(features)
        encoded_owners.extend(owners)
        images.clear()
        owners.clear()

    for owner, character in enumerate(characters):
        for view in range(views):
            images.append(
                render_glyph_fovea(
                    character,
                    render_config=render_config,
                    fovea_size=model.config.fovea_size,
                    variant=seed + owner * 10_007 + view * 1_000_003,
                )
            )
            owners.append(owner)
            if len(images) >= batch_size:
                flush()
    flush()
    features = torch.cat(encoded)
    owner_tensor = torch.tensor(encoded_owners, dtype=torch.long)
    bank = torch.zeros(len(characters), features.shape[-1])
    bank.index_add_(0, owner_tensor, features)
    counts = torch.bincount(owner_tensor, minlength=len(characters)).float()[:, None]
    return F.normalize(bank / counts.clamp_min(1.0), dim=-1).to(device)


def rank_metrics(similarity: torch.Tensor, expected: torch.Tensor) -> dict[str, torch.Tensor]:
    target_score = similarity.gather(1, expected[:, None])
    rank = 1 + (similarity > target_score).sum(dim=1)
    return {
        "top1": (rank == 1).float(),
        "top5": (rank <= 5).float(),
        "mrr": 1.0 / rank.float(),
        "target_score": target_score[:, 0],
    }


@torch.inference_mode()
def evaluate_branch(
    name: str,
    model: VisualSaccadeLM,
    loader: DataLoader,
    bank: torch.Tensor,
    bank_characters: Sequence[str],
    unigram: Counter[str],
    bigram_best: dict[str, str],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    bank_index = {character: index for index, character in enumerate(bank_characters)}
    sums: dict[str, float] = defaultdict(float)
    examples = 0
    ink_true_positive = 0.0
    ink_predicted = 0.0
    ink_target = 0.0
    qualitative: list[dict[str, str]] = []
    started = time.perf_counter()
    for batch in loader:
        eligible = [
            index
            for index, metadata in enumerate(batch["metadata"])
            if metadata.get("target_character") in bank_index
        ]
        if not eligible:
            continue
        context = batch["context"][eligible].to(device, non_blocking=True)
        target_ink = batch["target_ink"][eligible, -1].to(device, non_blocking=True)
        target_reference = batch["target_reference"][eligible, -1].to(device, non_blocking=True)
        metadata = [batch["metadata"][index] for index in eligible]
        expected = torch.tensor(
            [bank_index[item["target_character"]] for item in metadata],
            dtype=torch.long,
            device=device,
        )
        with autocast_context(device, precision):
            full = model.predict(context)
            last = model.predict(context[:, -1:])
            oracle_visual = model.target_retina(target_reference).float()
        full_visual = F.normalize(full["predicted_visual"][:, -1].float(), dim=-1)
        last_visual = F.normalize(last["predicted_visual"][:, -1].float(), dim=-1)
        full_similarity = model.score_visual_candidates(full, bank, position=-1)
        last_similarity = model.score_visual_candidates(last, bank, position=-1)
        oracle_similarity = F.normalize(oracle_visual, dim=-1) @ bank.transpose(0, 1)
        full_metrics = rank_metrics(full_similarity, expected)
        last_metrics = rank_metrics(last_similarity, expected)
        oracle_metrics = rank_metrics(oracle_similarity, expected)
        for key, values in full_metrics.items():
            sums[f"full_{key}"] += float(values.sum())
        for key, values in last_metrics.items():
            sums[f"last_{key}"] += float(values.sum())
        for key, values in oracle_metrics.items():
            sums[f"oracle_{key}"] += float(values.sum())
        sums["prediction_change"] += float(
            (1.0 - F.cosine_similarity(full_visual, last_visual, dim=-1)).sum()
        )
        most_frequent = bank_characters[0]
        sums["unigram_top1"] += sum(item["target_character"] == most_frequent for item in metadata)
        sums["bigram_top1"] += sum(
            bigram_best.get(item["previous_character"]) == item["target_character"] for item in metadata
        )
        predicted_ink = full["predicted_ink_logits"][:, -1].float().sigmoid() >= 0.5
        target_binary = target_ink >= 0.5
        ink_true_positive += float((predicted_ink & target_binary).sum())
        ink_predicted += float(predicted_ink.sum())
        ink_target += float(target_binary.sum())
        if len(qualitative) < 24:
            predictions = full_similarity.argmax(dim=1)
            for item, prediction in zip(metadata, predictions):
                qualitative.append(
                    {
                        "previous": item["previous_character"],
                        "target": item["target_character"],
                        "predicted": bank_characters[int(prediction)],
                    }
                )
                if len(qualitative) >= 24:
                    break
        examples += len(eligible)
    report = {key: value / max(1, examples) for key, value in sums.items()}
    for prefix in ("full", "last", "oracle"):
        for metric in ("top1", "top5", "mrr", "target_score"):
            report.setdefault(f"{prefix}_{metric}", 0.0)
    for metric in ("prediction_change", "unigram_top1", "bigram_top1"):
        report.setdefault(metric, 0.0)
    report.update(
        {
            "branch": name,
            "examples": examples,
            "ink_f1": 2.0 * ink_true_positive / max(1.0, ink_predicted + ink_target),
            "elapsed_seconds": time.perf_counter() - started,
            "qualitative": qualitative,
        }
    )
    return report


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") not in {
        "visual-saccade-language-model-v1",
        "visual-saccade-language-model-v2",
    }:
        raise ValueError("checkpoint is not a visual saccade language model")
    records = load_visual_grammar_manifest(args.manifest)
    characters, unigram, bigram_best = language_statistics(records, args.bank_size)
    bank_digest = hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()
    render_config = render_config_from_checkpoint(checkpoint)
    model_config = visual_saccade_config_from_payload(checkpoint["model_config"])
    sequence_length = int(checkpoint.get("arguments", {}).get("sequence_length", 48))
    dataset = VisualSaccadeDataset(
        records,
        render_config=render_config,
        spec=SaccadeSequenceSpec(sequence_length=sequence_length, fovea_size=model_config.fovea_size),
        split="validation",
        validation_fraction=args.validation_fraction,
        length=args.evaluation_samples,
        seed=args.seed + 7_000_003,
        expose_evaluation_labels=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=visual_saccade_collate,
    )
    reports: dict[str, Any] = {}
    for offset, (name, random_weights) in enumerate((("pretrained", False), ("random", True))):
        model = load_model(
            checkpoint,
            device=device,
            random_weights=random_weights,
            seed=args.seed + offset * 100_003,
        )
        bank = prototype_bank(
            model,
            characters,
            render_config=render_config,
            views=args.prototype_views,
            batch_size=max(args.batch_size, 64),
            device=device,
            precision=args.precision,
            seed=args.seed + offset * 200_003,
        )
        report = evaluate_branch(
            name,
            model,
            loader,
            bank,
            characters,
            unigram,
            bigram_best,
            device=device,
            precision=args.precision,
        )
        reports[name] = report
        print(json.dumps({"stage": "branch_complete", **report}, ensure_ascii=False), flush=True)
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pretrained = reports["pretrained"]
    random_report = reports["random"]
    acceptance = {
        "beats_random": pretrained["full_top1"] > random_report["full_top1"],
        "beats_unigram": pretrained["full_top1"] > pretrained["unigram_top1"],
        "beats_bigram": pretrained["full_top1"] > pretrained["bigram_top1"],
        "uses_longer_context": pretrained["full_top1"] > pretrained["last_top1"],
        "renders_nonzero_ink": pretrained["ink_f1"] > 0.0,
    }
    final = {
        "architecture": "visual-saccade-fixed-glyph-bank-evaluation-v1",
        "checkpoint": args.checkpoint,
        "bank_size": len(characters),
        "bank_sha256": bank_digest,
        "prototype_views": args.prototype_views,
        "chance_top1": 1.0 / len(characters),
        "evaluator_used_labels_for_scoring_only": True,
        "student_received_labels": False,
        "student_received_token_ids": False,
        "student_received_ocr": False,
        "branches": reports,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "glyph_bank.json").write_text(
        json.dumps(
            [
                {"rank": index + 1, "character": character, "frequency": unigram[character]}
                for index, character in enumerate(characters)
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "complete", **final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
