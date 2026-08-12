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
from PIL import Image
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import RetinalRenderConfig, load_visual_grammar_manifest
from ilm.visual_lm.retinal_flow_lm import (
    RetinalFlowLanguageModel,
    retinal_flow_config_from_payload,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    render_glyph_fovea,
    visual_saccade_collate,
)


ARCHITECTURE = "retinal-flow-language-model-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate visual energy and sampled image output without exposing labels to the model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/visual_grammar/chinese_wikisource_public_domain.jsonl")
    parser.add_argument("--out", default="artifacts/retinal_flow_evaluation")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--prototype-views", type=int, default=4)
    parser.add_argument("--evaluation-samples", type=int, default=4096)
    parser.add_argument("--generation-contexts", type=int, default=192)
    parser.add_argument("--samples-per-context", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
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
    if len(bank) != bank_size:
        raise ValueError(f"requested {bank_size} Han glyphs, but corpus supplied only {len(bank)}")
    bank_set = set(bank)
    bigram_best: dict[str, str] = {}
    for previous, counts in bigram.items():
        choice = next(
            (character for character, _ in counts.most_common() if character in bank_set),
            None,
        )
        if choice is not None:
            bigram_best[previous] = choice
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
) -> RetinalFlowLanguageModel:
    if random_weights:
        torch.manual_seed(seed + 91_337)
    model = RetinalFlowLanguageModel(
        retinal_flow_config_from_payload(checkpoint["model_config"])
    )
    if not random_weights:
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def prototype_bank(
    model: RetinalFlowLanguageModel,
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
    expected_owners = torch.arange(len(characters)).repeat_interleave(views)
    if not torch.equal(owner_tensor, expected_owners):
        raise RuntimeError("prototype rendering order is not owner-major")
    bank = features.reshape(len(characters), views, features.shape[-1])
    return F.normalize(bank, dim=-1).to(device)


def aggregate_view_scores(scores: torch.Tensor) -> torch.Tensor:
    """Marginalize rendered views without averaging before a nonlinear scorer."""

    return torch.logsumexp(scores, dim=-1) - torch.tensor(
        scores.shape[-1],
        device=scores.device,
        dtype=scores.dtype,
    ).log()


def rank_metrics(scores: torch.Tensor, expected: torch.Tensor) -> dict[str, torch.Tensor]:
    target_score = scores.gather(1, expected[:, None])
    rank = 1 + (scores > target_score).sum(dim=1)
    return {
        "top1": (rank == 1).float(),
        "top5": (rank <= 5).float(),
        "mrr": 1.0 / rank.float(),
        "target_score": target_score[:, 0],
    }


def save_generation_sheet(
    branch: str,
    offset: int,
    context: torch.Tensor,
    target: torch.Tensor,
    generated: torch.Tensor,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(min(4, context.shape[0])):
        cells = [
            *context[index, -8:, 0].float().cpu().clamp(0, 1),
            target[index, 0].float().cpu().clamp(0, 1),
            *generated[index, :, 0].float().cpu().clamp(0, 1),
        ]
        size = cells[0].shape[-1]
        canvas = torch.zeros(size, size * len(cells))
        for cell_index, cell in enumerate(cells):
            canvas[:, cell_index * size : (cell_index + 1) * size] = cell
        image = (255.0 * (1.0 - canvas)).round().byte().numpy()
        Image.fromarray(image, mode="L").save(
            root / f"{branch}_{offset + index:05d}_context-target-samples.png",
            optimize=True,
        )


@torch.inference_mode()
def evaluate_branch(
    name: str,
    model: RetinalFlowLanguageModel,
    loader: DataLoader,
    bank: torch.Tensor,
    bank_characters: Sequence[str],
    unigram: Counter[str],
    bigram_best: dict[str, str],
    *,
    device: torch.device,
    precision: str,
    generation_contexts: int,
    samples_per_context: int,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    sample_root: Path,
) -> dict[str, Any]:
    bank_index = {character: index for index, character in enumerate(bank_characters)}
    sums: dict[str, float] = defaultdict(float)
    examples = 0
    generated_examples = 0
    qualitative: list[dict[str, Any]] = []
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
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
            flat_bank = bank.flatten(0, 1)
            full_scores = aggregate_view_scores(
                model.score_visual_candidates(full, flat_bank, position=-1).reshape(
                    context.shape[0],
                    bank.shape[0],
                    bank.shape[1],
                )
            )
            last_scores = aggregate_view_scores(
                model.score_visual_candidates(last, flat_bank, position=-1).reshape(
                    context.shape[0],
                    bank.shape[0],
                    bank.shape[1],
                )
            )
        flat_bank = bank.flatten(0, 1)
        oracle_scores = (
            F.normalize(oracle_visual, dim=-1) @ flat_bank.transpose(0, 1)
        ).reshape(context.shape[0], bank.shape[0], bank.shape[1]).amax(dim=-1)
        full_metrics = rank_metrics(full_scores, expected)
        last_metrics = rank_metrics(last_scores, expected)
        oracle_metrics = rank_metrics(oracle_scores, expected)
        for key, values in full_metrics.items():
            sums[f"full_{key}"] += float(values.sum())
        for key, values in last_metrics.items():
            sums[f"last_{key}"] += float(values.sum())
        for key, values in oracle_metrics.items():
            sums[f"oracle_{key}"] += float(values.sum())
        sums["context_target_score_gain"] += float(
            (full_metrics["target_score"] - last_metrics["target_score"]).sum()
        )
        most_frequent = bank_characters[0]
        sums["unigram_top1"] += sum(
            item["target_character"] == most_frequent for item in metadata
        )
        sums["bigram_top1"] += sum(
            bigram_best.get(item["previous_character"]) == item["target_character"]
            for item in metadata
        )

        if generated_examples < generation_contexts:
            count = min(len(eligible), generation_contexts - generated_examples)
            generation_context = context[:count]
            generation_target = target_ink[:count]
            with autocast_context(device, precision):
                generated = model.sample_next(
                    generation_context,
                    samples_per_context=samples_per_context,
                    steps=sample_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                generated_visual = model.target_retina(
                    generated.reshape(-1, *generated.shape[2:])
                ).float().reshape(count, samples_per_context, -1)
            normalized_generated = F.normalize(generated_visual, dim=-1)
            generated_bank_scores = torch.einsum(
                "bsd,nvd->bsnv",
                normalized_generated,
                bank,
            ).amax(dim=-1)
            generated_identity = generated_bank_scores.argmax(dim=-1)
            expected_generation = expected[:count]
            sample_hit = generated_identity.eq(expected_generation[:, None]).any(dim=1)
            condition = full["condition"][:count, -1]
            with autocast_context(device, precision):
                generated_energy = model.energy(condition, generated_visual)
            selected = generated_energy.argmax(dim=1)
            selected_identity = generated_identity[
                torch.arange(count, device=device),
                selected,
            ]
            target_visual = F.normalize(oracle_visual[:count], dim=-1)
            generated_target_cosine = torch.einsum(
                "bsd,bd->bs",
                normalized_generated,
                target_visual,
            )
            if count > 1:
                shuffled_target = target_visual.roll(1, dims=0)
                shuffled_target_cosine = torch.einsum(
                    "bsd,bd->bs",
                    normalized_generated,
                    shuffled_target,
                )
                context_cosine_gain = (
                    generated_target_cosine.amax(dim=1)
                    - shuffled_target_cosine.amax(dim=1)
                )
            else:
                context_cosine_gain = torch.zeros(
                    count,
                    device=device,
                    dtype=generated_target_cosine.dtype,
                )
            binary = generated >= 0.5
            target_binary = generation_target[:, None] >= 0.5
            true_positive = (binary & target_binary).sum(dim=(2, 3, 4)).float()
            pixel_f1 = 2.0 * true_positive / (
                binary.sum(dim=(2, 3, 4)) + target_binary.sum(dim=(2, 3, 4))
            ).clamp_min(1)
            sums["generated_sample_hit"] += float(sample_hit.sum())
            sums["generated_energy_reranked_top1"] += float(
                selected_identity.eq(expected_generation).sum()
            )
            sums["generated_best_target_cosine"] += float(
                generated_target_cosine.amax(dim=1).sum()
            )
            sums["generated_context_cosine_gain"] += float(context_cosine_gain.sum())
            sums["generated_best_pixel_f1"] += float(pixel_f1.amax(dim=1).sum())
            sums["generated_ink_fraction"] += float(generated.mean(dim=(1, 2, 3, 4)).sum())
            sums["generated_pixel_std"] += float(
                generated.float().flatten(1).std(dim=1).sum()
            )
            save_generation_sheet(
                name,
                generated_examples,
                generation_context,
                generation_target,
                generated,
                sample_root,
            )
            generated_examples += count

        if len(qualitative) < 24:
            predictions = full_scores.argmax(dim=1)
            for item, prediction in zip(metadata, predictions):
                qualitative.append(
                    {
                        "previous": item["previous_character"],
                        "target": item["target_character"],
                        "energy_prediction": bank_characters[int(prediction)],
                    }
                )
                if len(qualitative) >= 24:
                    break
        examples += len(eligible)

    report = {
        key: value / max(1, generated_examples if key.startswith("generated_") else examples)
        for key, value in sums.items()
    }
    for prefix in ("full", "last", "oracle"):
        for metric in ("top1", "top5", "mrr", "target_score"):
            report.setdefault(f"{prefix}_{metric}", 0.0)
    for metric in (
        "context_target_score_gain",
        "unigram_top1",
        "bigram_top1",
        "generated_sample_hit",
        "generated_energy_reranked_top1",
        "generated_best_target_cosine",
        "generated_context_cosine_gain",
        "generated_best_pixel_f1",
        "generated_ink_fraction",
        "generated_pixel_std",
    ):
        report.setdefault(metric, 0.0)
    report.update(
        {
            "branch": name,
            "examples": examples,
            "generated_examples": generated_examples,
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
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a retinal flow language model")
    records = load_visual_grammar_manifest(args.manifest)
    characters, unigram, bigram_best = language_statistics(records, args.bank_size)
    bank_digest = hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()
    render_config = render_config_from_checkpoint(checkpoint)
    model_config = retinal_flow_config_from_payload(checkpoint["model_config"])
    sequence_length = int(checkpoint.get("arguments", {}).get("sequence_length", 48))
    dataset = VisualSaccadeDataset(
        records,
        render_config=render_config,
        spec=SaccadeSequenceSpec(
            sequence_length=sequence_length,
            fovea_size=model_config.fovea_size,
        ),
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
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
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
        reports[name] = evaluate_branch(
            name,
            model,
            loader,
            bank,
            characters,
            unigram,
            bigram_best,
            device=device,
            precision=args.precision,
            generation_contexts=args.generation_contexts,
            samples_per_context=args.samples_per_context,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + offset * 300_007,
            sample_root=output / "samples",
        )
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretrained = reports["pretrained"]
    random_report = reports["random"]
    acceptance = {
        "beats_random": pretrained["full_top1"] > random_report["full_top1"],
        "beats_unigram": pretrained["full_top1"] > pretrained["unigram_top1"],
        "beats_bigram": pretrained["full_top1"] > pretrained["bigram_top1"],
        "uses_longer_context": (
            pretrained["context_target_score_gain"] > 0.0
            and pretrained["full_top1"] >= pretrained["last_top1"]
        ),
        "learned_visual_alphabet": (
            pretrained["oracle_top1"] > random_report["oracle_top1"]
        ),
        "writes_nontrivial_ink": (
            pretrained["generated_pixel_std"] > 0.05
            and 0.01 < pretrained["generated_ink_fraction"] < 0.70
        ),
        "generated_target_signal": (
            pretrained["generated_context_cosine_gain"] > 0.02
            and pretrained["generated_context_cosine_gain"]
            > random_report["generated_context_cosine_gain"] + 0.01
        ),
    }
    result = {
        "architecture": "retinal-flow-fixed-glyph-bank-evaluation-v1",
        "checkpoint": args.checkpoint,
        "bank_size": len(characters),
        "bank_sha256": bank_digest,
        "prototype_views": args.prototype_views,
        "chance_top1": 1.0 / len(characters),
        "student_received_token_ids": False,
        "student_received_labels": False,
        "student_received_ocr": False,
        "student_used_visual_codebook": False,
        "evaluator_used_labels_for_scoring_only": True,
        "branches": reports,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }
    (output / "glyph_bank.json").write_text(
        json.dumps(
            {
                "characters": characters,
                "sha256": bank_digest,
                "labels_visible_to_evaluator_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
