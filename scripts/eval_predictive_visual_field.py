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

from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.predictive_visual_field import (
    PredictiveVisualField,
    predictive_visual_field_config_from_payload,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    render_glyph_fovea,
    visual_saccade_collate,
)


ARCHITECTURE = "predictive-visual-field-state-flow-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sampled continuous retinal states on a frozen visual bank; "
            "labels remain evaluator-only."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/predictive_visual_field_evaluation")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--prototype-views", type=int, default=4)
    parser.add_argument("--evaluation-samples", type=int, default=3_000)
    parser.add_argument("--samples-per-context", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
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
        raise ValueError(f"requested {bank_size} Han glyphs, corpus supplied {len(bank)}")
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
    random_dynamics: bool,
    seed: int,
) -> PredictiveVisualField:
    if random_dynamics:
        torch.manual_seed(seed + 91_337)
    model = PredictiveVisualField(
        predictive_visual_field_config_from_payload(checkpoint["model_config"])
    )
    if random_dynamics:
        retina_state = {
            key.removeprefix("retina."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("retina.")
        }
        model.retina.load_state_dict(retina_state)
    else:
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"unexpected checkpoint parameters: {incompatible.unexpected_keys}"
            )
        if any(
            not key.startswith("visual_proposal.")
            for key in incompatible.missing_keys
        ):
            raise ValueError(
                f"unsupported missing checkpoint parameters: {incompatible.missing_keys}"
            )
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def prototype_bank(
    model: PredictiveVisualField,
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
    encoded: list[torch.Tensor] = []

    def flush() -> None:
        if not images:
            return
        batch = torch.stack(images).to(device, non_blocking=True)
        with autocast_context(device, precision):
            encoded.append(model.encode_images(batch).float().cpu())
        images.clear()

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
            if len(images) >= batch_size:
                flush()
    flush()
    features = torch.cat(encoded)
    return F.normalize(
        features.reshape(len(characters), views, features.shape[-1]),
        dim=-1,
    ).to(device)


def rank_metrics(scores: torch.Tensor, expected: torch.Tensor) -> dict[str, torch.Tensor]:
    target_score = scores.gather(1, expected[:, None])
    rank = 1 + (scores > target_score).sum(dim=1)
    return {
        "top1": (rank == 1).float(),
        "top5": (rank <= 5).float(),
        "mrr": 1.0 / rank.float(),
        "target_score": target_score[:, 0],
    }


@torch.inference_mode()
def evaluate_branch(
    name: str,
    model: PredictiveVisualField,
    loader: DataLoader,
    bank: torch.Tensor,
    bank_characters: Sequence[str],
    unigram: Counter[str],
    bigram_best: dict[str, str],
    *,
    device: torch.device,
    precision: str,
    samples_per_context: int,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    probe_root: Path,
) -> dict[str, Any]:
    bank_index = {character: index for index, character in enumerate(bank_characters)}
    sums: dict[str, float] = defaultdict(float)
    examples = 0
    qualitative: list[dict[str, Any]] = []
    first_probe_saved = False
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
        target_reference = batch["target_reference"][eligible, -1].to(
            device,
            non_blocking=True,
        )
        metadata = [batch["metadata"][index] for index in eligible]
        expected = torch.tensor(
            [bank_index[item["target_character"]] for item in metadata],
            dtype=torch.long,
            device=device,
        )
        with autocast_context(device, precision):
            full_prediction = model.predict(context)
            last_prediction = model.predict(context[:, -1:])
            noise = torch.randn(
                context.shape[0],
                samples_per_context,
                model.config.visual_dim,
                device=device,
                dtype=full_prediction["condition"].dtype,
                generator=generator,
            )
            full_states = model.sample_states(
                full_prediction["condition"][:, -1],
                samples_per_context=samples_per_context,
                steps=sample_steps,
                guidance_scale=guidance_scale,
                noise=noise,
            )
            last_states = model.sample_states(
                last_prediction["condition"][:, -1],
                samples_per_context=samples_per_context,
                steps=sample_steps,
                guidance_scale=guidance_scale,
                noise=noise,
            )
            full_scores = model.score_candidates(full_states, bank)
            last_scores = model.score_candidates(last_states, bank)
            full_proposal = full_prediction["proposal_visual"][:, -1]
            last_proposal = last_prediction["proposal_visual"][:, -1]
            proposal_full_scores = model.score_candidates(full_proposal[:, None], bank)
            proposal_last_scores = model.score_candidates(last_proposal[:, None], bank)
            oracle_visual = F.normalize(model.encode_images(target_reference), dim=-1)
        oracle_scores = torch.einsum("bd,nvd->bnv", oracle_visual, bank).amax(dim=-1)
        full_metrics = rank_metrics(full_scores, expected)
        last_metrics = rank_metrics(last_scores, expected)
        oracle_metrics = rank_metrics(oracle_scores, expected)
        proposal_full_metrics = rank_metrics(proposal_full_scores, expected)
        proposal_last_metrics = rank_metrics(proposal_last_scores, expected)
        full_log_probability = full_scores.log_softmax(dim=-1).gather(
            1,
            expected[:, None],
        )[:, 0]
        last_log_probability = last_scores.log_softmax(dim=-1).gather(
            1,
            expected[:, None],
        )[:, 0]
        proposal_full_log_probability = proposal_full_scores.log_softmax(dim=-1).gather(
            1,
            expected[:, None],
        )[:, 0]
        proposal_last_log_probability = proposal_last_scores.log_softmax(dim=-1).gather(
            1,
            expected[:, None],
        )[:, 0]
        for key, values in full_metrics.items():
            sums[f"full_{key}"] += float(values.sum())
        for key, values in last_metrics.items():
            sums[f"last_{key}"] += float(values.sum())
        for key, values in oracle_metrics.items():
            sums[f"oracle_{key}"] += float(values.sum())
        for key, values in proposal_full_metrics.items():
            sums[f"proposal_full_{key}"] += float(values.sum())
        for key, values in proposal_last_metrics.items():
            sums[f"proposal_last_{key}"] += float(values.sum())
        sums["full_target_log_probability"] += float(full_log_probability.sum())
        sums["last_target_log_probability"] += float(last_log_probability.sum())
        sums["context_target_log_probability_gain"] += float(
            (full_log_probability - last_log_probability).sum()
        )
        sums["context_target_score_gain"] += float(
            (full_metrics["target_score"] - last_metrics["target_score"]).sum()
        )
        sums["proposal_full_target_log_probability"] += float(
            proposal_full_log_probability.sum()
        )
        sums["proposal_last_target_log_probability"] += float(
            proposal_last_log_probability.sum()
        )
        sums["proposal_context_target_log_probability_gain"] += float(
            (proposal_full_log_probability - proposal_last_log_probability).sum()
        )
        sums["proposal_context_target_score_gain"] += float(
            (
                proposal_full_metrics["target_score"]
                - proposal_last_metrics["target_score"]
            ).sum()
        )
        most_frequent = bank_characters[0]
        sums["unigram_top1"] += sum(
            item["target_character"] == most_frequent for item in metadata
        )
        sums["bigram_top1"] += sum(
            bigram_best.get(item["previous_character"]) == item["target_character"]
            for item in metadata
        )
        full_target_cosine = torch.einsum("bsd,bd->bs", full_states, oracle_visual)
        last_target_cosine = torch.einsum("bsd,bd->bs", last_states, oracle_visual)
        shuffled = oracle_visual.roll(1, dims=0) if context.shape[0] > 1 else oracle_visual
        shuffled_cosine = torch.einsum("bsd,bd->bs", full_states, shuffled)
        cosine_gain = (
            full_target_cosine.amax(dim=1) - shuffled_cosine.amax(dim=1)
            if context.shape[0] > 1
            else torch.zeros_like(full_target_cosine[:, 0])
        )
        sums["sampled_best_target_cosine"] += float(full_target_cosine.amax(dim=1).sum())
        sums["last_sampled_best_target_cosine"] += float(
            last_target_cosine.amax(dim=1).sum()
        )
        sums["sampled_context_cosine_gain"] += float(cosine_gain.sum())
        proposal_target_cosine = (full_proposal * oracle_visual).sum(dim=-1)
        proposal_last_target_cosine = (last_proposal * oracle_visual).sum(dim=-1)
        sums["proposal_target_cosine"] += float(proposal_target_cosine.sum())
        sums["proposal_last_target_cosine"] += float(
            proposal_last_target_cosine.sum()
        )
        if samples_per_context > 1:
            pair_cosine = (full_states[:, 0] * full_states[:, -1]).sum(dim=-1)
            sums["sampled_pair_cosine"] += float(pair_cosine.sum())
        else:
            sums["sampled_pair_cosine"] += float(context.shape[0])

        if not first_probe_saved:
            probe_root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "branch": name,
                    "context_pixels": context[:8].float().cpu(),
                    "target_pixels": target_reference[:8].float().cpu(),
                    "full_states": full_states[:8].float().cpu(),
                    "last_states": last_states[:8].float().cpu(),
                    "expected_evaluator_indices": expected[:8].cpu(),
                    "labels_visible_to_evaluator_only": True,
                },
                probe_root / f"{name}_state_probe.pt",
            )
            first_probe_saved = True

        if len(qualitative) < 24:
            predictions = full_scores.argmax(dim=1)
            proposal_predictions = proposal_full_scores.argmax(dim=1)
            for item, prediction, proposal_prediction in zip(
                metadata,
                predictions,
                proposal_predictions,
            ):
                qualitative.append(
                    {
                        "previous": item["previous_character"],
                        "target": item["target_character"],
                        "sample_density_prediction": bank_characters[int(prediction)],
                        "proposal_prediction": bank_characters[int(proposal_prediction)],
                    }
                )
                if len(qualitative) >= 24:
                    break
        examples += len(eligible)

    report = {key: value / max(1, examples) for key, value in sums.items()}
    for prefix in (
        "full",
        "last",
        "oracle",
        "proposal_full",
        "proposal_last",
    ):
        for metric in ("top1", "top5", "mrr", "target_score"):
            report.setdefault(f"{prefix}_{metric}", 0.0)
    for metric in (
        "full_target_log_probability",
        "last_target_log_probability",
        "context_target_log_probability_gain",
        "context_target_score_gain",
        "unigram_top1",
        "bigram_top1",
        "sampled_best_target_cosine",
        "last_sampled_best_target_cosine",
        "sampled_context_cosine_gain",
        "sampled_pair_cosine",
        "proposal_full_target_log_probability",
        "proposal_last_target_log_probability",
        "proposal_context_target_log_probability_gain",
        "proposal_context_target_score_gain",
        "proposal_target_cosine",
        "proposal_last_target_cosine",
    ):
        report.setdefault(metric, 0.0)
    elapsed = time.perf_counter() - started
    report.update(
        {
            "branch": name,
            "examples": examples,
            "elapsed_seconds": elapsed,
            "sampled_states_per_second": (
                examples * samples_per_context * 2 / max(1e-6, elapsed)
            ),
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
        raise ValueError("checkpoint is not a Predictive Visual Field state flow")
    records = load_visual_grammar_manifest(args.manifest)
    characters, unigram, bigram_best = language_statistics(records, args.bank_size)
    bank_digest = hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()
    render_config = render_config_from_checkpoint(checkpoint)
    model_config = predictive_visual_field_config_from_payload(checkpoint["model_config"])
    sequence_length = int(
        checkpoint.get("arguments", {}).get("sequence_length", 48)
    )
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
    for name, random_dynamics in (("pretrained", False), ("random", True)):
        model = load_model(
            checkpoint,
            device=device,
            random_dynamics=random_dynamics,
            seed=args.seed,
        )
        bank = prototype_bank(
            model,
            characters,
            render_config=render_config,
            views=args.prototype_views,
            batch_size=max(args.batch_size, 64),
            device=device,
            precision=args.precision,
            seed=args.seed,
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
            samples_per_context=args.samples_per_context,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            probe_root=output / "state_probes",
        )
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretrained = reports["pretrained"]
    random_report = reports["random"]
    state_flow_acceptance = {
        "beats_random": pretrained["full_top1"] > random_report["full_top1"],
        "beats_unigram": pretrained["full_top1"] > pretrained["unigram_top1"],
        "uses_longer_context": (
            pretrained["context_target_log_probability_gain"] > 0.0
            and pretrained["full_top1"] >= pretrained["last_top1"]
        ),
        "sampled_target_signal": (
            pretrained["sampled_context_cosine_gain"] > 0.02
            and pretrained["sampled_context_cosine_gain"]
            > random_report["sampled_context_cosine_gain"] + 0.01
        ),
        "retina_is_not_bottleneck": pretrained["oracle_top1"] > 0.90,
    }
    language_acceptance = {
        **state_flow_acceptance,
        "beats_bigram": pretrained["full_top1"] > pretrained["bigram_top1"],
    }
    proposal_acceptance = {
        "beats_random": (
            pretrained["proposal_full_top1"]
            > random_report["proposal_full_top1"]
        ),
        "beats_unigram": (
            pretrained["proposal_full_top1"] > pretrained["unigram_top1"]
        ),
        "uses_longer_context": (
            pretrained["proposal_context_target_log_probability_gain"] > 0.0
            and pretrained["proposal_full_top1"]
            >= pretrained["proposal_last_top1"]
        ),
        "target_signal": (
            pretrained["proposal_target_cosine"]
            > random_report["proposal_target_cosine"] + 0.02
        ),
        "retina_is_not_bottleneck": pretrained["oracle_top1"] > 0.90,
    }
    proposal_language_acceptance = {
        **proposal_acceptance,
        "beats_bigram": (
            pretrained["proposal_full_top1"] > pretrained["bigram_top1"]
        ),
    }
    result = {
        "architecture": "predictive-visual-field-fixed-state-bank-evaluation-v1",
        "checkpoint": args.checkpoint,
        "bank_size": len(characters),
        "bank_sha256": bank_digest,
        "prototype_views": args.prototype_views,
        "samples_per_context": args.samples_per_context,
        "sample_steps": args.sample_steps,
        "flow_geometry": model_config.flow_geometry,
        "retinal_fonts": retinal_font_manifest(),
        "chance_top1": 1.0 / len(characters),
        "student_received_token_ids": False,
        "student_received_labels": False,
        "student_received_ocr": False,
        "student_used_visual_codebook": False,
        "student_used_pixel_writer": False,
        "student_used_candidate_classifier": False,
        "evaluator_used_labels_for_scoring_only": True,
        "candidate_scores_derived_from_sampled_continuous_states": True,
        "proposal_scores_derived_from_continuous_image_state": True,
        "branches": reports,
        "state_flow_acceptance": state_flow_acceptance,
        "state_flow_accepted": all(state_flow_acceptance.values()),
        "language_acceptance": language_acceptance,
        "language_accepted": all(language_acceptance.values()),
        "proposal_acceptance": proposal_acceptance,
        "proposal_accepted": all(proposal_acceptance.values()),
        "proposal_language_acceptance": proposal_language_acceptance,
        "proposal_language_accepted": all(proposal_language_acceptance.values()),
    }
    (output / "state_bank.json").write_text(
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
