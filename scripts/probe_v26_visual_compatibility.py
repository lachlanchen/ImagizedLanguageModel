#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ilm.visual_lm.factorized_visual_context import FactorizedVisualContextModel
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualPairAuditDataset,
    FactorizedVisualPairDataset,
    FactorizedVisualSuffixPair,
    build_factorized_suffix_pairs,
    factorized_visual_pair_collate,
    factorized_visual_pair_student_batch,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import build_visual_character_statistics
from ilm.visual_lm.visual_compatibility_probe import (
    V26_PROBE_STATES,
    VisualCandidateCompatibilityProbe,
    VisualCompatibilityProbeConfig,
    paired_compatibility_loss,
    visual_compatibility_probe_boundary_receipt,
    visual_compatibility_probe_config_payload,
)
from scripts.eval_factorized_visual_context_v26 import (
    AUDIT_SEED,
    load_model_checkpoint,
)
from scripts.train_factorized_visual_context_v26 import FIXED_RENDER_CONFIG
from scripts.train_visual_state_actuator import (
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


ARCHITECTURE = "v26-frozen-visual-compatibility-probe"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/factorized_visual_context_v26_evidence/checkpoint_final.pt"
)
DEFAULT_V26_AUDIT = (
    "artifacts/factorized_visual_context_v26_evidence/development_audit.json"
)
DEFAULT_OUTPUT = "artifacts/v26_visual_compatibility_probe"
TRAIN_PAIR_COUNT = 16_384
DEVELOPMENT_PAIR_COUNT = 512
PROBE_SEED = 20260912
SOURCE_FILES = (
    "ilm/visual_lm/visual_compatibility_probe.py",
    "scripts/probe_v26_visual_compatibility.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc diagnostic: test whether frozen V26 visual states retain "
            "next-glyph information before the stochastic proposal head."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--v26-audit", default=DEFAULT_V26_AUDIT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--steps", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _pair_digest(
    pairs: Sequence[FactorizedVisualSuffixPair],
    *,
    split: str,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(
            "\0".join(
                (
                    pair.suffix,
                    pair.identifier_a,
                    pair.identifier_b,
                    pair.target_a,
                    pair.target_b,
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "count": len(pairs),
        "sha256": digest.hexdigest(),
        "suffix_cells": 4,
        "all_identifiers_differ": all(
            pair.identifier_a != pair.identifier_b for pair in pairs
        ),
        "all_targets_differ": all(
            pair.target_a != pair.target_b for pair in pairs
        ),
        "student_receives_strings": False,
        "split": split,
    }


def _student_images(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in factorized_visual_pair_student_batch(raw).items()
    }


@torch.no_grad()
def extract_paired_visual_states(
    model: FactorizedVisualContextModel,
    raw: Mapping[str, Any],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, torch.Tensor]:
    """Return two cross-font assignments per source pair using images only."""

    images = _student_images(raw, device)
    context_pairs = torch.stack(
        (
            torch.stack((images["context_a"], images["context_b"]), dim=1),
            torch.stack(
                (
                    images["reference_context_a"],
                    images["reference_context_b"],
                ),
                dim=1,
            ),
        ),
        dim=1,
    )
    candidate_pairs = torch.stack(
        (
            torch.stack(
                (
                    images["reference_target_a"],
                    images["reference_target_b"],
                ),
                dim=1,
            ),
            torch.stack((images["target_a"], images["target_b"]), dim=1),
        ),
        dim=1,
    )
    query_candidate_pairs = torch.stack(
        (
            torch.stack((images["target_a"], images["target_b"]), dim=1),
            torch.stack(
                (
                    images["reference_target_a"],
                    images["reference_target_b"],
                ),
                dim=1,
            ),
        ),
        dim=1,
    )
    batch, views = context_pairs.shape[:2]
    context_pairs = context_pairs.reshape(
        batch * views * 2, *context_pairs.shape[3:]
    )
    candidate_pairs = candidate_pairs.reshape(
        batch * views * 2, *candidate_pairs.shape[3:]
    )
    query_candidate_pairs = query_candidate_pairs.reshape(
        batch * views * 2, *query_candidate_pairs.shape[3:]
    )
    with autocast_context(device, precision):
        parts = model.factorize(context_pairs)
        candidate_visual = model.encode_cells(candidate_pairs, target=True)[:, 0]
        query_candidate_visual = model.encode_cells(
            query_candidate_pairs, target=True
        )[:, 0]
    output = {
        name: parts[name].float().reshape(batch * views, 2, -1)
        for name in V26_PROBE_STATES
    }
    output["candidate_visual"] = candidate_visual.float().reshape(
        batch * views, 2, -1
    )
    output["query_candidate_visual"] = query_candidate_visual.float().reshape(
        batch * views, 2, -1
    )
    return output


def _make_probes(
    model: FactorizedVisualContextModel,
    *,
    device: torch.device,
) -> tuple[nn.ModuleDict, VisualCompatibilityProbeConfig]:
    config = VisualCompatibilityProbeConfig(
        context_dim=model.config.model_dim,
        candidate_dim=model.config.visual_dim,
        hidden_dim=model.config.model_dim,
        projection_dim=model.config.visual_dim,
    )
    probes = nn.ModuleDict(
        {
            name: VisualCandidateCompatibilityProbe(config)
            for name in V26_PROBE_STATES
        }
    )
    return probes.to(device), config


def _loader(
    dataset: FactorizedVisualPairDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=factorized_visual_pair_collate,
    )


@torch.no_grad()
def evaluate(
    model: FactorizedVisualContextModel,
    probes: nn.ModuleDict,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    probes.eval()
    totals = {
        name: {
            "assignments": 0.0,
            "arms": 0.0,
            "loss_sum": 0.0,
            "correct_arm_credit": 0.0,
            "strict_correct_arms": 0.0,
            "tied_arms": 0.0,
            "both_correct": 0.0,
            "margin_sum": 0.0,
        }
        for name in V26_PROBE_STATES
    }
    source_pairs = 0
    suffix_checks = 0
    suffix_equal = 0
    identity_total = {
        "assignments": 0.0,
        "arms": 0.0,
        "correct_arm_credit": 0.0,
        "strict_correct_arms": 0.0,
        "tied_arms": 0.0,
        "both_correct": 0.0,
        "margin_sum": 0.0,
    }
    for raw in loader:
        source_pairs += len(raw["metadata"])
        for name_a, name_b in (
            ("context_a", "context_b"),
            ("reference_context_a", "reference_context_b"),
        ):
            suffix_checks += raw[name_a].shape[0]
            suffix_equal += int(
                (raw[name_a][:, -4:] == raw[name_b][:, -4:])
                .flatten(1)
                .all(dim=1)
                .sum()
            )
        states = extract_paired_visual_states(
            model, raw, device=device, precision=precision
        )
        assignments = states["candidate_visual"].shape[0]
        identity_logits = torch.einsum(
            "bqd,bkd->bqk",
            states["query_candidate_visual"],
            states["candidate_visual"],
        )
        identity_margin_a = identity_logits[:, 0, 0] - identity_logits[:, 0, 1]
        identity_margin_b = identity_logits[:, 1, 1] - identity_logits[:, 1, 0]
        identity_margins = torch.stack(
            (identity_margin_a, identity_margin_b), dim=1
        )
        identity_ties = identity_margins == 0
        identity_total["assignments"] += assignments
        identity_total["arms"] += 2 * assignments
        identity_total["correct_arm_credit"] += float(
            (identity_margins > 0).float().sum()
            + 0.5 * identity_ties.float().sum()
        )
        identity_total["strict_correct_arms"] += float(
            (identity_margins > 0).sum()
        )
        identity_total["tied_arms"] += float(identity_ties.sum())
        identity_total["both_correct"] += float(
            (identity_margins > 0).all(dim=1).sum()
        )
        identity_total["margin_sum"] += float(identity_margins.sum())
        for name in V26_PROBE_STATES:
            logits = probes[name](states[name], states["candidate_visual"])
            loss, _ = paired_compatibility_loss(logits)
            margin_a = logits[:, 0, 0] - logits[:, 0, 1]
            margin_b = logits[:, 1, 1] - logits[:, 1, 0]
            margins = torch.stack((margin_a, margin_b), dim=1)
            ties = margins == 0
            total = totals[name]
            total["assignments"] += assignments
            total["arms"] += 2 * assignments
            total["loss_sum"] += float(loss) * assignments
            total["correct_arm_credit"] += float(
                (margins > 0).float().sum() + 0.5 * ties.float().sum()
            )
            total["strict_correct_arms"] += float((margins > 0).sum())
            total["tied_arms"] += float(ties.sum())
            total["both_correct"] += float((margins > 0).all(dim=1).sum())
            total["margin_sum"] += float(margins.sum())
    result: dict[str, Any] = {
        "source_pairs": source_pairs,
        "cross_font_assignments": source_pairs * 2,
        "suffix_pixel_equality": suffix_equal / suffix_checks,
        "retina_identity_control": {
            "arm_accuracy": (
                identity_total["correct_arm_credit"] / identity_total["arms"]
            ),
            "strict_arm_accuracy": (
                identity_total["strict_correct_arms"] / identity_total["arms"]
            ),
            "arm_tie_rate": (
                identity_total["tied_arms"] / identity_total["arms"]
            ),
            "both_correct_rate": (
                identity_total["both_correct"]
                / identity_total["assignments"]
            ),
            "mean_cosine_margin": (
                identity_total["margin_sum"] / identity_total["arms"]
            ),
        },
        "states": {},
    }
    for name, total in totals.items():
        result["states"][name] = {
            "loss": total["loss_sum"] / total["assignments"],
            "arm_accuracy": total["correct_arm_credit"] / total["arms"],
            "strict_arm_accuracy": (
                total["strict_correct_arms"] / total["arms"]
            ),
            "arm_tie_rate": total["tied_arms"] / total["arms"],
            "both_correct_rate": (
                total["both_correct"] / total["assignments"]
            ),
            "mean_margin": total["margin_sum"] / total["arms"],
            "logit_scale": float(probes[name].scale),
        }
    return result


def train_probes(
    model: FactorizedVisualContextModel,
    probes: nn.ModuleDict,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    log_every: int,
) -> list[dict[str, float]]:
    probes.train()
    optimizer = torch.optim.AdamW(
        probes.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            raw = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("probe loader ended before the fixed update count") from exc
        states = extract_paired_visual_states(
            model, raw, device=device, precision=precision
        )
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        for name in V26_PROBE_STATES:
            logits = probes[name](states[name], states["candidate_visual"])
            loss, values = paired_compatibility_loss(logits)
            losses[name] = loss
            metrics[name] = values["arm_accuracy"]
        total_loss = torch.stack(tuple(losses.values())).mean()
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            probes.parameters(), gradient_clip
        )
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            record = {
                "step": float(step),
                "loss": float(total_loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            for name in V26_PROBE_STATES:
                record[f"{name}_loss"] = float(losses[name].detach())
                record[f"{name}_arm_accuracy"] = float(metrics[name])
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return history


def _load_v26_audit(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"available": False, "path": str(source)}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {
        "available": True,
        "path": str(source),
        "sha256": file_sha256(source),
        "suffix4_pair_ranking_accuracy": payload["suffix4"][
            "pair_ranking_accuracy"
        ],
        "suffix4_mean_pair_margin": payload["suffix4"]["mean_pair_margin"],
    }


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.num_workers + 1) < 1:
        raise ValueError("probe sizes must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("probe optimization values must be positive")
    steps = 2 if args.smoke else args.steps
    train_pair_count = 32 if args.smoke else TRAIN_PAIR_COUNT
    development_pair_count = 8 if args.smoke else DEVELOPMENT_PAIR_COUNT
    expected_steps = (train_pair_count + args.batch_size - 1) // args.batch_size
    if not args.smoke and steps != expected_steps:
        raise ValueError(
            f"fixed probe requires --steps={expected_steps} for one complete pair pass"
        )

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(PROBE_SEED)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    manifest_receipt = verify_v25_manifest(args.manifest, strict=not args.smoke)
    records = load_v25_records(args.manifest, strict_manifest=not args.smoke)
    statistics = build_visual_character_statistics(records, bank_size=1_024)
    allowed_targets = set(statistics.characters)
    train_pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=train_pair_count,
        seed=20260910,
        require_different_identifiers=True,
        allowed_targets=allowed_targets,
    )
    development_pairs = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=development_pair_count,
        seed=AUDIT_SEED,
        require_different_identifiers=True,
        allowed_targets=allowed_targets,
    )
    model, checkpoint = load_model_checkpoint(
        args.checkpoint, device=device, allow_smoke=args.smoke
    )
    model.requires_grad_(False).eval()
    probes, probe_config = _make_probes(model, device=device)

    train_dataset = FactorizedVisualPairDataset(
        train_pairs,
        split="train",
        render_config=FIXED_RENDER_CONFIG,
        seed=20260910,
        length=steps * args.batch_size,
    )
    development_dataset = FactorizedVisualPairAuditDataset(
        development_pairs,
        character_index=statistics.index,
    )
    development_loader = _loader(
        development_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    initial = evaluate(
        model,
        probes,
        development_loader,
        device=device,
        precision=args.precision,
    )
    history = train_probes(
        model,
        probes,
        _loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        device=device,
        precision=args.precision,
        steps=steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        log_every=args.log_every,
    )
    final = evaluate(
        model,
        probes,
        development_loader,
        device=device,
        precision=args.precision,
    )
    elapsed = time.monotonic() - started
    peak_vram_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "architecture": ARCHITECTURE,
        "status": "post-v26-exploratory-diagnostic",
        "is_preregistered_evidence": False,
        "authorizes_frozen_evaluation": False,
        "checkpoint": {
            "path": args.checkpoint,
            "sha256": file_sha256(args.checkpoint),
            "architecture": checkpoint.get("architecture"),
            "step": checkpoint.get("step"),
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
        },
        "v26_reference_audit": _load_v26_audit(args.v26_audit),
        "manifest": manifest_receipt,
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "train_pairs": _pair_digest(train_pairs, split="train"),
        "development_pairs": _pair_digest(
            development_pairs, split="development"
        ),
        "probe_config": visual_compatibility_probe_config_payload(probe_config),
        "probe_states": list(V26_PROBE_STATES),
        "probe_parameters": sum(
            parameter.numel() for parameter in probes.parameters()
        ),
        "optimization": {
            "steps": steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "backbone_precision": args.precision,
            "probe_precision": "fp32",
            "seed": PROBE_SEED,
            "one_complete_training_pair_pass": not args.smoke,
        },
        "initial_development": initial,
        "final_development": final,
        "selection_threshold_is_diagnostic_only": 0.65,
        "history_probe_above_diagnostic_threshold": (
            final["states"]["history_residual"]["arm_accuracy"] > 0.65
        ),
        "fused_probe_above_diagnostic_threshold": (
            final["states"]["fused_state"]["arm_accuracy"] > 0.65
        ),
        "retina_identity_control_passes": (
            final["retina_identity_control"]["arm_accuracy"] >= 0.99
        ),
        "boundary": visual_compatibility_probe_boundary_receipt(),
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "runtime_seconds": elapsed,
        "peak_allocated_vram_gib": peak_vram_gib,
        "frozen_images_instantiated": False,
        "training_log": history,
    }
    atomic_save(
        {
            "architecture": ARCHITECTURE,
            "probe_config": result["probe_config"],
            "probe_states": result["probe_states"],
            "probes": probes.state_dict(),
            "result": result,
        },
        output / "probe.pt",
    )
    (output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {output / 'diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
