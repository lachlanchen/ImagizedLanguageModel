#!/usr/bin/env python3
"""Diagnose which visual prompt roles a rejected V22 endpoint selected.

This is a development-only mechanism audit. It does not select a checkpoint,
run the paired gate, instantiate frozen identities, or authorize human review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import load_visual_grammar_manifest
from ilm.visual_lm.visual_binding_data import (
    PARTITION_SALT,
    VisualBindingEpisodeConfig,
    VisualBindingEpisodeDataset,
    binding_partition_receipt,
    build_binding_character_bank,
    split_binding_characters,
    visual_binding_collate,
)
from ilm.visual_lm.visual_binding_stream import (
    QUERY_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    VisualBindingStream,
    visual_binding_config_from_payload,
)
from scripts.train_visual_binding_stream import (
    ARCHITECTURE,
    EXPECTED_PARAMETERS,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    load_student_state,
    student_boundary_is_clean,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


PROMPT_ROLES = (
    "label_1",
    "glyph_1",
    "label_2",
    "glyph_2",
    "operation",
    "query_label",
)
DEVELOPMENT_SEED_OFFSET = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V22 endpoint attention by visual prompt role."
    )
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pvf-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    return parser.parse_args()


def summarize_attention(attention: torch.Tensor) -> dict[str, Any]:
    if attention.ndim != 2 or attention.shape[1] != len(PROMPT_ROLES):
        raise ValueError(
            f"attention must have shape [N,{len(PROMPT_ROLES)}], got "
            f"{tuple(attention.shape)}"
        )
    if attention.shape[0] < 1:
        raise ValueError("attention audit requires at least one example")
    attention = attention.detach().float().cpu()
    if not torch.isfinite(attention).all():
        raise ValueError("attention contains non-finite values")
    row_error = (attention.sum(dim=1) - 1.0).abs().max().item()
    if row_error > 1e-4:
        raise ValueError(f"attention rows are not normalized: {row_error}")
    winners = attention.argmax(dim=1)
    counts = torch.bincount(winners, minlength=len(PROMPT_ROLES))
    means = attention.mean(dim=0)
    return {
        "examples": int(attention.shape[0]),
        "max_row_sum_error": float(row_error),
        "mean_attention": {
            role: float(means[index])
            for index, role in enumerate(PROMPT_ROLES)
        },
        "argmax_counts": {
            role: int(counts[index])
            for index, role in enumerate(PROMPT_ROLES)
        },
    }


def _load_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_endpoint(
    checkpoint: dict[str, Any],
    *,
    name: str,
    route_mode: str,
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{name} is not a V22 visual binding checkpoint")
    if checkpoint.get("route_mode") != route_mode:
        raise ValueError(f"{name} has the wrong route mode")
    if checkpoint.get("smoke_only", False):
        raise ValueError("smoke-only checkpoints cannot enter the V22 audit")
    if int(checkpoint.get("step", -1)) != int(
        FIXED_EVIDENCE_ARGUMENTS["maximum_steps"]
    ):
        raise ValueError(f"{name} is not the fixed V22 endpoint")
    if checkpoint.get("pvf_sha256") != EXPECTED_PVF_SHA256:
        raise ValueError(f"{name} does not use the preregistered retina")
    if checkpoint.get("trainable_parameters") != EXPECTED_PARAMETERS:
        raise ValueError(f"{name} has the wrong parameter count")
    if not student_boundary_is_clean(
        checkpoint.get("boundary_receipt", {}), route_mode
    ):
        raise ValueError(f"{name} violates the image-only boundary")
    if checkpoint.get("partition", {}).get("frozen_images_instantiated"):
        raise ValueError(f"{name} reports instantiated frozen images")
    for key, expected in EXPECTED_PARTITION.items():
        if checkpoint.get("partition", {}).get(key) != expected:
            raise ValueError(f"{name} partition differs at {key!r}")


def _load_model(
    checkpoint: dict[str, Any],
    retina: torch.nn.Module,
    device: torch.device,
) -> VisualBindingStream:
    model = VisualBindingStream(
        visual_binding_config_from_payload(checkpoint["model_config"]),
        retina,
    )
    load_student_state(model, checkpoint["student"])
    return model.to(device).eval()


@torch.no_grad()
def _audit_model(
    model: VisualBindingStream,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    rows: list[torch.Tensor] = []
    for batch in loader:
        prompt = torch.cat(
            (batch["prompt"], batch["counterfactual_prompt"]), dim=0
        ).to(device, non_blocking=True)
        with autocast_context(device, precision):
            _, trace = model.logits_with_trace(prompt)
        rows.append(trace["selection_attention"].float().cpu())
    return summarize_attention(torch.cat(rows, dim=0))


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V22 audit: {output}")

    candidate_checkpoint = _load_checkpoint(args.candidate_checkpoint)
    control_checkpoint = _load_checkpoint(args.control_checkpoint)
    _validate_endpoint(
        candidate_checkpoint,
        name="candidate",
        route_mode=QUERY_AWARE_ROUTE,
    )
    _validate_endpoint(
        control_checkpoint,
        name="control",
        route_mode=QUERY_BLIND_ROUTE,
    )
    if candidate_checkpoint.get("best_development") is not None:
        raise ValueError("the V22 candidate unexpectedly selected")
    if control_checkpoint.get("best_development") is None:
        raise ValueError("the V22 structural control did not select")
    if candidate_checkpoint["partition"] != control_checkpoint["partition"]:
        raise ValueError("V22 endpoint partitions differ")

    manifest_path = args.manifest or candidate_checkpoint["args"]["manifest"]
    if file_sha256(manifest_path) != candidate_checkpoint["manifest_sha256"]:
        raise ValueError("V22 manifest bytes differ from training")
    pvf_path = args.pvf_checkpoint or candidate_checkpoint["pvf_checkpoint"]
    if file_sha256(pvf_path) != EXPECTED_PVF_SHA256:
        raise ValueError("V22 retina bytes differ from preregistration")

    records = load_visual_grammar_manifest(manifest_path)
    bank = build_binding_character_bank(
        records,
        bank_size=int(candidate_checkpoint["args"]["bank_size"]),
    )
    partitions = split_binding_characters(bank, salt=PARTITION_SALT)
    partition = binding_partition_receipt(partitions, salt=PARTITION_SALT)
    if partition != candidate_checkpoint["partition"]:
        raise ValueError("fresh V22 partition differs from the checkpoint")

    dataset_seed = int(candidate_checkpoint["args"]["dataset_seed"])
    dataset = VisualBindingEpisodeDataset(
        partitions["development"],
        split="development",
        length=int(FIXED_EVIDENCE_ARGUMENTS["development_samples"]),
        config=VisualBindingEpisodeConfig(),
        seed=dataset_seed + DEVELOPMENT_SEED_OFFSET,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(FIXED_EVIDENCE_ARGUMENTS["batch_size"]),
        shuffle=False,
        num_workers=int(FIXED_EVIDENCE_ARGUMENTS["num_workers"]),
        pin_memory=args.device != "cpu",
        persistent_workers=False,
        collate_fn=visual_binding_collate,
    )

    seed_everything(int(candidate_checkpoint["args"]["seed"]))
    device = choose_device(args.device)
    pvf, _ = load_pvf(pvf_path, device)
    candidate = _load_model(candidate_checkpoint, pvf.retina, device)
    control = _load_model(control_checkpoint, pvf.retina, device)

    payload = {
        "architecture": "visual-binding-stream-v22-endpoint-attention-audit-v1",
        "status": "development-only diagnostic; not checkpoint selection",
        "prompt_roles": list(PROMPT_ROLES),
        "development_pairs": len(dataset),
        "frozen_images_instantiated": 0,
        "candidate": _audit_model(
            candidate,
            loader,
            device=device,
            precision=args.precision,
        ),
        "control": _audit_model(
            control,
            loader,
            device=device,
            precision=args.precision,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
