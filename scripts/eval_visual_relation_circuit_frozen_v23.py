#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
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
    validate_selected_arm,
)
from scripts.prepare_visual_relation_blinded_review_v23 import (
    EXPECTED_PAIRED_AUDIT_SHA256,
)
from scripts.score_visual_relation_blinded_review_v23 import (
    ARCHITECTURE as REVIEW_RESULT_ARCHITECTURE,
)
from scripts.train_visual_relation_circuit_v23 import (
    EXPECTED_CANONICALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    _development_bank_images,
    candidate_selection_gate_report,
    encode_identity_bank,
    evaluate_development,
    load_relation_state,
    save_sample_sheet,
    validate_canonicalizer_checkpoint,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


ARCHITECTURE = "visual-relation-circuit-v23-frozen-evaluation"
EXPECTED_REVIEW_RESULT_SHA256 = (
    "5f5aa82ff3fd92ebb0deabc9f98759fe3061507003498c979deeb4d4bf5df568"
)
FROZEN_EPISODES = 1_024
FROZEN_BATCH_SIZE = 64
FROZEN_IDENTITY_BANK_VIEWS = 4
FROZEN_SEED = FIXED_OPTIMIZATION_ARGUMENTS["dataset_seed"] + 4_000_037


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single authorized V23 frozen-identity evaluation."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--paired-audit", required=True)
    parser.add_argument("--review-result", required=True)
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--canonicalizer-checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument(
        "--out", default="artifacts/visual_relation_circuit_v23_frozen"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=16)
    return parser.parse_args()


def validate_frozen_authorization(
    *,
    candidate: dict[str, Any],
    candidate_sha256: str,
    paired: dict[str, Any],
    paired_sha256: str,
    review_result: dict[str, Any],
    review_result_sha256: str,
) -> None:
    validate_selected_arm(candidate, route_mode=RELATION_AWARE_ROUTE)
    if paired_sha256 != EXPECTED_PAIRED_AUDIT_SHA256:
        raise ValueError("V23 frozen evaluator paired-audit hash differs")
    if paired.get("architecture") != AUDIT_ARCHITECTURE:
        raise ValueError("V23 frozen evaluator requires the paired audit")
    if not paired.get("paired_gate_passed"):
        raise ValueError("V23 frozen evaluator requires a passed paired gate")
    if paired.get("checkpoint_sha256", {}).get(
        RELATION_AWARE_ROUTE
    ) != candidate_sha256:
        raise ValueError("V23 frozen candidate differs from the paired audit")
    if paired.get("frozen_images_instantiated") != 0:
        raise ValueError("V23 paired audit already broke the frozen seal")
    if review_result_sha256 != EXPECTED_REVIEW_RESULT_SHA256:
        raise ValueError("V23 frozen evaluator review-result hash differs")
    if review_result.get("architecture") != REVIEW_RESULT_ARCHITECTURE:
        raise ValueError("V23 frozen evaluator requires the review result")
    if not review_result.get("paired_gate_passed"):
        raise ValueError("V23 review result lost the paired gate")
    if not review_result.get("blinded_review_passed"):
        raise ValueError("V23 blinded review did not pass")
    if not review_result.get("frozen_evaluation_permitted"):
        raise ValueError("V23 blinded review did not authorize frozen evaluation")
    if review_result.get("frozen_images_instantiated") != 0:
        raise ValueError("V23 review stage already instantiated frozen images")


def frozen_performance_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    sealed_metrics = dict(metrics)
    sealed_metrics["frozen_images_instantiated"] = 0.0
    gates = candidate_selection_gate_report(sealed_metrics)
    gates.pop("frozen_bank_sealed")
    return gates


def main() -> None:
    args = parse_args()
    if args.precision != "bf16":
        raise ValueError("V23 frozen evidence requires BF16")
    if args.num_workers != 8:
        raise ValueError("V23 frozen evidence requires 8 data workers")
    if args.sample_count != 16:
        raise ValueError("V23 frozen evidence requires 16 sample images")
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing repeated V23 frozen evaluation: {output_dir}")

    candidate_sha256 = file_sha256(args.candidate)
    paired_sha256 = file_sha256(args.paired_audit)
    review_result_sha256 = file_sha256(args.review_result)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    paired = json.loads(Path(args.paired_audit).read_text(encoding="utf-8"))
    review_result = json.loads(
        Path(args.review_result).read_text(encoding="utf-8")
    )
    # Authorization validation completes before corpus loading or rendering.
    validate_frozen_authorization(
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        paired=paired,
        paired_sha256=paired_sha256,
        review_result=review_result,
        review_result_sha256=review_result_sha256,
    )
    if file_sha256(args.pvf_checkpoint) != EXPECTED_PVF_SHA256:
        raise ValueError("V23 frozen evaluator PVF hash differs")
    if file_sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("V23 frozen evaluator manifest hash differs")
    canonicalizer_sha256 = file_sha256(args.canonicalizer_checkpoint)
    if canonicalizer_sha256 != EXPECTED_CANONICALIZER_SHA256:
        raise ValueError("V23 frozen evaluator canonicalizer hash differs")
    canonicalizer_checkpoint = torch.load(
        args.canonicalizer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_canonicalizer_checkpoint(
        canonicalizer_checkpoint,
        checkpoint_sha256=canonicalizer_sha256,
    )
    if retinal_font_manifest() != candidate["retinal_fonts"]:
        raise ValueError("V23 frozen evaluator retinal fonts differ")

    seed_everything(FIXED_OPTIMIZATION_ARGUMENTS["seed"])
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
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
            raise ValueError(f"V23 frozen partition differs for {key}")

    # This line is the single authorized transition that instantiates frozen images.
    episode_config = VisualRelationEpisodeConfig()
    dataset = VisualRelationEpisodeDataset(
        partitions["frozen"],
        split="development",
        length=FROZEN_EPISODES,
        config=episode_config,
        seed=FROZEN_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=FROZEN_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=visual_relation_collate,
    )
    bank_images = _development_bank_images(
        partitions["frozen"],
        views=FROZEN_IDENTITY_BANK_VIEWS,
        config=episode_config,
        seed=FROZEN_SEED + 100_000,
    ).to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        bank_visual = encode_identity_bank(model, bank_images)
    del bank_images

    started = time.perf_counter()
    metrics = evaluate_development(
        model,
        loader,
        bank_visual=bank_visual,
        bank_characters=partitions["frozen"],
        device=device,
        precision=args.precision,
    )
    gates = frozen_performance_gate_report(metrics)
    passed = all(gates.values())
    output_dir.mkdir(parents=True)
    save_sample_sheet(
        model,
        loader,
        path=output_dir / "frozen_samples.png",
        device=device,
        precision=args.precision,
        sample_count=args.sample_count,
    )
    result = {
        "architecture": ARCHITECTURE,
        "stage": "complete",
        "candidate_sha256": candidate_sha256,
        "paired_audit_sha256": paired_sha256,
        "review_result_sha256": review_result_sha256,
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "canonicalizer_sha256": canonicalizer_sha256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "frozen_seed": FROZEN_SEED,
        "frozen_episodes": FROZEN_EPISODES,
        "frozen_identities": len(partitions["frozen"]),
        "frozen_identity_bank_views": FROZEN_IDENTITY_BANK_VIEWS,
        "frozen_identity_bank_images": (
            len(partitions["frozen"]) * FROZEN_IDENTITY_BANK_VIEWS
        ),
        "frozen_images_instantiated": True,
        "frozen_evaluation_repeated": False,
        "model_selection_performed": False,
        "thresholds_changed": False,
        "metrics": metrics,
        "performance_gates": gates,
        "frozen_gate_passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "precision": args.precision,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
    }
    result_path = output_dir / "frozen_evaluation.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
