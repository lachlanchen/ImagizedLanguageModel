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
from scripts.prepare_visual_packet_opaque_review_v24 import (
    ARCHITECTURE as REVIEW_ARCHITECTURE,
    EXPECTED_PAIRED_AUDIT_SHA256,
)
from scripts.score_visual_packet_opaque_review_v24 import (
    ARCHITECTURE as REVIEW_RESULT_ARCHITECTURE,
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
    _development_bank_images,
    _label_bank_images,
    candidate_selection_gate_report,
    encode_identity_bank,
    evaluate_development,
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


ARCHITECTURE = "visual-packet-reread-stream-v24-frozen-evaluation"
EXPECTED_REVIEW_RESULT_SHA256 = (
    "b1fe4a8b02518cce8ce268ae13f50249ba814ac93fc63496d4dccf7ab9318b29"
)
FROZEN_EPISODES = 1_024
FROZEN_BATCH_SIZE = 64
FROZEN_IDENTITY_BANK_VIEWS = 4
FROZEN_SEED = FIXED_OPTIMIZATION_ARGUMENTS["dataset_seed"] + 4_000_037
DEFAULT_EVIDENCE_ROOT = Path("artifacts/visual_packet_stream_v24_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single authorized V24 frozen-identity evaluation."
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
    parser.add_argument(
        "--review-receipt",
        default=(
            "artifacts/visual_packet_stream_v24_opaque_review/" "review_receipt.json"
        ),
    )
    parser.add_argument(
        "--review-result",
        default=(
            "artifacts/visual_packet_stream_v24_opaque_review/" "review_result.json"
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
    parser.add_argument("--out", default="artifacts/visual_packet_stream_v24_frozen")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def validate_frozen_authorization(
    *,
    candidate: dict[str, Any],
    candidate_sha256: str,
    paired: dict[str, Any],
    paired_sha256: str,
    review_receipt: dict[str, Any],
    review_receipt_sha256: str,
    review_result: dict[str, Any],
    review_result_sha256: str,
) -> None:
    validate_selected_arm(candidate, route_mode=PACKET_AWARE_ROUTE)
    if paired_sha256 != EXPECTED_PAIRED_AUDIT_SHA256:
        raise ValueError("V24 frozen evaluator paired-audit hash differs")
    if paired.get("architecture") != AUDIT_ARCHITECTURE:
        raise ValueError("V24 frozen evaluator requires the paired audit")
    if not paired.get("paired_gate_passed"):
        raise ValueError("V24 frozen evaluator requires a passed paired gate")
    if paired.get("checkpoint_sha256", {}).get(PACKET_AWARE_ROUTE) != candidate_sha256:
        raise ValueError("V24 frozen candidate differs from the paired audit")
    if paired.get("frozen_images_instantiated") != 0:
        raise ValueError("V24 paired audit already broke the frozen seal")

    if review_receipt.get("architecture") != REVIEW_ARCHITECTURE:
        raise ValueError("V24 frozen evaluator requires the review receipt")
    if review_receipt.get("candidate_sha256") != candidate_sha256:
        raise ValueError("V24 review receipt has the wrong candidate")
    if review_receipt.get("paired_audit_sha256") != paired_sha256:
        raise ValueError("V24 review receipt has the wrong paired audit")
    if review_receipt.get("frozen_images_instantiated") != 0:
        raise ValueError("V24 review pack already broke the frozen seal")

    if review_result_sha256 != EXPECTED_REVIEW_RESULT_SHA256:
        raise ValueError("V24 frozen evaluator review-result hash differs")
    if review_result.get("architecture") != REVIEW_RESULT_ARCHITECTURE:
        raise ValueError("V24 frozen evaluator requires the review result")
    if review_result.get("review_receipt_sha256") != review_receipt_sha256:
        raise ValueError("V24 review result has the wrong review receipt")
    if not review_result.get("paired_gate_passed"):
        raise ValueError("V24 review result lost the paired gate")
    if not review_result.get("opaque_review_passed"):
        raise ValueError("V24 opaque review did not pass")
    if not review_result.get("frozen_evaluation_permitted"):
        raise ValueError("V24 opaque review did not authorize frozen evaluation")
    if review_result.get("frozen_images_instantiated") != 0:
        raise ValueError("V24 review stage already instantiated frozen images")


def frozen_performance_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    sealed_metrics = dict(metrics)
    sealed_metrics["identity_bank_identities"] = 88.0
    sealed_metrics["frozen_images_instantiated"] = 0.0
    gates = candidate_selection_gate_report(sealed_metrics)
    gates.pop("identity_bank_complete")
    gates.pop("frozen_bank_sealed")
    gates["frozen_identity_bank_complete"] = metrics[
        "identity_bank_identities"
    ] == float(EXPECTED_PARTITION["frozen_identities"])
    gates["frozen_images_instantiated"] = metrics["frozen_images_instantiated"] == 1.0
    return gates


def main() -> None:
    args = parse_args()
    if args.precision != "bf16":
        raise ValueError("V24 frozen evidence requires BF16")
    if args.num_workers != 8:
        raise ValueError("V24 frozen evidence requires 8 data workers")
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing repeated V24 frozen evaluation: {output_dir}")

    candidate_sha256 = file_sha256(args.candidate)
    paired_sha256 = file_sha256(args.paired_audit)
    review_receipt_sha256 = file_sha256(args.review_receipt)
    review_result_sha256 = file_sha256(args.review_result)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    paired = json.loads(Path(args.paired_audit).read_text(encoding="utf-8"))
    review_receipt = json.loads(Path(args.review_receipt).read_text(encoding="utf-8"))
    review_result = json.loads(Path(args.review_result).read_text(encoding="utf-8"))
    # Authorization must complete before corpus loading or any frozen rendering.
    validate_frozen_authorization(
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        paired=paired,
        paired_sha256=paired_sha256,
        review_receipt=review_receipt,
        review_receipt_sha256=review_receipt_sha256,
        review_result=review_result,
        review_result_sha256=review_result_sha256,
    )

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
            raise ValueError(f"V24 frozen evaluator {name} hash differs")
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
    if retinal_font_manifest() != candidate["retinal_fonts"]:
        raise ValueError("V24 frozen evaluator retinal fonts differ")

    seed_everything(FIXED_OPTIMIZATION_ARGUMENTS["seed"])
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
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
            raise ValueError(f"V24 frozen partition differs for {key}")

    # This is the single authorized transition that instantiates frozen images.
    episode_config = VisualPacketEpisodeConfig()
    dataset = VisualPacketEpisodeDataset(
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
        collate_fn=visual_packet_collate,
    )
    bank_images = _development_bank_images(
        partitions["frozen"],
        views=FROZEN_IDENTITY_BANK_VIEWS,
        config=episode_config,
        seed=FROZEN_SEED + 100_000,
    ).to(device)
    label_images = _label_bank_images(episode_config).to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        bank_visual = encode_identity_bank(model, bank_images)
        label_visual = model.encode_images(label_images)
    del bank_images, label_images

    started = time.perf_counter()
    metrics = evaluate_development(
        model,
        loader,
        bank_visual=bank_visual,
        bank_characters=partitions["frozen"],
        label_visual=label_visual,
        device=device,
        precision=args.precision,
    )
    metrics["frozen_images_instantiated"] = 1.0
    gates = frozen_performance_gate_report(metrics)
    passed = all(gates.values())
    result = {
        "architecture": ARCHITECTURE,
        "stage": "complete",
        "candidate_sha256": candidate_sha256,
        "paired_audit_sha256": paired_sha256,
        "review_receipt_sha256": review_receipt_sha256,
        "review_result_sha256": review_result_sha256,
        "pvf_sha256": input_hashes["pvf"],
        "canonicalizer_sha256": input_hashes["canonicalizer"],
        "relation_sha256": input_hashes["relation"],
        "manifest_sha256": input_hashes["manifest"],
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
    output_dir.mkdir(parents=True)
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
