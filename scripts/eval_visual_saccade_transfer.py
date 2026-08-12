#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ilm.visual_lm.folio_data import load_teacher_cache
from ilm.visual_lm.ink_jepa_data import RetinalRenderConfig
from ilm.visual_lm.saccade_data import render_saccade_foveas
from ilm.visual_lm.saccade_lm import VisualSaccadeLM, visual_saccade_config_from_payload
from scripts.eval_ink_jepa_transfer import (
    PromptDocument,
    load_paraphrases,
    prompt_documents,
    retrieval_metrics,
    train_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test frozen visual-saccade states for semantic transfer against random weights."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-cache", default="data/teacher/folio_bge_m3_zh5k.pt")
    parser.add_argument("--paraphrases", default="data/teacher/folio_paraphrases_zh_holdout.jsonl")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--maximum-fixations", type=int, default=96)
    parser.add_argument("--train-views", type=int, default=2)
    parser.add_argument("--evaluation-views", type=int, default=1)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=600)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.20)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--out", default="artifacts/visual_saccade_transfer")
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


def load_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    random_weights: bool,
    seed: int,
) -> VisualSaccadeLM:
    if checkpoint.get("architecture") not in {
        "visual-saccade-language-model-v1",
        "visual-saccade-language-model-v2",
    }:
        raise ValueError("checkpoint is not a visual saccade language model")
    if random_weights:
        torch.manual_seed(seed + 81_337)
    model = VisualSaccadeLM(visual_saccade_config_from_payload(checkpoint["model_config"]))
    if not random_weights:
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval().requires_grad_(False)


def render_config_from_checkpoint(checkpoint: dict[str, Any]) -> RetinalRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = True
    return RetinalRenderConfig(**payload)


def visible_writing(text: str, maximum: int) -> str:
    writing = "".join(character for character in text if not character.isspace())
    if not writing:
        writing = "。"
    if len(writing) > maximum:
        prefix = maximum // 2
        writing = writing[:prefix] + writing[-(maximum - prefix) :]
    return writing


@torch.inference_mode()
def encode_texts(
    model: VisualSaccadeLM,
    texts: Sequence[str],
    identifiers: Sequence[str],
    *,
    render_config: RetinalRenderConfig,
    maximum_fixations: int,
    views: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    features: list[torch.Tensor] = []
    owners: list[int] = []
    pending: list[torch.Tensor] = []
    pending_lengths: list[int] = []
    pending_owners: list[int] = []

    def flush() -> None:
        if not pending:
            return
        maximum_length = max(pending_lengths)
        images = torch.zeros(
            len(pending),
            maximum_length,
            1,
            model.config.fovea_size,
            model.config.fovea_size,
        )
        for index, sequence in enumerate(pending):
            images[index, : sequence.shape[0]] = sequence
        images = images.to(device, non_blocking=True)
        lengths = torch.tensor(pending_lengths, device=device, dtype=torch.long)
        with autocast_context(device, precision):
            visual = model.encode_sequence(images)
            state, _ = model.dynamics(visual)
            selected = state[
                torch.arange(state.shape[0], device=device),
                lengths - 1,
            ].float().cpu()
        features.append(selected)
        owners.extend(pending_owners)
        pending.clear()
        pending_lengths.clear()
        pending_owners.clear()

    for owner, (text, identifier) in enumerate(zip(texts, identifiers)):
        writing = visible_writing(text, min(maximum_fixations, render_config.capacity))
        digest = hashlib.sha256(identifier.encode("utf-8")).digest()
        base_variant = int.from_bytes(digest[:8], "big") ^ seed
        for view in range(views):
            sequence = render_saccade_foveas(
                writing,
                render_config=render_config,
                fovea_size=model.config.fovea_size,
                variant=base_variant + view * 1_000_003,
            )
            pending.append(sequence)
            pending_lengths.append(sequence.shape[0])
            pending_owners.append(owner)
            if len(pending) >= batch_size:
                flush()
    flush()
    return torch.cat(features), owners


def evaluate_branch(
    name: str,
    model: VisualSaccadeLM,
    documents: Sequence[PromptDocument],
    teacher_bank: torch.Tensor,
    *,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    render_config = render_config_from_checkpoint(checkpoint)
    training = [document for document in documents if not document.validation]
    validation = [document for document in documents if document.validation]
    started = time.perf_counter()
    train_features, train_owners = encode_texts(
        model,
        [document.text for document in training],
        [document.identifier for document in training],
        render_config=render_config,
        maximum_fixations=args.maximum_fixations,
        views=args.train_views,
        batch_size=args.feature_batch_size,
        device=device,
        precision=args.precision,
        seed=seed,
    )
    train_targets = torch.tensor(
        [training[owner].bank_index for owner in train_owners],
        dtype=torch.long,
    )
    probe, history = train_probe(
        train_features,
        train_targets,
        teacher_bank,
        args=args,
        device=device,
        seed=seed,
    )
    validation_features, validation_owners = encode_texts(
        model,
        [document.text for document in validation],
        [document.identifier for document in validation],
        render_config=render_config,
        maximum_fixations=args.maximum_fixations,
        views=args.evaluation_views,
        batch_size=args.feature_batch_size,
        device=device,
        precision=args.precision,
        seed=seed + 9_973,
    )
    validation_targets = torch.tensor(
        [validation[owner].bank_index for owner in validation_owners],
        dtype=torch.long,
    )
    rerender = retrieval_metrics(
        probe,
        validation_features,
        validation_targets,
        teacher_bank,
        device=device,
    )
    paraphrase_texts, paraphrase_ids, paraphrase_targets = load_paraphrases(args.paraphrases, documents)
    if paraphrase_texts:
        paraphrase_features, paraphrase_owners = encode_texts(
            model,
            paraphrase_texts,
            paraphrase_ids,
            render_config=render_config,
            maximum_fixations=args.maximum_fixations,
            views=1,
            batch_size=args.feature_batch_size,
            device=device,
            precision=args.precision,
            seed=seed + 19_999,
        )
        expected = torch.tensor(
            [paraphrase_targets[owner] for owner in paraphrase_owners],
            dtype=torch.long,
        )
        paraphrase = retrieval_metrics(
            probe,
            paraphrase_features,
            expected,
            teacher_bank,
            device=device,
        )
    else:
        paraphrase = {"examples": 0}
    return (
        {
            "branch": name,
            "train_examples": int(train_features.shape[0]),
            "feature_dimension": int(train_features.shape[1]),
            "probe_parameters": sum(parameter.numel() for parameter in probe.parameters()),
            "probe_history": history,
            "held_out_rerender": rerender,
            "validated_paraphrase": paraphrase,
            "elapsed_seconds": time.perf_counter() - started,
        },
        probe.state_dict(),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = load_teacher_cache(args.teacher_cache)
    documents = prompt_documents(
        cache,
        validation_fraction=args.validation_fraction,
        limit=args.document_limit,
    )
    teacher_bank = torch.stack([document.teacher for document in documents])
    reports: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    for offset, (name, random_weights) in enumerate((("pretrained", False), ("random", True))):
        model = load_model(
            checkpoint,
            device=device,
            random_weights=random_weights,
            seed=args.seed + offset * 100_003,
        )
        report, probe = evaluate_branch(
            name,
            model,
            documents,
            teacher_bank,
            checkpoint=checkpoint,
            args=args,
            device=device,
            seed=args.seed + offset * 100_003,
        )
        reports[name] = report
        probes[name] = probe
        print(json.dumps({"stage": "branch_complete", **report}, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pretrained = reports["pretrained"]["validated_paraphrase"]
    random_report = reports["random"]["validated_paraphrase"]
    delta = None
    if pretrained.get("examples") and random_report.get("examples"):
        delta = float(pretrained["top1_accuracy"] - random_report["top1_accuracy"])
    final = {
        "architecture": "visual-saccade-frozen-semantic-transfer-v1",
        "checkpoint": args.checkpoint,
        "teacher_cache": args.teacher_cache,
        "documents": len(documents),
        "maximum_visual_fixations": args.maximum_fixations,
        "student_received_text_during_pretraining": False,
        "student_used_tokens": False,
        "branches": reports,
        "paraphrase_top1_delta_over_random": delta,
        "acceptance_rule": (
            "pretrained must materially beat the identically sized random frozen visual state on validated paraphrases"
        ),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "architecture": "visual-saccade-linear-semantic-probes-v1",
            "source_checkpoint": args.checkpoint,
            "states": probes,
        },
        output / "probes.pt",
    )
    print(json.dumps({"stage": "complete", **final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
