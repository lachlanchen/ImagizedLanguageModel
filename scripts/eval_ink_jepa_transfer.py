#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.folio_data import load_teacher_cache, semantic_residual_fields, stable_fraction
from ilm.visual_lm.ink_jepa import InkJEPA, ink_jepa_config_from_payload
from ilm.visual_lm.ink_jepa_data import RetinalRenderConfig, render_retinal_page


@dataclass(frozen=True)
class PromptDocument:
    bank_index: int
    identifier: str
    text: str
    teacher: torch.Tensor
    validation: bool


class LinearSemanticProbe(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dimension)
        self.output = nn.Linear(input_dimension, output_dimension)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output(self.norm(features)).float(), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Falsify InkJEPA with a frozen-encoder semantic transfer comparison."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-cache", default="data/teacher/folio_bge_m3_zh5k.pt")
    parser.add_argument("--paraphrases", default="data/teacher/folio_paraphrases_zh_holdout.jsonl")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--train-views", type=int, default=2)
    parser.add_argument("--evaluation-views", type=int, default=1)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=600)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.20)
    parser.add_argument("--out", default="artifacts/ink_jepa_transfer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_documents(
    cache: dict[str, Any],
    *,
    validation_fraction: float,
    limit: int | None,
) -> list[PromptDocument]:
    residuals, _ = semantic_residual_fields(cache)
    output: list[PromptDocument] = []
    for document_index, document in enumerate(cache["documents"]):
        if document.get("kind") != "prompt":
            continue
        identifier = str(document["record_identifier"])
        output.append(
            PromptDocument(
                bank_index=len(output),
                identifier=identifier,
                text=str(document["text"]),
                teacher=residuals[document_index],
                validation=stable_fraction(identifier) < validation_fraction,
            )
        )
        if limit is not None and len(output) >= limit:
            break
    if not output:
        raise ValueError("teacher cache contains no prompt documents")
    return output


def load_encoder(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    random_weights: bool,
    seed: int,
) -> nn.Module:
    if checkpoint.get("architecture") != "ink-jepa-retinal-predictive-field-v1":
        raise ValueError("checkpoint is not an InkJEPA retinal predictive field")
    if random_weights:
        torch.manual_seed(seed + 81_337)
    model = InkJEPA(ink_jepa_config_from_payload(checkpoint["model_config"]))
    if not random_weights:
        model.load_state_dict(checkpoint["model"])
    encoder = model.target_encoder.to(device).eval()
    encoder.requires_grad_(False)
    return encoder


def render_config_from_checkpoint(checkpoint: dict[str, Any]) -> RetinalRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = True
    return RetinalRenderConfig(**payload)


@torch.inference_mode()
def encode_texts(
    encoder: nn.Module,
    texts: Sequence[str],
    identifiers: Sequence[str],
    *,
    render_config: RetinalRenderConfig,
    views: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    features: list[torch.Tensor] = []
    owners: list[int] = []
    pending_images: list[torch.Tensor] = []
    pending_owners: list[int] = []

    def flush() -> None:
        if not pending_images:
            return
        images = torch.stack(pending_images).to(device, non_blocking=True)
        page = encoder(images)["page"].float().cpu()
        features.append(page)
        owners.extend(pending_owners)
        pending_images.clear()
        pending_owners.clear()

    for owner, (text, identifier) in enumerate(zip(texts, identifiers)):
        digest = hashlib.sha256(identifier.encode("utf-8")).digest()
        base_variant = int.from_bytes(digest[:8], "big") ^ seed
        for view in range(views):
            pending_images.append(
                render_retinal_page(text, config=render_config, variant=base_variant + view * 1_000_003)
            )
            pending_owners.append(owner)
            if len(pending_images) >= batch_size:
                flush()
    flush()
    return torch.cat(features, dim=0), owners


def train_probe(
    features: torch.Tensor,
    target_indices: torch.Tensor,
    teacher_bank: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[LinearSemanticProbe, list[dict[str, float]]]:
    torch.manual_seed(seed)
    probe = LinearSemanticProbe(features.shape[1], teacher_bank.shape[1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=0.01)
    features = features.to(device)
    target_indices = target_indices.to(device)
    teacher_bank = F.normalize(teacher_bank.float(), dim=-1).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    history: list[dict[str, float]] = []
    probe.train()
    for step in range(1, args.probe_steps + 1):
        selected = torch.randint(
            0,
            features.shape[0],
            (min(args.probe_batch_size, features.shape[0]),),
            device=device,
            generator=generator,
        )
        prediction = probe(features[selected])
        labels = target_indices[selected]
        target = teacher_bank[labels]
        cosine = (1.0 - (prediction * target).sum(dim=-1)).mean()
        logits = 30.0 * prediction @ teacher_bank.transpose(0, 1)
        contrastive = F.cross_entropy(logits, labels)
        loss = cosine + args.contrastive_weight * contrastive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.probe_steps:
            with torch.no_grad():
                accuracy = (logits.argmax(dim=1) == labels).float().mean()
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "cosine_loss": float(cosine.detach()),
                    "full_bank_contrastive": float(contrastive.detach()),
                    "batch_top1": float(accuracy),
                }
            )
    return probe.eval(), history


@torch.inference_mode()
def retrieval_metrics(
    probe: LinearSemanticProbe,
    features: torch.Tensor,
    expected: torch.Tensor,
    teacher_bank: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    prediction = probe(features.to(device))
    bank = F.normalize(teacher_bank.float(), dim=-1).to(device)
    similarities = prediction @ bank.transpose(0, 1)
    expected = expected.to(device)
    target_scores = similarities.gather(1, expected[:, None])
    ranks = 1 + (similarities > target_scores).sum(dim=1)
    return {
        "examples": int(features.shape[0]),
        "top1_accuracy": float((ranks == 1).float().mean()),
        "top5_accuracy": float((ranks <= 5).float().mean()),
        "mean_reciprocal_rank": float((1.0 / ranks.float()).mean()),
        "mean_target_cosine": float(target_scores.mean()),
    }


def load_paraphrases(
    path: str | None,
    documents: Sequence[PromptDocument],
) -> tuple[list[str], list[str], list[int]]:
    if not path or not Path(path).exists():
        return [], [], []
    by_identifier = {document.identifier: document.bank_index for document in documents}
    texts: list[str] = []
    identifiers: list[str] = []
    targets: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = str(record.get("identifier", ""))
        paraphrase = record.get("paraphrase")
        if identifier not in by_identifier or not isinstance(paraphrase, str) or not paraphrase.strip():
            continue
        texts.append(paraphrase.strip())
        identifiers.append(f"paraphrase:{identifier}")
        targets.append(by_identifier[identifier])
    return texts, identifiers, targets


def evaluate_branch(
    name: str,
    encoder: nn.Module,
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
        encoder,
        [document.text for document in training],
        [document.identifier for document in training],
        render_config=render_config,
        views=args.train_views,
        batch_size=args.feature_batch_size,
        device=device,
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
        encoder,
        [document.text for document in validation],
        [document.identifier for document in validation],
        render_config=render_config,
        views=args.evaluation_views,
        batch_size=args.feature_batch_size,
        device=device,
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
            encoder,
            paraphrase_texts,
            paraphrase_ids,
            render_config=render_config,
            views=1,
            batch_size=args.feature_batch_size,
            device=device,
            seed=seed + 19_999,
        )
        expected = torch.tensor([paraphrase_targets[owner] for owner in paraphrase_owners], dtype=torch.long)
        paraphrase = retrieval_metrics(
            probe,
            paraphrase_features,
            expected,
            teacher_bank,
            device=device,
        )
    else:
        paraphrase = {"examples": 0}
    report = {
        "branch": name,
        "train_examples": int(train_features.shape[0]),
        "feature_dimension": int(train_features.shape[1]),
        "probe_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "probe_history": history,
        "held_out_rerender": rerender,
        "validated_paraphrase": paraphrase,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report, probe.state_dict()


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
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    for offset, (name, random_weights) in enumerate((("pretrained", False), ("random", True))):
        encoder = load_encoder(
            checkpoint,
            device=device,
            random_weights=random_weights,
            seed=args.seed,
        )
        report, probe = evaluate_branch(
            name,
            encoder,
            documents,
            teacher_bank,
            checkpoint=checkpoint,
            args=args,
            device=device,
            seed=args.seed + offset * 100_003,
        )
        reports[name] = report
        probes[name] = probe
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"stage": "branch_complete", **report}, ensure_ascii=False), flush=True)
    pretrained = reports["pretrained"]["validated_paraphrase"]
    random_report = reports["random"]["validated_paraphrase"]
    delta = None
    if pretrained.get("examples") and random_report.get("examples"):
        delta = float(pretrained["top1_accuracy"] - random_report["top1_accuracy"])
    final = {
        "architecture": "ink-jepa-frozen-semantic-transfer-v1",
        "checkpoint": args.checkpoint,
        "teacher_cache": args.teacher_cache,
        "documents": len(documents),
        "student_received_text_during_pretraining": False,
        "student_used_tokens": False,
        "branches": reports,
        "paraphrase_top1_delta_over_random": delta,
        "acceptance_rule": (
            "pretrained must materially beat the identically sized random frozen encoder on validated paraphrases"
        ),
    }
    (output / "evaluation.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "architecture": "ink-jepa-linear-semantic-probes-v1",
            "source_checkpoint": args.checkpoint,
            "states": probes,
        },
        output / "probes.pt",
    )
    print(json.dumps({"stage": "complete", **final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
