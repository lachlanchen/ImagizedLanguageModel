#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.visual_answer_trajectory import (
    V39_ARCHITECTURE,
    VisualAnswerTrajectoryConfig,
    VisualAnswerTrajectoryModel,
    visual_answer_trajectory_boundary_receipt,
)
from ilm.visual_lm.visual_answer_trajectory_data import (
    V39_DEVELOPMENT_FONT,
    V39_HELD_FONT,
    VisualAnswerTrajectoryRecord,
    convert_visual_script,
    load_v39_instruction_records,
)
from ilm.visual_lm.visual_answer_trajectory_evaluation import (
    VisualAnswerTrajectoryEvaluationOutputs,
    indexed_retrieval_metrics,
    output_effective_rank,
    stop_and_length_metrics,
    trajectory_consistency_metrics,
    trajectory_content_metrics,
)
from ilm.visual_lm.visual_answer_trajectory_training import (
    VisualAnswerTrajectoryTargetBank,
)
from ilm.visual_lm.visual_semantic_distillation import file_sha256
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_SEMANTIC_DIM,
    VisualSemanticDistillationRenderConfig,
    render_visual_semantic_distillation_strip,
)


EXPERIMENT = V39_ARCHITECTURE
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_DEVELOPMENT_BANK = "artifacts/visual_answer_trajectory_v39_targets/development.pt"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
SEED = 20_263_901
VIEW_NAMES = ("canonical", "held_font", "traditional_script", "shuffled", "blank")
OUTPUT_FIELDS = tuple(field.name for field in fields(VisualAnswerTrajectoryEvaluationOutputs))
SOURCE_FILES = (
    "ilm/visual_lm/visual_answer_trajectory.py",
    "ilm/visual_lm/visual_answer_trajectory_data.py",
    "ilm/visual_lm/visual_answer_trajectory_evaluation.py",
    "scripts/eval_visual_answer_trajectory_v39.py",
)


@dataclass(frozen=True)
class InferenceReceipt:
    records: int
    visual_views: int
    elapsed_seconds: float
    records_per_second: float
    raster_views_per_second: float
    script_changed_records: int
    finite: bool


class VisualAnswerTrajectoryEvaluationDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic held-out raster views; text stays on the host side."""

    def __init__(
        self,
        records: Sequence[VisualAnswerTrajectoryRecord],
        *,
        render_config: VisualSemanticDistillationRenderConfig,
        seed: int = SEED,
    ) -> None:
        if len(records) < 2:
            raise ValueError("V39 evaluation requires at least two records")
        if render_config.augment:
            raise ValueError("V39 development rendering must be augmentation-free")
        self.records = tuple(records)
        self.render_config = render_config
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _render(
        self,
        text: str,
        *,
        font_path: str,
        variant: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels, mask, _metadata = render_visual_semantic_distillation_strip(
            text,
            config=self.render_config,
            font_path=font_path,
            font_size=self.render_config.evaluation_font_size,
            variant=variant,
            force_origin=0,
        )
        return pixels, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = self.records[index]
        shuffled = self.records[(index + 1) % len(self.records)]
        traditional = convert_visual_script(record.prompt, "s2t")
        base_variant = self.seed + index * 17
        specifications = {
            "canonical": (record.prompt, V39_DEVELOPMENT_FONT, base_variant),
            "held_font": (record.prompt, V39_HELD_FONT, base_variant + 1),
            "traditional_script": (
                traditional,
                V39_DEVELOPMENT_FONT,
                base_variant + 2,
            ),
            "shuffled": (shuffled.prompt, V39_DEVELOPMENT_FONT, base_variant + 3),
        }
        item: dict[str, torch.Tensor] = {}
        for name, (text, font_path, variant) in specifications.items():
            pixels, mask = self._render(
                text,
                font_path=font_path,
                variant=variant,
            )
            item[f"{name}_pixels"] = pixels
            item[f"{name}_mask"] = mask
        item["blank_pixels"] = torch.ones_like(item["canonical_pixels"])
        item["blank_mask"] = torch.zeros_like(item["canonical_mask"])
        item["script_changed"] = torch.tensor(
            traditional != record.prompt,
            dtype=torch.bool,
        )
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V39 continuous visual answer trajectories."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--development-bank", default=DEFAULT_DEVELOPMENT_BANK)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retrieval-block-size", type=int, default=256)
    parser.add_argument("--maximum-records", type=int, default=0)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V39 requested CUDA but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("V39 evaluation supports CPU or CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tensors_are_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_are_finite(item) for item in value)
    return True


def _iter_tensor_paths(
    value: Any,
    path: tuple[str, ...] = (),
):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensor_paths(item, path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensor_paths(item, path + (str(index),))


def checkpoint_tensor_boundary(checkpoint: Mapping[str, Any]) -> bool:
    # ViT's learned ``cls_token`` is a visual pooling parameter, not a text token.
    forbidden = (
        "target",
        "teacher",
        "candidate",
        "bge",
        "unicode",
        "ocr",
        "input_id",
        "vocab",
        "token_embed",
        "token_embedding",
    )
    return not any(
        any(fragment in ".".join(path).lower() for fragment in forbidden)
        for path, _tensor in _iter_tensor_paths(checkpoint)
    )


def load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    raw_weights: bool,
) -> tuple[VisualAnswerTrajectoryModel, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V39 checkpoint must contain a state mapping")
    if checkpoint.get("architecture") != V39_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V39")
    forbidden_flags = (
        "contains_target_tensors",
        "contains_teacher_model",
        "contains_candidate_tensors",
        "contains_source_language_strings",
    )
    if any(bool(checkpoint.get(name, True)) for name in forbidden_flags):
        raise ValueError("V39 checkpoint violates the deployable tensor boundary")
    config_state = checkpoint.get("model_config")
    model_state = checkpoint.get("model")
    if not isinstance(config_state, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("V39 checkpoint lacks its model configuration or state")
    config = VisualAnswerTrajectoryConfig(**dict(config_state))
    model = VisualAnswerTrajectoryModel(config)
    model.load_state_dict(model_state, strict=True)
    ema = checkpoint.get("ema")
    if raw_weights:
        weight_route = (
            "all-parameter-ema"
            if checkpoint.get("weight_route") == "all-parameter-ema" and not ema
            else "raw"
        )
    elif isinstance(ema, Mapping) and isinstance(ema.get("shadow"), Mapping):
        parameters = dict(model.named_parameters())
        names = tuple(str(name) for name in ema.get("names", ()))
        shadow = ema["shadow"]
        if set(names) != set(parameters) or set(shadow) != set(parameters):
            raise ValueError("V39 checkpoint EMA is not all-parameter")
        for name, value in shadow.items():
            if not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape:
                raise ValueError(f"V39 EMA parameter mismatch: {name}")
            parameters[name].data.copy_(value.to(parameters[name]))
        weight_route = "all-parameter-ema"
    elif checkpoint.get("weight_route") == "all-parameter-ema":
        weight_route = "all-parameter-ema"
    else:
        raise ValueError("V39 checkpoint has no all-parameter EMA state")
    model.requires_grad_(False).eval().to(device)
    boundary = visual_answer_trajectory_boundary_receipt(model)
    receipt = {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "weight_route": weight_route,
        "global_update": int(checkpoint.get("global_update", 0)),
        "finite_model_state": tensors_are_finite(model.state_dict()),
        "tensor_boundary": checkpoint_tensor_boundary(checkpoint),
        "boundary": boundary,
    }
    if not receipt["finite_model_state"]:
        raise FloatingPointError("V39 checkpoint model is non-finite")
    return model, dict(checkpoint), receipt


def load_development_bank(
    path: str | Path,
    *,
    manifest_sha256: str,
) -> VisualAnswerTrajectoryTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V39 development bank must contain a state mapping")
    bank = VisualAnswerTrajectoryTargetBank.from_state_dict(state)
    receipt = bank.receipt
    if receipt.get("split") != "development":
        raise ValueError("V39 evaluation requires a development target bank")
    if receipt.get("data", {}).get("sha256") != manifest_sha256:
        raise ValueError("V39 development bank has another instruction corpus")
    if bank.prompt_targets.shape[1] != V37_SEMANTIC_DIM:
        raise ValueError("V39 development target width differs from the model")
    if receipt.get("source_text_strings_stored", True):
        raise ValueError("V39 development bank declares stored source text")
    if receipt.get("token_ids_stored", True) or receipt.get("unicode_ids_stored", True):
        raise ValueError("V39 development bank declares symbolic target payloads")
    if receipt.get("student_runtime_teacher_dependency", True):
        raise ValueError("V39 development bank declares a runtime teacher")
    if any(
        value.device.type != "cpu"
        for value in (
            bank.prompt_targets,
            bank.answer_targets,
            bank.segment_targets,
            bank.segment_offsets,
            bank.segment_lengths,
        )
    ):
        raise ValueError("V39 evaluation candidates must remain on CPU")
    return bank


def select_bank_records(
    records: Sequence[VisualAnswerTrajectoryRecord],
    identifiers: Sequence[str],
) -> list[VisualAnswerTrajectoryRecord]:
    by_identifier = {record.identifier: record for record in records}
    try:
        return [by_identifier[identifier] for identifier in identifiers]
    except KeyError as error:
        raise KeyError(f"V39 corpus lacks development record {error.args[0]!r}") from error


def _split_output(
    value: torch.Tensor,
    *,
    batch_size: int,
    view_index: int,
) -> torch.Tensor:
    start = view_index * batch_size
    return value[start : start + batch_size].detach().float().cpu().half()


@torch.no_grad()
def infer_evaluation_views(
    model: VisualAnswerTrajectoryModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, VisualAnswerTrajectoryEvaluationOutputs], InferenceReceipt]:
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        view: {name: [] for name in OUTPUT_FIELDS} for view in VIEW_NAMES
    }
    records = 0
    changed = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        batch_size = int(batch["canonical_pixels"].shape[0])
        pixels = torch.cat(
            [batch[f"{view}_pixels"] for view in VIEW_NAMES],
            dim=0,
        ).to(device, non_blocking=device.type == "cuda")
        masks = torch.cat(
            [batch[f"{view}_mask"] for view in VIEW_NAMES],
            dim=0,
        ).to(device, non_blocking=device.type == "cuda")
        with autocast_context(device, precision):
            output = model.generate_plan(pixels, masks)
        for view_index, view in enumerate(VIEW_NAMES):
            for name in OUTPUT_FIELDS:
                collected[view][name].append(
                    _split_output(
                        getattr(output, name),
                        batch_size=batch_size,
                        view_index=view_index,
                    )
                )
        records += batch_size
        changed += int(batch["script_changed"].sum())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    outputs = {
        view: VisualAnswerTrajectoryEvaluationOutputs(
            **{name: torch.cat(values) for name, values in collected[view].items()}
        )
        for view in VIEW_NAMES
    }
    for value in outputs.values():
        value.validate()
    finite = all(
        bool(torch.isfinite(tensor).all())
        for output in outputs.values()
        for tensor in output.__dict__.values()
    )
    return outputs, InferenceReceipt(
        records=records,
        visual_views=len(VIEW_NAMES),
        elapsed_seconds=elapsed,
        records_per_second=records / max(elapsed, 1e-9),
        raster_views_per_second=records * len(VIEW_NAMES) / max(elapsed, 1e-9),
        script_changed_records=changed,
        finite=finite,
    )


def global_state_metrics(
    states: torch.Tensor,
    candidates: torch.Tensor,
    *,
    indices: torch.Tensor | None = None,
    block_size: int,
) -> dict[str, float | int]:
    if indices is None:
        indices = torch.arange(len(states), dtype=torch.long)
    if indices.dtype != torch.long or indices.ndim != 1 or not len(indices):
        raise ValueError("V39 global evaluation indices must be non-empty long values")
    return indexed_retrieval_metrics(
        states[indices],
        candidates,
        indices,
        block_size=block_size,
    )


def route_metrics(
    outputs: VisualAnswerTrajectoryEvaluationOutputs,
    bank: VisualAnswerTrajectoryTargetBank,
    *,
    records: int,
    block_size: int,
) -> dict[str, Any]:
    labels = torch.arange(records, dtype=torch.long)
    answer_targets = bank.answer_targets
    prompt_targets = bank.prompt_targets
    return {
        "read_prompt": global_state_metrics(
            outputs.read_state,
            prompt_targets,
            indices=labels,
            block_size=block_size,
        ),
        "baseline_answer": global_state_metrics(
            outputs.baseline_answer_state,
            answer_targets,
            indices=labels,
            block_size=block_size,
        ),
        "stage1_answer": global_state_metrics(
            outputs.stage1_answer_state,
            answer_targets,
            indices=labels,
            block_size=block_size,
        ),
        "final_answer": global_state_metrics(
            outputs.answer_state,
            answer_targets,
            indices=labels,
            block_size=block_size,
        ),
    }


def metric_delta(
    reference: Mapping[str, float | int],
    comparison: Mapping[str, float | int],
) -> dict[str, float]:
    names = ("top1", "top5", "mrr", "positive_cosine")
    return {
        f"{name}_delta": float(reference[name]) - float(comparison[name])
        for name in names
    }


def operation_bucket(prompt: str) -> str:
    value = prompt.casefold()
    categories = (
        ("translate", ("翻译", "译成", "译为", "translate")),
        ("summarize", ("总结", "概括", "摘要", "summar")),
        ("classify", ("分类", "判断", "选择", "属于", "classif")),
        ("generate", ("写一", "写出", "创作", "生成", "列出", "拟定", "write")),
        ("explain", ("解释", "说明", "为什么", "什么是", "如何", "explain", "why")),
    )
    for name, markers in categories:
        if any(marker in value for marker in markers):
            return name
    return "other"


def stratum_metrics(
    outputs: VisualAnswerTrajectoryEvaluationOutputs,
    bank: VisualAnswerTrajectoryTargetBank,
    records: Sequence[VisualAnswerTrajectoryRecord],
    *,
    block_size: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {
        "answer_short_le_32": [],
        "answer_long_gt_32": [],
    }
    for index, record in enumerate(records):
        length_name = "answer_long_gt_32" if len(record.answer) > 32 else "answer_short_le_32"
        groups[length_name].append(index)
        groups.setdefault(f"operation_{operation_bucket(record.prompt)}", []).append(index)
    result: dict[str, Any] = {}
    for name, values in sorted(groups.items()):
        if not values:
            continue
        indices = torch.tensor(values, dtype=torch.long)
        baseline = global_state_metrics(
            outputs.baseline_answer_state,
            bank.answer_targets,
            indices=indices,
            block_size=block_size,
        )
        final = global_state_metrics(
            outputs.answer_state,
            bank.answer_targets,
            indices=indices,
            block_size=block_size,
        )
        result[name] = {
            "records": len(values),
            "baseline": baseline,
            "final": final,
            "final_minus_baseline": {
                key: float(final[key]) - float(baseline[key])
                for key in ("top1", "top5", "mrr", "positive_cosine")
            },
        }
    return result


def main() -> None:
    args = parse_args()
    if args.smoke and args.exploratory:
        raise ValueError("V39 smoke and exploratory labels are mutually exclusive")
    if not args.smoke and not args.exploratory:
        raise RuntimeError("V39 has no frozen evidence protocol; use --exploratory")
    if min(args.batch_size, args.retrieval_block_size) < 1:
        raise ValueError("V39 evaluation batch settings must be positive")
    if args.num_workers < 0 or args.maximum_records < 0:
        raise ValueError("V39 evaluation worker or record limit is invalid")
    manifest_sha256 = file_sha256(args.instruction_manifest)
    if not args.smoke and manifest_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise RuntimeError("V39 instruction data differs from the fixed corpus")

    device = choose_device(args.device)
    if device.type == "cpu" and args.precision != "fp32":
        raise ValueError("V39 CPU evaluation requires --precision fp32")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    model, checkpoint, checkpoint_receipt = load_checkpoint_model(
        args.checkpoint,
        device=device,
        raw_weights=args.raw_weights,
    )
    bank = load_development_bank(
        args.development_bank,
        manifest_sha256=manifest_sha256,
    )
    all_records = load_v39_instruction_records(args.instruction_manifest)
    records = select_bank_records(all_records, bank.identifiers)
    if args.maximum_records:
        records = records[: args.maximum_records]
    if len(records) < 2:
        raise ValueError("V39 evaluation selected fewer than two records")
    record_count = len(records)
    segment_stop = int(bank.segment_offsets[record_count])
    offsets = bank.segment_offsets[: record_count + 1]
    segment_targets = bank.segment_targets[:segment_stop]
    segment_lengths = bank.segment_lengths[:segment_stop]

    render_config = VisualSemanticDistillationRenderConfig(augment=False)
    dataset = VisualAnswerTrajectoryEvaluationDataset(
        records,
        render_config=render_config,
        seed=SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    outputs, inference = infer_evaluation_views(
        model,
        loader,
        device=device,
        precision=args.precision,
    )
    peak_vram = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    canonical = outputs["canonical"]
    routes = route_metrics(
        canonical,
        bank,
        records=record_count,
        block_size=args.retrieval_block_size,
    )
    final_trajectory = trajectory_content_metrics(
        canonical.segment_states,
        segment_targets,
        offsets,
        block_size=args.retrieval_block_size,
    )
    stage1_trajectory = trajectory_content_metrics(
        canonical.stage1_segment_states,
        segment_targets,
        offsets,
        block_size=args.retrieval_block_size,
    )
    view_routes = {
        name: route_metrics(
            value,
            bank,
            records=record_count,
            block_size=args.retrieval_block_size,
        )
        for name, value in outputs.items()
        if name != "canonical"
    }
    controls: dict[str, Any] = {}
    for name in ("shuffled", "blank"):
        trajectory = trajectory_content_metrics(
            outputs[name].segment_states,
            segment_targets,
            offsets,
            block_size=args.retrieval_block_size,
        )
        controls[name] = {
            "routes": view_routes[name],
            "trajectory": trajectory,
            "final_answer_drop": metric_delta(
                routes["final_answer"],
                view_routes[name]["final_answer"],
            ),
            "trajectory_paired_cosine_drop": (
                float(final_trajectory["paired_cosine"])
                - float(trajectory["paired_cosine"])
            ),
            "canonical_consistency": trajectory_consistency_metrics(
                canonical,
                outputs[name],
                offsets,
            ),
        }

    font_and_script = {
        "held_font": {
            "font_path": V39_HELD_FONT,
            "routes": view_routes["held_font"],
            "canonical_consistency": trajectory_consistency_metrics(
                canonical,
                outputs["held_font"],
                offsets,
            ),
        },
        "traditional_script": {
            "conversion": "OpenCC s2t before the pixel-only boundary",
            "changed_records": inference.script_changed_records,
            "unchanged_records": record_count - inference.script_changed_records,
            "routes": view_routes["traditional_script"],
            "canonical_consistency": trajectory_consistency_metrics(
                canonical,
                outputs["traditional_script"],
                offsets,
            ),
        },
    }
    stage_improvement = {
        "final_minus_baseline": {
            key: float(routes["final_answer"][key])
            - float(routes["baseline_answer"][key])
            for key in ("top1", "top5", "mrr", "positive_cosine")
        },
        "final_minus_stage1": {
            key: float(routes["final_answer"][key])
            - float(routes["stage1_answer"][key])
            for key in ("top1", "top5", "mrr", "positive_cosine")
        },
        "final_segment_minus_stage1": {
            key: float(final_trajectory[key]) - float(stage1_trajectory[key])
            for key in (
                "paired_cosine",
                "exact_beats_next",
                "transition_direction_cosine",
            )
        },
    }
    boundary = checkpoint_receipt["boundary"]
    report: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "exploratory",
        "split": "development",
        "claim_status": (
            "continuous visual trajectory experiment; no pixel-language generation claim"
        ),
        "checkpoint": checkpoint_receipt,
        "training": {
            "global_update": int(checkpoint.get("global_update", 0)),
            "finite": bool(checkpoint.get("finite", True)),
            "stage_summaries": checkpoint.get("stage_summaries", {}),
        },
        "data": {
            "instruction_manifest": str(Path(args.instruction_manifest).resolve()),
            "instruction_sha256": manifest_sha256,
            "development_bank": str(Path(args.development_bank).resolve()),
            "development_bank_sha256": file_sha256(args.development_bank),
            "development_bank_label": bank.receipt.get("label"),
            "bank_records": len(bank.identifiers),
            "evaluated_records": record_count,
            "evaluated_segments": segment_stop,
            "sealed_rows_rendered": 0,
            "target_tensors_moved_to_accelerator": False,
        },
        "runtime_contract": {
            "model_inputs": ["prompt_pixels[B,3,16,1024]", "prompt_mask[B,64]"],
            "model_outputs": "continuous global and ordered segment states",
            "uses_token_ids": bool(boundary["uses_token_ids"]),
            "uses_unicode_ids": bool(boundary["uses_unicode_ids"]),
            "uses_ocr": bool(boundary["uses_ocr"]),
            "uses_runtime_teacher": bool(
                boundary["uses_bge_at_runtime"] or boundary["uses_qwen_at_runtime"]
            ),
            "uses_runtime_script_converter": bool(boundary["uses_opencc_at_runtime"]),
            "offline_text_used_for_rasterization_and_scoring_only": True,
            "renderer_opened": False,
        },
        "resources": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
            ),
            "precision": args.precision,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "peak_vram_bytes": int(peak_vram),
            "inference": asdict(inference),
        },
        "canonical": {
            "font_path": V39_DEVELOPMENT_FONT,
            "routes": routes,
            "final_trajectory": final_trajectory,
            "stage1_trajectory": stage1_trajectory,
            "stop_and_length": stop_and_length_metrics(
                canonical,
                offsets,
                segment_lengths,
            ),
            "effective_rank": output_effective_rank(canonical, offsets),
            "stage_improvement": stage_improvement,
        },
        "font_and_script": font_and_script,
        "controls": controls,
        "strata": stratum_metrics(
            canonical,
            bank,
            records,
            block_size=args.retrieval_block_size,
        ),
        "diagnostic_signals": {
            "final_answer_improves_baseline_mrr": (
                routes["final_answer"]["mrr"] > routes["baseline_answer"]["mrr"]
            ),
            "final_segments_improve_stage1_cosine": (
                final_trajectory["paired_cosine"]
                > stage1_trajectory["paired_cosine"]
            ),
            "exact_segment_position_beats_next_more_than_half": (
                final_trajectory["exact_beats_next"] > 0.5
            ),
            "shuffled_prompt_reduces_answer_cosine": (
                controls["shuffled"]["final_answer_drop"]["positive_cosine_delta"] > 0
            ),
            "blank_prompt_reduces_answer_cosine": (
                controls["blank"]["final_answer_drop"]["positive_cosine_delta"] > 0
            ),
            "all_outputs_finite": inference.finite,
        },
        "protocol": {
            "frozen": False,
            "selection_allowed": False,
            "renderer_gate_open": False,
            "next_action": "diagnose development behavior before freezing a protocol",
        },
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "finite": bool(inference.finite),
    }
    output_path = Path(args.out) if args.out else Path(args.checkpoint).parent / (
        "development_report_raw_v39.json"
        if args.raw_weights
        else "development_report_ema_v39.json"
    )
    atomic_write_json(report, output_path)
    print(
        json.dumps(
            {
                "report": str(output_path.resolve()),
                "weight_route": checkpoint_receipt["weight_route"],
                "records": record_count,
                "segments": segment_stop,
                "final_answer": routes["final_answer"],
                "final_segment": final_trajectory["retrieval"],
                "diagnostic_signals": report["diagnostic_signals"],
                "peak_vram_bytes": int(peak_vram),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
