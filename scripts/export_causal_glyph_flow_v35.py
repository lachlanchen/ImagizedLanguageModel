#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from ilm.visual_lm.causal_glyph_flow import (
    V35_ARCHITECTURE,
    causal_glyph_flow_boundary_receipt,
    file_sha256,
)
from ilm.visual_lm.causal_glyph_flow_development import load_v35_checkpoint_model
from ilm.visual_lm.direct_visual_patch_training import module_state_sha256


STANDALONE_ARTIFACT = "causal-glyph-flow-v35-standalone"
DEFAULT_TRAINING_CHECKPOINT = (
    "artifacts/causal_glyph_flow_v35_20260814/checkpoint_latest.pt"
)
DEFAULT_DEVELOPMENT_REPORT = (
    "artifacts/causal_glyph_flow_v35_20260814/development/development_report.json"
)
DEFAULT_OUTPUT = "artifacts/causal_glyph_flow_v35_20260814/ilm_v35_ema_standalone.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a self-contained V35 raster-in/raster-out checkpoint."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    parser.add_argument("--development-report", default=DEFAULT_DEVELOPMENT_REPORT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--state", choices=("raw", "ema"), default="ema")
    parser.add_argument("--allow-unqualified", action="store_true")
    return parser.parse_args()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def standalone_checkpoint_is_clean(payload: Mapping[str, Any]) -> bool:
    state = payload.get("model")
    return (
        payload.get("artifact") == STANDALONE_ARTIFACT
        and payload.get("architecture") == V35_ARCHITECTURE
        and isinstance(state, Mapping)
        and all(isinstance(value, torch.Tensor) for value in state.values())
        and payload.get("optimizer") is None
        and payload.get("rng") is None
        and payload.get("runtime_teacher") is None
        and payload.get("ocr") is None
        and payload.get("tokenizer") is None
        and payload.get("retrieval") is None
        and payload.get("resumable") is False
    )


def build_standalone_payload(
    training_checkpoint: str | Path,
    development_report: str | Path,
    *,
    state: str = "ema",
    allow_unqualified: bool = False,
) -> dict[str, Any]:
    report_path = Path(development_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = report.get("decision", {})
    status = str(decision.get("status", "not-qualified"))
    if status == "not-qualified" and not allow_unqualified:
        raise RuntimeError(
            "V35 standalone export requires a qualified development report; "
            "use --allow-unqualified only for explicitly labeled diagnostics"
        )
    model, checkpoint, receipt = load_v35_checkpoint_model(
        training_checkpoint,
        device=torch.device("cpu"),
        state=state,
    )
    planned = sum(
        int(stage["updates"])
        for stage in checkpoint.get("run_receipt", {}).get("stages", [])
    )
    if int(checkpoint.get("global_update", -1)) != planned or planned < 1:
        raise RuntimeError("V35 standalone export requires completed training")
    report_receipt = report.get("checkpoint", {}).get(state, {})
    if report_receipt.get("checkpoint_sha256") != receipt["checkpoint_sha256"]:
        raise RuntimeError("V35 development report evaluated a different checkpoint")
    writer = str(decision.get("selected_writer", "anchor"))
    if writer not in {"anchor", "flow"}:
        raise ValueError("V35 development report selected an invalid writer")
    boundary = causal_glyph_flow_boundary_receipt(model)
    payload = {
        "artifact": STANDALONE_ARTIFACT,
        "architecture": V35_ARCHITECTURE,
        "model_config": receipt["model_config"],
        "model": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "model_state_sha256": module_state_sha256(model),
        "weight_state": state,
        "writer": writer,
        "generation": {
            "maximum_new_patches": 31,
            "minimum_new_patches": 1,
            "stop_threshold": 0.5,
            "raster_threshold": 0.5,
            "flow_steps": 8,
            "seed": 20_263_535,
        },
        "decision": {
            "status": status,
            "evidence_eligible": bool(report.get("evidence_eligible", False)),
        },
        "provenance": {
            "training_checkpoint": str(Path(training_checkpoint).resolve()),
            "training_checkpoint_sha256": receipt["checkpoint_sha256"],
            "development_report": str(report_path.resolve()),
            "development_report_sha256": file_sha256(report_path),
            "global_update": receipt["global_update"],
            "external_initialization": checkpoint.get("run_receipt", {}).get(
                "initialization"
            ),
            "weight_redistribution_authorized": False,
        },
        "boundary": boundary,
        "optimizer": None,
        "rng": None,
        "runtime_teacher": None,
        "ocr": None,
        "tokenizer": None,
        "retrieval": None,
        "resumable": False,
    }
    if not standalone_checkpoint_is_clean(payload):
        raise RuntimeError("V35 standalone artifact failed its clean-state audit")
    return payload


def main() -> None:
    args = parse_args()
    payload = build_standalone_payload(
        args.checkpoint,
        args.development_report,
        state=args.state,
        allow_unqualified=args.allow_unqualified,
    )
    output = Path(args.out)
    _atomic_torch_save(payload, output)
    receipt = {
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": file_sha256(output),
        "model_state_sha256": payload["model_state_sha256"],
        "status": payload["decision"]["status"],
        "writer": payload["writer"],
        "clean": standalone_checkpoint_is_clean(payload),
        "redistribution_authorized": False,
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
