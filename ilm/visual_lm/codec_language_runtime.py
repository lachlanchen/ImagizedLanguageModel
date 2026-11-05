"""Load completed V68 adaptation into its source-authenticated image actor.

The shared archive initializes fixed weights at loading. There is no source
model service, text vocabulary, renderer or reference-image path in generation.
A standalone weight bundle still requires a separate export and isolation test.
"""

import json
from pathlib import Path

import torch

from .codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from .codec_visual_language_actor import CodecVisualLanguageActor
from .grounded_image_weights import file_sha, load_grounded_image_actor


def completed_identity(run):
    """Refuse unfinished training before initializing CUDA or reading weights."""
    run = Path(run)
    report = json.loads((run / "training_report.json").read_text())
    if (report["update"] != 10000 or report["trained_parameters"] != 9351937
            or report["development_examples"] != 0 or report["teacher_calls"] != 0
            or report["language_quality_claim"] or report["config"]["use_ema"]):
        raise ValueError("the complete registered instruction run is required")
    replay = json.loads((run / "resume_verification.json").read_text())
    if not replay["passed"] or not all(replay["checks"].values()):
        raise ValueError("complete production resume verification required")
    if file_sha(run / "actor_trainable_final.pt") != report["final_trainable_sha256"]:
        raise ValueError("fixed final adaptation changed")
    return report


def load_completed_actor(run, *, root, device, progress=None):
    run, root = Path(run), Path(root)
    report = completed_identity(run)
    for name, digest in report["sources"].items():
        if file_sha(root / name) != digest:
            raise ValueError("accepted training implementation changed")
    for name, digest in report["prerequisites"].items():
        if file_sha(root / name) != digest:
            raise ValueError("accepted execution prerequisite changed")
    saved = torch.load(run / "actor_trainable_final.pt", map_location="cpu", weights_only=True)
    for key in ("sources", "prerequisites", "config", "writing_report_sha256"):
        if saved[key] != report[key]:
            raise ValueError("final adaptation and training report differ")
    if saved["update"] != 10000 or saved["standalone_export"] or saved["language_quality_evaluated"]:
        raise ValueError("unexpected final adaptation identity")
    base, provenance = load_grounded_image_actor(
        ollama_root=root.parent / "LocalLLM/.local/models/ollama",
        gguf_source=root.parent / "LocalLLM/.local/tools/llama.cpp/gguf-py",
        device=device, progress=progress)
    if provenance != report["initialization"] or provenance != saved["initialization"]:
        raise ValueError("inherited source weights differ from training")
    base.writer = CodecBlockWriter(load_qualified_writing_codec(
        root / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt", device))
    actor = CodecVisualLanguageActor(base)
    actor.load_trainable_state(saved["trainable"])
    return actor.requires_grad_(False).eval(), report
