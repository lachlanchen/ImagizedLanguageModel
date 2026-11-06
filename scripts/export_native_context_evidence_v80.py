#!/usr/bin/env python3
"""Export completed context diagnostics without weights or private runtime state."""
import json
from scripts.cache_semantic_images_v72 import ROOT, atomic_json, sha


def main():
    suffix = ROOT / "artifacts/suffix_kv_v78_20261106"
    execution = ROOT / "artifacts/native_image_execution_v79r2_20261106"
    verification = json.loads((suffix / "verification.json").read_text())
    generation = json.loads((suffix / "generation_report.json").read_text())
    assert verification["offline_checks_pass"]
    assert verification["generation_report_sha256"] == sha(suffix / "generation_report.json")
    native = json.loads((execution / "report.json").read_text())
    replay = json.loads((execution / "supplemental_verification.json").read_text())
    assert replay["offline_checks_pass"] and replay["original_report_sha256"] == sha(execution / "report.json")
    target = ROOT / "references/evidence/native_context_v80.json"
    atomic_json(target, dict(
        suffix_visibility=dict(verification_sha256=sha(suffix / "verification.json"),
            generation_report_sha256=sha(suffix / "generation_report.json"),
            source_commit=verification["identity"]["source_commit"], summaries=verification["summaries"],
            measurements=verification["measurements"], first_decisions_equal=verification["first_decisions_equal"],
            normal_reader_memories_equal=verification["normal_reader_memories_equal"],
            masked_poison_states_identical=verification["masked_poison_states_identical"],
            helper_projection_positions=verification["helper_projection_positions"],
            helper_projection_argmax_agreements=verification["helper_projection_argmax_agreements"],
            suffix_layerwise_bytes=verification["suffix_layerwise_bytes"],
            generation_seconds=generation["elapsed_seconds"],
            peak_allocated_vram_gib=generation["peak_allocated_vram_gib"],
            source_kv_physically_deleted=False, whole_context_forward_independently_replayed=False),
        native_execution=dict(report_sha256=sha(execution / "report.json"),
            supplemental_verification_sha256=sha(execution / "supplemental_verification.json"),
            **{key:native[key] for key in ("source_layers_exercised", "trainable_parameters",
                "complete_with_prefill_vision_parameters", "forward_max_abs_error",
                "checkpoint_forward_max_abs_error", "padding_max_abs_error", "future_max_abs_error",
                "incremental_max_abs_error", "physical_feedback_exact", "frozen_context_unchanged",
                "prefix_unchanged", "peak_allocated_vram_gib", "elapsed_seconds")},
            replay=replay, optimizer_updates=0, trained=False),
        language_quality_claim=False, general_ilm_goal_complete=False))
    print(target.relative_to(ROOT))


if __name__ == "__main__": main()
