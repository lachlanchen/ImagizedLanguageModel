"""Export the completed visual-context comparison without modifying its evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/trainable_visual_context_comparison_v1"
REG = ROOT / "references/evidence/trainable_visual_context_comparison_registration_20261116.json"
OUTPUT = ROOT / "references/evidence/trainable_visual_context_result_20261117.json"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize():
    review = RUN / "review"
    frozen = load(review / "readings_frozen.json")
    readings = load(review / "readings_live.json")
    scores = load(review / "scores.json")
    require(sha(review / "readings_live.json") == frozen["readings_sha256"]
            == scores["readings_sha256"], "Frozen readings changed")
    require(sha(review / "mapping.json") == frozen["mapping_sha256"], "Mapping changed")
    require(scores["registration_sha256"] == sha(REG), "Registration changed")
    require(scores["development_stage_passed"] is False, "This export records the rejected comparison")
    require(frozen["images"] == len(scores["records"]) == len(readings["readings"]) == 792,
            "Incomplete comparison")
    # These receipts were written during opaque review, before freeze or scoring.
    receipts = sorted((review / "reading_batches").glob("batch_*.json"))
    seen = {}
    boards = []
    for path in receipts:
        batch = load(path)
        require(batch["references_opened"] is False and batch["mapping_opened"] is False,
                "Batch was not recorded under opaque review")
        for row in batch["records"]:
            require(row["id"] not in seen, "Duplicate reviewed image")
            require(row["reading"] == readings["readings"][row["id"]],
                    "Recorded reading was subsequently changed")
            seen[row["id"]] = row["reading"]
        boards.extend(batch["boards"])
    require(seen.keys() == readings["readings"].keys(), "Missing visual review receipts")
    require(boards == [f"board_{n:03d}.png" for n in range(180)], "Incomplete board review")
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in
                [REG, review / "scores.json", review / "readings_live.json",
                 review / "readings_frozen.json", review / "mapping.json",
                 review / "boards_v1/manifest.json", *receipts]}
    arms = {}
    for arm in ("trainable", "frozen"):
        report_path = RUN / arm / "fit/report.json"
        audit_path = RUN / arm / "fit/audit.json"
        cold_path = RUN / arm / "cold_deployment_v1/verification.json"
        report, audit, cold = map(load, (report_path, audit_path, cold_path))
        for path in (report_path, audit_path, cold_path):
            bindings[str(path.relative_to(ROOT))] = sha(path)
        require(audit["audit_passed"] and audit["updates"] == report["updates"] == 8172,
                "Incomplete or unaudited fit")
        require(audit["final_checkpoint_sha256"] == report["final_checkpoint_sha256"]
                == cold["checkpoint_sha256"], "Final model identities differ")
        require(cold["actual_pixels_stopping_and_feedback_exact"]
                and not cold["forbidden_paths"] and not cold["IP_connection_attempts"]
                and not cold["other_project_artifacts_read"]
                and not cold["obsolete_metadata_exemption_added"], "Strict cold check failed")
        rows = [row for row in scores["records"] if row["arm"] == arm]
        require(len(rows) == 396, "Incomplete arm")
        final = next(row for row in scores["summaries"] if row["arm"] == arm and row["step"] == 8172)
        arms[arm] = dict(
            updates=report["updates"], training_records=report["training_records"],
            source_target_glyph_positions=report["target_glyph_positions"],
            source_teacher_states=report["causal_state_targets"],
            total_deployed_parameters=report["total_model_parameters"],
            training_parameters_including_offline_projection=report["training_parameters"],
            optimizer_moment_bytes=report["optimizer_moment_bytes"],
            final_checkpoint_sha256=report["final_checkpoint_sha256"],
            fit_resources={key: report["resources"][key] for key in
                           ("elapsed_seconds", "resource_wait_seconds", "peak_allocated_vram_gib",
                            "minimum_available_gib")},
            original_fit_audit_passed=True, strict_fixed_case_cold_check_passed=True,
            streams=396, final_summary=final,
            visual_review_counts=dict(Counter("uncertain" if row["reading"]["uncertain"]
                else "blank" if not row["reading"]["text"] else "nonempty_certain" for row in rows)))
    return dict(schema="ilm-completed-visual-context-result-v1", date="2026-11-17",
                decision="reject_complete_six_objective_visual_span_recipe",
                development_stage_passed=scores["development_stage_passed"],
                goal_achieved=False, reviewer=readings["evaluator"],
                images_reviewed=792, native_boards_reviewed=180,
                normal_book_visible_prefixes=144, frozen_before_reference_scoring=True,
                uncertain_region_notation="□ is an unreadable region, not a counted glyph",
                arms=arms, all_snapshot_summaries=scores["summaries"], bindings=bindings,
                review_receipt_bindings_verified=True, new_model_calls_for_export=0,
                limitations=["One model reviewer with prior development familiarity; no independent human adjudication.",
                    "Unreadable writing prevents assessment of latent semantic correctness.",
                    "Natural stopping and cold execution do not establish useful language.",
                    "First and last shard losses are different batches, not a fixed-set learning curve.",
                    "Source position counts are not independent counts of conditional training exposures.",
                    "Fit resource scopes exclude earlier inherited pretraining, teacher preparation and failed executions.",
                    "Proportional layout, untouched confirmation, historical forms, etymology and matched-quality savings remain unproved."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = summarize()
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.verify:
        require(OUTPUT.read_text() == payload, "Public result differs from retained evidence")
    else:
        with OUTPUT.open("x") as stream:
            stream.write(payload)
    print(json.dumps(dict(result=str(OUTPUT.relative_to(ROOT)), sha256=sha(OUTPUT),
                          streams=792, development_stage_passed=result["development_stage_passed"],
                          goal_achieved=False, verified=args.verify)))
