#!/usr/bin/env python3
"""Authenticate complete native QA outputs and measure paired question effects."""
import argparse
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import save_json, sha
from ilm.image_edit.qa_learning import (INITIAL_SHA, ROOT, SOURCE, dataset, digest,
    identity, schedule)
from scripts.evaluate_image_edit_baseline import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args(); run = args.run.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("independent assessment must hide CUDA")
    import torch
    from scripts.export_balanced_autonomous_writer import tensor_png
    train, held = dataset(); source = {r["id"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    manifests = {r["id"]: r for r in train + held}
    prepared = json.loads((run / "preparation.json").read_text())
    if identity() != prepared["code_sha256"] or digest(schedule(train)) != prepared["schedule_sha256"]:
        raise ValueError("unchanged registered complete assessment required")
    trained = json.loads((run / "training/report.json").read_text())
    checkpoint = torch.load(run / "training/checkpoint.pt", map_location="cpu", weights_only=True)
    if (trained["updates"] != 128 or checkpoint["step"] != 128 or len(checkpoint["history"]) != 128
            or trained["checkpoint_sha256"] != sha(run / "training/checkpoint.pt")
            or checkpoint["metadata"]["schedule_sha256"] != prepared["schedule_sha256"]
            or any(int(s["step"]) != 128 for s in checkpoint["optimizer"]["state"].values())):
        raise ValueError("complete actual optimization required")
    for actual, planned in zip(checkpoint["history"], schedule(train)):
        if any(actual[k] != v for k, v in planned.items()): raise ValueError("actual training schedule differs")
        folder = run / "training" / actual["folder"]; out = actual["autonomous_page"]
        tensor_png(folder / "pixels.pt", folder / "answer.png", out["pixels_sha256"], out["png_sha256"])
    final_sha = trained["checkpoint_sha256"]; del checkpoint
    observations, summaries, bindings = [], {}, {}
    for arm, expected_checkpoint in (("initial", INITIAL_SHA), ("final", final_sha)):
        summaries[arm] = {}
        for variant in ("normal", "question_erased", "passage_erased"):
            name = arm + "_" + variant; folder = run / (name + "_inference")
            report = json.loads((folder / "report.json").read_text())
            index = json.loads((run / (variant + "_inputs.json")).read_text())
            outputs = json.loads((folder / "outputs.json").read_text())
            scores = json.loads((run / (name + "_ocr/report.json")).read_text())
            if ([r["id"] for r in outputs] != [r["id"] for r in index]
                    or [r["id"] for r in scores["rows"]] != [r["id"] for r in index]
                    or report["checkpoint_sha256"] != expected_checkpoint
                    or report["inputs_sha256"] != sha(run / (variant + "_inputs.json"))
                    or report["complete_generations"] != len(index)
                    or any(report[k] for k in ("reference_answer_used", "training_manifest_loaded", "question_specific_text_condition", "output_repair"))):
                raise ValueError("fixed complete input-only generation required")
            if variant == "normal" and len(index) != 64 or variant != "normal" and len(index) != 32:
                raise ValueError("complete normal/erasure cohorts required")
            group = []
            for actual, given, scored in zip(outputs, index, scores["rows"]):
                if any(actual[k] != given[k] for k in ("id", "seed", "input_sha256")) or scored["png_sha256"] != actual["sha256"]:
                    raise ValueError("actual input/seed/scoring binding differs")
                path = folder / actual["path"]; out = actual["result"]
                tensor_png(path.parent / "pixels.pt", path, out["pixels_sha256"], out["png_sha256"])
                accepted = sorted({a["text"] for a in source[actual["id"]]["references"]})
                row = dict(arm=arm, variant=variant, id=actual["id"], split=manifests[actual["id"]]["split"],
                    pair=manifests[actual["id"]]["pair"], raw_transcript=scored["raw_transcript"],
                    source_accepted_answers=accepted, canonical_exact=scored["ocr_exact"],
                    accepted_exact=normalize(scored["raw_transcript"], "zh") in {normalize(a, "zh") for a in accepted},
                    answer_png=str(path.relative_to(run)), answer_sha256=actual["sha256"],
                    input_png=given["input"], input_sha256=given["input_sha256"], seed=given["seed"])
                group.append(row); observations.append(row)
            if variant == "question_erased":
                for i in range(0, 32, 2):
                    a, b = outputs[i:i+2]; ia, ib = index[i:i+2]
                    tensors = [torch.load((folder / r["path"]).parent / "pixels.pt", map_location="cpu", weights_only=True) for r in (a,b)]
                    if ia["input_sha256"] != ib["input_sha256"] or ia["seed"] != ib["seed"] or a["sha256"] != b["sha256"] or not torch.equal(*tensors):
                        raise ValueError("identical question-erased inputs/seeds must give identical actual tensors")
            by_split = {}
            for split in ("train", "validation"):
                selected = [r for r in group if r["split"] == split]
                pairs = {}
                for r in selected: pairs.setdefault(r["pair"], []).append(r)
                if any(len(v) != 2 for v in pairs.values()): raise ValueError("complete passage pairs required")
                by_split[split] = dict(cases=len(selected), canonical_exact=sum(r["canonical_exact"] for r in selected),
                    accepted_exact=sum(r["accepted_exact"] for r in selected), pairs=len(pairs),
                    both_correct_pairs=sum(all(r["accepted_exact"] for r in pair) for pair in pairs.values()))
            summaries[arm][variant] = by_split
            bindings[name] = dict(inference_report_sha256=sha(folder / "report.json"), ocr_report_sha256=sha(run / (name + "_ocr/report.json")))
    # Existing initial copying is reused with explicit model/index binding.
    copy_root = ROOT / "artifacts/ink_mass_comparison_v1"
    initial_copy = json.loads((copy_root / "ink_mass_ocr/report.json").read_text())
    initial_copy_inference = json.loads((copy_root / "ink_mass_inference/report.json").read_text())
    final_copy = json.loads((run / "final_copy_ocr/report.json").read_text())
    final_copy_inference = json.loads((run / "final_copy_inference/report.json").read_text())
    if initial_copy_inference["checkpoint_sha256"] != INITIAL_SHA or final_copy_inference["checkpoint_sha256"] != final_sha or initial_copy_inference["inputs_sha256"] != final_copy_inference["inputs_sha256"]:
        raise ValueError("identical copying interface and checkpoint bindings required")
    if any(len(r["rows"]) != 64 or not r["complete_cohort"] for r in (initial_copy, final_copy)):
        raise ValueError("complete 64-case copying observations required")
    for arm, folder in (("initial", copy_root / "ink_mass_inference"), ("final", run / "final_copy_inference")):
        for out in json.loads((folder / "outputs.json").read_text()):
            path = folder / out["path"]; evidence = out["result"]
            tensor_png(path.parent / "pixels.pt", path, evidence["pixels_sha256"], evidence["png_sha256"])
    a = summaries["initial"]["normal"]["validation"]; b = summaries["final"]["normal"]["validation"]
    gates = dict(seen_fit=summaries["final"]["normal"]["train"]["accepted_exact"] >= 24,
        held_answers=b["accepted_exact"] >= max(12, a["accepted_exact"] + 4),
        held_pairs=b["both_correct_pairs"] >= max(4, a["both_correct_pairs"] + 2),
        question_dependence=b["accepted_exact"] >= summaries["final"]["question_erased"]["validation"]["accepted_exact"] + 4,
        passage_dependence=b["accepted_exact"] >= summaries["final"]["passage_erased"]["validation"]["accepted_exact"] + 4,
        copying_retention=final_copy["summary"]["validation"]["ocr_exact"] >= initial_copy["summary"]["validation"]["ocr_exact"] - 2)
    result = dict(summaries=summaries, rows=observations, prospective_exploratory_gates=gates,
        exploratory_rule_met=all(gates.values()), copying_summaries={"initial":initial_copy["summary"],"final":final_copy["summary"]},
        binding_reports=bindings, preparation_sha256=sha(run / "preparation.json"), training_report_sha256=sha(run / "training/report.json"),
        initial_checkpoint_sha256=INITIAL_SHA, final_checkpoint_sha256=final_sha,
        actual_qa_assessment_calls=256, actual_final_copying_calls=64, reused_initial_copying_calls=64,
        all_training_and_assessment_pngs_equal_actual_tensors=True, all_question_erased_pairs_bit_exact=True,
        accepted_normalization="NFC and whitespace only; case and punctuation retained; complete source-accepted answer required",
        raw_review_completed=False, reliable_writing_qualified=False, semantic_learning_validated=False,
        independent_neural_replay=False, general_ilm_goal_complete=False)
    save_json(run / "assessment.json", result)
    print(json.dumps(dict(summaries=summaries, gates=gates), indent=2), flush=True)


if __name__ == "__main__": main()
