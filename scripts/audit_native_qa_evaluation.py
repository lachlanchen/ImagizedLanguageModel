#!/usr/bin/env python3
"""Audit actual QA/reference/control bindings without invoking a neural model."""
import argparse
import json
from pathlib import Path
import unicodedata

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.qa_learning import CONTROLS, PAGES, SOURCE, dataset


def normalized(text): return "".join(unicodedata.normalize("NFC", text).split())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); run = args.run.resolve()
    train, held = dataset(); selected = train + held
    prep = json.loads((run / "preparation.json").read_text())
    source = {r["id"]:r for r in map(json.loads, SOURCE.read_text().splitlines())}
    calibration_path = CONTROLS / "reference_ocr/report.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["cohort"] != [r["id"] for r in selected] or calibration["manifest_sha256"] != sha(PAGES / "pairs.jsonl"):
        raise ValueError("complete unchanged canonical reference calibration required")
    protocol = ROOT / "references/native_paired_qa_learning_protocol.md"
    if prep["code_sha256"][str(protocol.relative_to(ROOT))] != sha(protocol) or "At least 24/32 correct seen" not in protocol.read_text():
        raise ValueError("unchanged registered seen threshold required")
    groups, observations, summaries = {}, [], {}
    for row, observed in zip(selected, calibration["rows"]):
        if row["id"] != observed["id"] or row["target_sha256"] != observed["png_sha256"]:
            raise ValueError("reference observation not bound to actual target pixels")
        accepted = {normalized(r["text"]) for r in source[row["id"]]["references"]}
        groups.setdefault((row["split"],row["pair"]),[]).append((row,accepted))
        observations.append(dict(id=row["id"],split=row["split"],pair=row["pair"],
            canonical_reference_ocr_exact=observed["ocr_exact"],
            source_accepted_reference_ocr_exact=normalized(observed["raw_transcript"]) in accepted))
    for split in ("train","validation"):
        cases = [r for r in observations if r["split"]==split]; pairs={}
        for r in cases: pairs.setdefault(r["pair"],[]).append(r)
        summaries[split] = dict(cases=len(cases),canonical_reference_exact=sum(r["canonical_reference_ocr_exact"] for r in cases),
            accepted_reference_exact=sum(r["source_accepted_reference_ocr_exact"] for r in cases),pairs=len(pairs),
            accepted_reference_both_correct_pairs=sum(all(r["source_accepted_reference_ocr_exact"] for r in rs) for rs in pairs.values()))
    indices={}
    for variant in ("normal","question_erased","passage_erased"):
        path=run/(variant+"_inputs.json")
        if sha(path)!=prep["index_files_sha256"][variant]:raise ValueError("registered input index changed")
        rows=json.loads(path.read_text());expected=selected if variant=="normal" else held
        if [r["id"] for r in rows]!=[r["id"] for r in expected] or any(sha(Path(r["input"]))!=r["input_sha256"] for r in rows):
            raise ValueError("complete actual input images required")
        indices[variant]={r["id"]:r for r in rows}
    for (split,_),pair in groups.items():
        if len(pair)!=2 or pair[0][1]&pair[1][1]:raise ValueError("two disjoint normalized answer sets per passage required")
        if split=="validation":
            a,b=[indices["question_erased"][r["id"]] for r,_ in pair]
            if a["input_sha256"]!=b["input_sha256"] or a["seed"]!=b["seed"]:
                raise ValueError("question-erased pair retains a varying input or noise cue")
    result=dict(summaries=summaries,rows=observations,implementation_sha256=sha(__file__),
        source_qa_sha256=sha(SOURCE),manifest_sha256=sha(PAGES/"pairs.jsonl"),reference_ocr_sha256=sha(calibration_path),
        registered_preparation_sha256=sha(run/"preparation.json"),registered_protocol_sha256=sha(protocol),
        all_held_question_erased_pairs_identical_input_and_seed=True,all_normalized_paired_accepted_answers_disjoint=True,
        theoretical_question_erased_correct_upper_bound=16,theoretical_question_erased_both_correct_pairs_upper_bound=0,
        bound_requires_deterministic_same_input_same_seed=True,actual_all_held_neural_control_replay_pending=True,
        registered_seen_ocr_fit_threshold=24,seen_threshold_exceeds_clean_reference_ocr_count=summaries["train"]["accepted_reference_exact"]<24,
        clean_reference_count_is_not_upper_bound_on_other_generated_typography=True,
        prospective_criteria_changed=False,new_neural_calls=0,quality_claim=False,general_ilm_goal_complete=False)
    if args.output.exists():raise ValueError("preserve earlier audit")
    args.output.parent.mkdir(parents=True,exist_ok=True);save_json(args.output,result)
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__": main()
