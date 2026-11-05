#!/usr/bin/env python3
"""Export the completed, independently scored full-corpus result without repair."""

import argparse
import json
import os
from pathlib import Path
import shutil

from PIL import Image
import torch

from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json, committed_identity
from scripts.verify_codec_language_v68 import independent_actor_prompt

SOURCES=("scripts/export_clean_corpus_v75.py","scripts/verify_semantic_ilm_v72.py",
    "ilm/visual_lm/codec_language_audit.py")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/clean_corpus_v75_20261106")
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("export must be CPU-only")
    torch.set_num_threads(2)
    identity=committed_identity(SOURCES)
    training=json.loads((args.run/"training/training_report.json").read_text())
    report=json.loads((args.run/"development/report.json").read_text())
    verification=json.loads((args.run/"verification.json").read_text())
    if (training["updates"]!=40000 or training["draws"]!=640000
            or training["identity"]["purpose"]!="full_corpus_clean"
            or training["checkpoint_sha256"]!=sha(args.run/"training/checkpoint_latest.pt")
            or report["identity"]["training_report_sha256"]!=sha(args.run/"training/training_report.json")
            or report["identity"]["checkpoint_sha256"]!=training["checkpoint_sha256"]
            or report["generations"]!=384 or not verification["offline_checks_pass"]
            or not verification["all_384_outputs_independently_scored"]
            or verification["report_sha256"]!=sha(args.run/"development/report.json")):
        raise ValueError("completed authenticated full training and original audit required")
    for name,digest in training["identity"]["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("frozen training source changed")
    for name,digest in report["files_sha256"].items():
        if sha(args.run/"development"/name)!=digest:
            raise ValueError("raw generation changed")
    for name in ("tokenizer","ocr","language_vocabulary","teacher_service","persistent_glyph_inventory"):
        if report["provenance"][name] is not False:
            raise ValueError("runtime boundary differs from the declared actor")
    data=CodecLanguageAuditData(ROOT/"data/general_instructions/codec_language_audit_v68_20261105")
    output=ROOT/"references/evidence/clean_corpus_v75_raw"
    output.mkdir(exist_ok=True)
    evidence={}
    for i in range(2):
        payload=data.payload(i)
        target=output/f"{i:03}_reference.png"
        Image.fromarray(payload["references"][0].numpy()).save(target)
        evidence[str(target.relative_to(ROOT))]=sha(target)
        for route in ("normal","blank"):
            prompt=independent_actor_prompt(payload,blank=route=="blank")
            target=output/f"{i:03}_{route}_prompt.png"
            Image.fromarray(((prompt[0,0]+1)*127.5).round().to(torch.uint8).numpy()).save(target)
            evidence[str(target.relative_to(ROOT))]=sha(target)
            folder=args.run/"development/predictions"/f"{i:03}_{route}"
            header=json.loads((folder/"header.json").read_text())
            source=folder/"answer.png"
            if header["output"]["answer_png_sha256"] is not None:
                if sha(source)!=header["output"]["answer_png_sha256"]:
                    raise ValueError("raw answer receipt changed")
                target=output/f"{i:03}_{route}_answer.png"
                shutil.copy2(source,target)
                evidence[str(target.relative_to(ROOT))]=sha(target)
            elif source.exists():
                raise ValueError("empty answer has a stale placeholder")
    compact=dict(export_identity=identity,source_run=str(args.run.resolve().relative_to(ROOT)),
        training={k:training[k] for k in ("updates","draws","completed_epochs",
            "checkpoint_sha256","trainable_parameters","active_elapsed_seconds","peak_allocated_vram_gib")},
        training_identity=training["identity"],
        training_report_sha256=sha(args.run/"training/training_report.json"),
        audit={k:report[k] for k in ("generations","elapsed_seconds","setup_seconds","peak_allocated_vram_gib")},
        report_sha256=sha(args.run/"development/report.json"),verification=verification,
        verification_sha256=sha(args.run/"verification.json"),public_raw_files_sha256=evidence,
        external_pretraining=True,development_benchmark_reused=True,model_output_pixels_repaired=False,
        language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(ROOT/"references/evidence/clean_corpus_v75_result.json",compact)
    chinese=verification["measurements"]["cmrc"]
    normal,blank=chinese["normal"],chinese["blank"]
    effect=verification["question_effect"]
    passed=verification["bounded_chinese_development_pass"]
    english=verification["measurements"]["dolly"]["normal"]
    outcome="passes" if passed else "fails"
    decision="pass" if passed else "fail"
    failures=[key for key,value in verification["gates"].items() if not value]
    text=f'''# Full-corpus clean writing {outcome} the original Chinese requirements

The completed clean writer produces **{round(normal['exact']*128)}/128 exact
Chinese answer images**, with **{normal['foreground_f1']*100:.3f}% foreground
F1**, versus {blank['foreground_f1']*100:.3f}% with question ink erased.
Both questions are correct in {round(verification['paired_both_correct']*64)}/64
article pairs. The fixed Chinese development decision is **{decision}**.

The mean question effect is {effect['mean']*100:.3f} F1 points, with 95%
article-bootstrap interval [{effect['ci95'][0]*100:.3f}, {effect['ci95'][1]*100:.3f}].
Learned stopping is {normal['stopped']*100:.3f}%. Failed requirements:
{', '.join(failures) if failures else 'none'}.

The writer has {training['trainable_parameters']:,} trainable parameters and
starts afresh on all 12,976 admitted examples. It completes 40,000 updates,
640,000 draws and {training['completed_epochs']} full corpus passes in
{training['active_elapsed_seconds']:.3f} active seconds, at
{training['peak_allocated_vram_gib']:.3f} GiB allocated memory without the
inherited reader. The 384 autonomous normal/control generations take
{report['elapsed_seconds']:.3f} seconds, including {report['setup_seconds']:.3f}
seconds setup, at {report['peak_allocated_vram_gib']:.3f} GiB including the reader.
These costs exclude earlier writing-codec learning, the reused image-feature
cache's original computation and external foundation pretraining.

English literal reference F1 is {english['foreground_f1']*100:.3f}%, with
{round(english['exact']*64)}/64 exact reference images. Those references do
not exhaust valid English answers. The benchmark is reused development data;
this is not a fresh confirmatory or general-language qualification.

Independent CPU verification authenticates all 384 retained outputs and the
full 12,976-example cache, reconstructs source inputs, and independently
recomputes physical scores and all original Chinese decisions. It does not
replay the entire neural sampler or every feedback encoding. Sealed examples
remain unopened. The separately prepared copy probe has its own result.

The first two fixed original references, normal/control input images and
unmodified answer PNGs are exported under evidence/clean_corpus_v75_raw/.
Empty answers have no placeholder PNG. No output marks are repaired, removed
or selected for appearance. All 384 raw streams and the checkpoint remain in
{args.run.resolve().relative_to(ROOT)}.

The [evidence receipt](evidence/clean_corpus_v75_result.json) and
[fixed protocol](clean_corpus_v75_protocol.md) bind the numerical claims and
their scope. The small V74 training-set fit remains a distinct control.
The general usable ILM objective remains unachieved.
'''
    (ROOT/"references/clean_corpus_v75_result.md").write_text(text)
    print(json.dumps(dict(chinese_exact=round(normal["exact"]*128),
        foreground_f1=normal["foreground_f1"],bounded_chinese_development_pass=passed,
        failed_requirements=failures),indent=2))


if __name__=="__main__":
    main()
