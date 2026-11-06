#!/usr/bin/env python3
"""Export every verified matched-loss answer; reject an unfinished experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, sha, committed_identity
from scripts.probe_corpus_fit_v75 import metrics
from scripts.probe_native_corpus_fit_v80 import cases_and_data, targets
from scripts.score_native_training_samples_v80 import check_answer
from scripts.train_native_chinese_v80 import same_identity
from scripts.verify_codec_language_v68 import assert_close

ARMS = ("control", "metric")
SOURCES = ("scripts/export_decoder_metric_v82_result.py", "scripts/probe_corpus_fit_v75.py",
    "scripts/probe_native_corpus_fit_v80.py", "scripts/score_native_training_samples_v80.py",
    "scripts/verify_codec_language_v68.py")


def validate(run):
    status = json.loads((run/"status.json").read_text())
    if status != dict(stage="completed_verified",pid=None):
        raise ValueError("both arms and CPU verification must finish before export")
    verified = json.loads((run/"verification.json").read_text())
    identity = json.loads((run/"identity.json").read_text())
    if (not same_identity(verified["identity"],identity)
            or identity["purpose"] != "matched_state_only_decoder_metric"
            or identity["arms"] != dict(control=0.,metric=.001)
            or not verified["all_64_final_outputs_independently_scored"]
            or not verified["all_64_initial_generations_bit_exact"]
            or not verified["matched_case_sampler_and_noise_rng"]
            or verified["language_quality_claim"] or verified["general_ilm_goal_complete"]):
        raise ValueError("complete matched training-only verification required")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name) != digest:raise ValueError("source-bound experiment changed")
    preflight = json.loads((run/"preflight/report.json").read_text())
    if not preflight["passed"] or not same_identity(preflight["identity"],identity):
        raise ValueError("matching actual complete-actor preflight required")
    cases,data,indices = cases_and_data()
    references = [targets(data,indices[c["source"]],c)[2] for c in cases]
    reports,headers,files = {},{},{}
    for arm in ARMS:
        folder = run/arm;report = json.loads((folder/"report.json").read_text())
        if (not same_identity(report["identity"],identity) or report["updates"]!=2000 or report["draws"]!=8000
                or report["metric_strength"]!=identity["arms"][arm]
                or sha(folder/"checkpoint_latest.pt")!=report["checkpoint_sha256"]
                or report["trainable_parameters"]!=53096961 or report["complete_actor_parameters"]!=7583225971):
            raise ValueError("declared complete final checkpoint and matched budget required")
        headers[arm] = {}
        for stage in ("before","after"):
            headers[arm][stage] = []
            for i,case in enumerate(cases):
                path = folder/stage/"predictions"/f"{i:02}"
                header = json.loads((path/"header.json").read_text())
                binding = identity["warm_checkpoint_sha256"] if stage=="before" else report["checkpoint_sha256"]
                if (header["position"]!=i or header["source"]!=case["source"]
                        or header["seed"]!=20267420+case["pair"] or header["checkpoint_binding"]!=binding):
                    raise ValueError("prediction identity differs")
                for name,digest in header["files_sha256"].items():
                    if sha(path/name)!=digest:raise ValueError("verified output payload changed")
                    if name.endswith(".png"):
                        relative = f"raw/{arm}_{stage}_{i:02}_{name}"
                        files[relative] = dict(source=path/name,sha256=digest)
                free = torch.load(path/"answer.pt",map_location="cpu",weights_only=True)
                pixels = check_answer(free)
                if len(free["images"]):
                    if not np.array_equal(pixels,np.asarray(Image.open(path/"answer.png"))):raise ValueError("free PNG differs from raw bits")
                elif (path/"answer.png").exists():
                    raise ValueError("empty output has a stale PNG")
                if not np.array_equal(references[i].numpy(),np.asarray(Image.open(path/"reference.png"))):
                    raise ValueError("owning reference changed")
                headers[arm][stage].append(header)
        rows,summary = metrics(folder/"after",cases,references,independent=True)
        assert_close(rows,report["rows"]);assert_close(summary,report["summary"])
        assert_close(summary,verified["summaries"][arm]);reports[arm] = report
    control,metric = (verified["summaries"][a] for a in ARMS)
    gates = dict(free_exact_at_least_24=metric["free_exact"]>=24,free_f1_at_least_098=metric["free_f1"]>=.98,
        free_exact_exceeds_control=metric["free_exact"]>control["free_exact"],free_f1_exceeds_control=metric["free_f1"]>control["free_f1"])
    if gates!=verified["gates"] or all(gates.values())!=verified["training_fit_repair_pass"]:
        raise ValueError("original training-fit decision differs")
    return verified,identity,reports,headers,files


def plates(raw,destination,headers):
    result = []
    labels = [("REFERENCE", "control", "after", "reference"),
        ("WARM MODEL", "control", "before", "answer"),
        ("CONTINUED CONTROL: FREE", "control", "after", "answer"),
        ("DECODER METRIC: FREE", "metric", "after", "answer"),
        ("CONTROL: CORRECT HISTORY", "control", "after", "assisted"),
        ("METRIC: CORRECT HISTORY", "metric", "after", "assisted"),
        ("SUPPLIED-CODE RECONSTRUCTION", "metric", "after", "supplied")]
    for start in range(0,32,8):
        panels = []
        for i in range(start,start+8):
            for label,arm,stage,kind in labels:
                path = raw/f"{arm}_{stage}_{i:02}_{kind}.png"
                picture = Image.open(path).convert("RGB") if path.exists() else None
                width = picture.width if picture is not None else 0
                lines = max(1,math.ceil(width/1024))
                panel = Image.new("RGB",(1056,32+40*lines),"white")
                extra = " (empty output)" if width==0 else ""
                ImageDraw.Draw(panel).text((12,8),
                    f"{i:02} {label} | {headers[arm][stage][i]['source']} | {width//64} blocks{extra}",fill="black")
                if picture is not None:
                    segments = []
                    for line in range(lines):
                        segment = picture.crop((line*1024,0,min(width,(line+1)*1024),32))
                        panel.paste(segment,(12,32+line*40));segments.append(np.asarray(segment))
                    if not np.array_equal(np.concatenate(segments,axis=1),np.asarray(picture)):
                        raise ValueError("display wrapping removed source pixels")
                panels.append(panel)
        canvas = Image.new("RGB",(1056,sum(p.height for p in panels)),"#e8e8e8")
        offset = 0
        for panel in panels:canvas.paste(panel,(0,offset));offset+=panel.height
        path = destination/f"all_cases_{start:02}_{start+7:02}.png";canvas.save(path)
        result.append(dict(path=path.name,sha256=sha(path),cases=list(range(start,start+8)),
            scale=1,display_wrapping_preserves_all_pixels=True,image_pixels_repaired=False))
    return result


def export(run,destination,*,check_only=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("CPU-only export required")
    torch.set_num_threads(2)
    verified,identity,reports,headers,files = validate(run)
    producing = committed_identity(SOURCES)
    if check_only:
        print(json.dumps(dict(ready=True,summaries=verified["summaries"],gates=verified["gates"],png_files=len(files)),indent=2))
        return
    temporary = destination.with_name(destination.name+".partial")
    if destination.exists() or temporary.exists():raise FileExistsError("preserve completed or partial evidence export")
    raw = temporary/"raw";raw.mkdir(parents=True)
    for relative,value in files.items():
        target = temporary/relative;shutil.copyfile(value["source"],target)
        if sha(target)!=value["sha256"]:raise ValueError("exported PNG is not byte-identical")
    views = plates(raw,temporary,headers)
    receipt = dict(**producing,source_run=str(run.relative_to(ROOT)),identity=identity,
        verification_sha256=sha(run/"verification.json"),verification=verified,arm_reports=reports,
        arm_report_sha256={a:sha(run/a/"report.json") for a in ARMS},
        source_prediction_headers=headers,all_case_plates=views,
        raw_png_sha256={k:v["sha256"] for k,v in files.items()},raw_png_bytes_unchanged=True,
        training_only=True,language_quality_claim=False,general_ilm_goal_complete=False)
    write(temporary/"result.json",receipt)
    summaries=verified["summaries"];decision="meets" if verified["training_fit_repair_pass"] else "fails"
    lines=["# Matched decoder-sensitive writing result", "",
        f"The metric arm {decision} the declared training-fit comparison. This is a seen-question experiment, not a qualification of general language.", "",
        "| Final writing path | Control exact / 32 | Control F1 (%) | Metric exact / 32 | Metric F1 (%) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for label,exact,f1 in (("Autonomous","free_exact","free_f1"),
            ("Correct history and known length","assisted_exact_pixels","assisted_f1"),
            ("Supplied-code reconstruction","supplied_exact_pixels","supplied_f1")):
        c,m=summaries["control"],summaries["metric"]
        lines.append(f"| {label} | {c[exact]} | {100*c[f1]:.6f} | {m[exact]} | {100*m[f1]:.6f} |")
    lines += ["", "Each arm restores the same V81 adapted checkpoint and receives 2,000 updates, batch four. Only the decoder-sensitive loss strength differs. The original quality criteria are retained:", ""]
    lines += [f"- `{k}`: {'pass' if v else 'fail'}." for k,v in verified["gates"].items()]
    lines += ["", "The complete actor retains 7,583,225,971 parameters, including externally pretrained weights; 53,096,961 are trained. Image-agreement scores are not semantic accuracy. Independent CPU scoring and retained-coordinate decoding do not replay the entire contextual body and sampler.", "",
        "All before/after PNGs are unchanged. The four plates show every case in its original order, including wrong and empty answers:", ""]
    lines += [f"- [Cases {start:02}–{start+7:02}](all_cases_{start:02}_{start+7:02}.png)." for start in range(0,32,8)]
    lines += ["", "[Full provenance, per-case results and CPU verification](result.json). Passing training fit would still require new-prompt/copy and development tests. The general ILM goal remains unachieved.", ""]
    (temporary/"README.md").write_text("\n".join(lines))
    temporary.rename(destination)
    print(json.dumps(dict(exported=str(destination.relative_to(ROOT)),png_files=len(files),
        training_fit_repair_pass=verified["training_fit_repair_pass"],gates=verified["gates"]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/decoder_metric_v82r2_20261107")
    parser.add_argument("--destination",type=Path,default=ROOT/"references/evidence/decoder_metric_v82_result")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args();export(args.run.resolve(),args.destination.resolve(),check_only=args.check_only)
