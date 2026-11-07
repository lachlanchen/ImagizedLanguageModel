#!/usr/bin/env python3
"""Publish unchanged canvas PNGs and source-bound, independently checked counts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from scripts.run_parallel_canvas import prepared, values, read, scores, DATA
from scripts.cache_semantic_images_v72 import ROOT, CODEC, sha, committed_identity, atomic_json


def main(run, native, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized(): raise ValueError("CPU-only export required")
    torch.set_num_threads(4)
    producer=committed_identity(("scripts/export_parallel_canvas.py",))
    identity,prep=prepared(run);verified=read(run/"verification.json");training=read(run/"training.json")
    if verified["identity"]!=identity or verified["optimizer_updates_verified"]!=2000: raise ValueError("complete independent verification required")
    for name,digest in verified["files_sha256"].items():
        if sha(run/name)!=digest: raise ValueError("verified raw evidence changed")
    if sha(run/"model_final.pt")!=training["bundle_sha256"]: raise ValueError("model bundle changed")
    nr=read(native/"report.json")
    if nr["calls"]!=44 or nr["forbidden_training_paths_opened"]: raise ValueError("complete standalone image-only calls required")
    output.mkdir(parents=True,exist_ok=False);files={};references={};rows=[];native_checks=[]

    def copy(source,relative):
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(source)!=sha(target): raise ValueError("PNG bytes changed on export")
        files[relative]=sha(target);return relative

    def reference(a):
        a=np.asarray(a,dtype=np.uint8)
        key=hashlib.sha256(a.tobytes()+str(a.shape).encode()).hexdigest()
        name="references/"+key+".png"
        if name not in files:
            p=output/name;p.parent.mkdir(exist_ok=True);Image.fromarray(a).save(p)
            if not np.array_equal(np.asarray(Image.open(p)),a): raise ValueError("exported reference pixels changed")
            files[name]=sha(p)
        return name

    data=CodecLanguageAuditData(DATA)
    for row in verified["results"]:
        if row["group"]=="development": refs=data.references(row["index"])
        else: refs=values(run,next(r for r in prep["rows"] if r["group"]==row["group"] and r["index"]==row["index"]))["references"]
        rel="raw/"+row["output"].replace("/answer.png",".png")
        name=copy(run/row["output"],rel)
        actual=scores(np.asarray(Image.open(output/name)),[r.numpy() for r in refs])
        if actual["exact"]!=row["exact"] or actual["foreground_f1"]!=row["foreground_f1"]: raise ValueError("PNG scoring differs from verification")
        rows.append(dict(**{k:v for k,v in row.items() if k!="output"},output=name,reference_images=[reference(r.numpy()) for r in refs]))
    for i,row in enumerate(prep["rows"]):
        p=native/f"{i:02}";h=read(p/"header.json")
        if sha(p/"answer.png")!=h["answer_sha256"] or sha(p/"coordinates.pt")!=h["coordinates_sha256"]: raise ValueError("native output changed")
        cached=run/"final"/f"{row['group']}_{row['index']:02}"/"answer.png"
        same=sha(cached)==sha(p/"answer.png");memory=h["memory_sha256"]==row["memory_sha256"]
        if not same or not memory: raise ValueError("standalone canvas/memory does not reproduce the cached route")
        if h["input_sha256"]!=row["input_sha256"]: raise ValueError("standalone input changed")
        out=copy(p/"answer.png",f"native/{i:02}.png");inp=copy(run/row["input"],f"inputs/{row['group']}_{row['index']:02}.png")
        native_checks.append(dict(group=row["group"],index=row["index"],output=out,input=inp,pixels_exact=True,memory_exact=True))

    # Post-result supplied-code control: observed reconstruction, not an upper
    # bound on the codec's full representational capacity or autonomous output.
    codec=load_qualified_writing_codec(CODEC,"cpu");white=torch.load(run/"white.pt",map_location="cpu",weights_only=True)
    controls=[]
    for row in [r for r in prep["rows"] if r["group"]=="training"]:
        value=values(run,row);z=white.expand(64,768).clone();z[:len(value["codes"])]=value["codes"]
        with torch.no_grad(): tile=(codec.decode(z)>=0).to(torch.uint8).mul(255).reshape(64,32,32)
        a=tile.permute(1,0,2).reshape(32,2048).numpy();rel=f"supplied_codes/{row['index']:02}.png"
        p=output/rel;p.parent.mkdir(exist_ok=True);Image.fromarray(a).save(p);files[rel]=sha(p)
        pred=np.asarray(Image.open(run/"final"/f"training_{row['index']:02}"/"answer.png"))
        ref=value["references"][0].numpy();target=np.pad(ref,((0,0),(0,2048-ref.shape[1])),constant_values=255)
        controls.append(dict(index=row["index"],output=rel,**scores(a,[ref]),
            prediction_wrong_pixels=int((pred!=target).sum()),supplied_wrong_pixels=int((a!=target).sum()),
            prediction_errors_shared_with_supplied=int(((pred!=target)&(a!=target)).sum())))
    for start in (0,8,16,24): copy(run/f"training_overview_{start:02}.png",f"training_overview_{start:02}.png")
    res=dict(producer=producer,training_identity=identity,groups=verified["groups"],fit_gates=verified["fit_gates"],
        fit_pass=verified["fit_pass"],training_both_correct_pairs=verified["training_both_correct_pairs"],
        reader_screen=verified["reader_screen"],reader_screen_pass=verified["reader_screen_pass"],
        development_both_correct_pairs=verified["development_both_correct_pairs"],question_erasure_f1_gain=verified["question_erasure_f1_gain"],
        training=training,evaluation=verified["evaluation"],standalone=dict(calls=44,native_checks=native_checks,
            setup_seconds=nr["setup_seconds"],total_seconds=nr["total_seconds"],peak_allocated_gib=nr["peak_allocated_gib"],
            provenance=nr["provenance"],python_open_audit=True,forbidden_training_paths_opened=False,
            python_audit_is_not_a_full_OS_sandbox=True),
        supplied_code_control=dict(post_result=True,cases=controls,exact=sum(r["exact"] for r in controls),
            foreground_f1=float(np.mean([r["foreground_f1"] for r in controls])),not_autonomous_or_capacity_upper_bound=True),
        raw_cases=rows,files_sha256=files,retained_coordinate_files_sha256={k:v for k,v in verified["files_sha256"].items() if k.endswith("coordinates.pt")},
        verification_sha256=sha(run/"verification.json"),native_report_sha256=sha(native/"report.json"),
        all332_raw_canvases_byte_identical=True,all44_native_canvases_byte_identical=True,
        coordinate_images_verified=verified["coordinate_images_replayed"],complete_runtime_parameters=nr["provenance"]["complete_parameters"],
        optimized_parameters=training["optimized_parameters"],optimizer_updates=2000,training_draws=8000,
        additional_training_optimizer_invocations=verified["additional_training_optimizer_invocations"],
        source_memories_reused=44,external_pretraining_retained=True,model_weights_published=False,
        fixed_canvas_without_learned_stream_termination=True,reused_public_development=True,
        useful_general_language_qualified=False,general_ilm_goal_complete=False,
        attribution="CMRC2018 Cui et al. (2019), CC-BY-SA-4.0; Qwen3-VL Alibaba, Apache-2.0; prepared copy requests.")
    atomic_json(output/"result.json",res)
    lines=["# Whole-canvas image writing: all retained outputs\n",
        "The writer fits seen answers but fails every new-copy and unseen-answer exact check.",
        "All 332 main canvases and 44 fresh-process canvases below are unchanged PNGs.",
        "Every accepted reference is retained; canvas scores pad exterior white and preserve the left origin.",
        "The full 32-by-2048 generated canvas remains visible when opening an image. No cleaning or OCR replacement is applied.\n",
        "CMRC2018: Cui et al. (2019), CC-BY-SA-4.0. Qwen3-VL: Alibaba, Apache-2.0.\n",
        "| Group | Stage | Index | Exact | Reference images | Complete output |",
        "|---|---|---:|---|---|---|"]
    for row in rows:
        refs=" ".join(f"[reference {i+1}]({p})" for i,p in enumerate(row["reference_images"]))
        lines.append(f"| {row['group']} | {row['stage']} | {row['index']} | {row['exact']} | {refs} | [PNG]({row['output']}) |")
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(groups=res["groups"],fit_pairs=res["training_both_correct_pairs"],fit_pass=res["fit_pass"],
        supplied_code_exact=res["supplied_code_control"]["exact"],native_replays=44,exported_png_files=len(files)),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=ROOT/"artifacts/parallel_canvas_v1")
    p.add_argument("--native",type=Path,default=ROOT/"artifacts/parallel_canvas_native_v1")
    p.add_argument("--output",type=Path,default=ROOT/"references/evidence/parallel_canvas_result")
    a=p.parse_args();main(a.run,a.native,a.output)
