#!/usr/bin/env python3
"""Inspect retained writing conditions and the actual trained projection on CPU."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, sha, committed_identity

PLAN = "references/writer_condition_precision_v82_protocol.md"
RESULT = ROOT/"references/evidence/adaptive_readout_v81_result/result.json"
SOURCES = ("scripts/audit_writer_condition_precision_v82.py", PLAN,
    "ilm/visual_lm/clean_writing_flow.py", "ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py", "scripts/cache_semantic_images_v72.py")


def digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().float().numpy().tobytes()).hexdigest()


def collision_groups(values, rows):
    groups = {}
    for i,row in enumerate(values):groups.setdefault(digest(row),[]).append(i)
    repeated = [dict(members=indices, distinct_target_images=len({rows[i]["target_sha256"] for i in indices}))
        for indices in groups.values() if len(indices)>1]
    return dict(unique_rows=len(groups),exact_collision_groups=repeated,
        groups_with_different_targets=sum(g["distinct_target_images"]>1 for g in repeated))


def spectrum(values):
    array=values.double().numpy();array=array-array.mean(0,keepdims=True)
    singular=np.linalg.svd(array,compute_uv=False)
    energy=singular**2;total=energy.sum()
    probability=energy[energy>0]/total
    return dict(singular_values=singular.tolist(),stable_rank=float(total/energy[0]),
        entropy_effective_rank=float(np.exp(-(probability*np.log(probability)).sum())),
        numerical_rank=int((singular>singular[0]*max(array.shape)*np.finfo(np.float64).eps).sum()))


def per_row_error(actual,reference):
    difference=(actual-reference).double()
    denominator=reference.double().norm(dim=-1).clamp_min(1e-30)
    return (difference.norm(dim=-1)/denominator).tolist()


def comparisons(values,rows):
    index={(r["case"],r["block"]):i for i,r in enumerate(rows)};pairs=[]
    for case in range(0,32,2):
        blocks=sorted(j for c,j in index if c==case and (case+1,j) in index)
        for block in blocks:
            a,b=index[(case,block)],index[(case+1,block)]
            distances={name:float((value[a].double()-value[b].double()).norm()) for name,value in values.items()}
            pairs.append(dict(case_a=case,case_b=case+1,block=block,row_a=a,row_b=b,
                different_targets=rows[a]["target_sha256"]!=rows[b]["target_sha256"],distances=distances,
                input_rounding_distance_ratio=(distances["rounded_input"]/distances["states"] if distances["states"] else None),
                projection_bf16_distance_ratio=(distances["cpu_bf16_projection"]/distances["fp32_projection"] if distances["fp32_projection"] else None)))
    return pairs


def run(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only audit required")
    if out.exists():raise FileExistsError("preserve completed or partial precision audit")
    torch.set_num_threads(2)
    result=json.loads(RESULT.read_text());run=ROOT/"artifacts/adaptive_readout_v81eval_20261107"
    trained=ROOT/"artifacts/adaptive_readout_v81r2_20261106"
    if (sha(run/"report.json")!=result["report_sha256"]
            or sha(run/"verification.json")!=result["verification_sha256"]
            or not result["verification"]["all_64_outputs_independently_scored"]):
        raise ValueError("completed authenticated V81 evidence required")
    cases_path=ROOT/"artifacts/native_chinese_v80_20261106/corpus_fit_v80r2/cases.json"
    if sha(cases_path)!=result["verification"]["identity"]["cohort_sha256"]:
        raise ValueError("original cohort changed")
    cases=json.loads(cases_path.read_text())
    identity=dict(**committed_identity(SOURCES),source_result_sha256=sha(RESULT),
        source_evaluation_report_sha256=result["report_sha256"],source_verification_sha256=result["verification_sha256"],
        cases_sha256=sha(cases_path),checkpoints_sha256=result["verification"]["identity"]["checkpoints_sha256"],
        torch_version=str(torch.__version__),device="cpu",target_cohort="V81 fixed32 training questions",
        v82_outputs_used=False,model_updates=0)
    out.mkdir();write(out/"identity.json",identity)
    summaries={};retained={}
    for arm in ("control","adapted"):
        checkpoint=trained/arm/"checkpoint_latest.pt"
        if sha(checkpoint)!=identity["checkpoints_sha256"][arm]:raise ValueError("fixed trained checkpoint changed")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if saved["update"]!=2000 or not saved["training_complete"] or saved["arm"]!=arm:
            raise ValueError("completed final V81 state required")
        weight=saved["model"]["flow.condition.weight"].clone();bias=saved["model"]["flow.condition.bias"].clone()
        del saved;gc.collect()
        states=[];rows=[]
        for i,case in enumerate(cases):
            folder=run/arm/"predictions"/f"{i:02}"
            header=json.loads((folder/"header.json").read_text())
            if header!=result["source_prediction_headers"][arm][i] or header["source"]!=case["source"]:
                raise ValueError("source-bound header changed")
            for name in ("full_history_trace.pt","reference.png"):
                if sha(folder/name)!=header["files_sha256"][name]:raise ValueError("retained states or target image changed")
            trace=torch.load(folder/"full_history_trace.pt",map_location="cpu",weights_only=True)
            h=trace["states"]
            n=case["blocks"];target=np.asarray(Image.open(folder/"reference.png"))
            if (h.shape!=(1,n+1,4096) or h.dtype!=torch.float32 or target.shape!=(32,64*n)
                    or trace["reference_codes_sha256"]!=case["codes_sha256"]):
                raise ValueError("owning before-block states differ")
            states.append(h[0,:n])
            for j in range(n):
                rows.append(dict(case=i,source=case["source"],block=j,first=j==0,
                    target_sha256=hashlib.sha256(target[:,j*64:(j+1)*64].copy().tobytes()).hexdigest(),
                    source_trace_sha256=sha(folder/"full_history_trace.pt")))
        states=torch.cat(states)
        if len(states)!=107 or weight.shape!=(1536,4096) or bias.shape!=(1536,):
            raise ValueError("actual complete cohort and learned projection required")
        with torch.no_grad():
            rounded=states.to(torch.bfloat16).float()
            fp32=F.linear(states,weight,bias)
            input_only=F.linear(rounded,weight,bias)
            rounded_operands=F.linear(rounded,weight.bfloat16().float(),bias.bfloat16().float()).bfloat16().float()
            local=F.linear(states.bfloat16(),weight.bfloat16(),bias.bfloat16()).float()
        values=dict(states=states,rounded_input=rounded,fp32_projection=fp32,input_only_projection=input_only,
            rounded_operands_projection=rounded_operands,cpu_bf16_projection=local)
        if not all(bool(torch.isfinite(v).all()) for v in values.values()):raise ValueError("nonfinite condition projection")
        collisions={name:collision_groups(value,rows) for name,value in values.items()}
        pairs=comparisons(values,rows)
        summaries[arm]=dict(blocks=len(states),rows=rows,collisions=collisions,matching_pair_blocks=pairs,
            relative_errors=dict(input_rounding=per_row_error(rounded,states),
                input_only_projection=per_row_error(input_only,fp32),
                rounded_operands_projection=per_row_error(rounded_operands,fp32),
                cpu_bf16_projection=per_row_error(local,fp32)),
            numerical_spectra={name:spectrum(values[name]) for name in ("states","rounded_input","fp32_projection","cpu_bf16_projection")},
            cpu_bf16_vs_rounded_operands_differing_values=int((local!=rounded_operands).sum()),
            condition_weight_sha256=digest(weight),condition_bias_sha256=digest(bias))
        retained[arm]=dict(values=values,rows=rows,weight=weight,bias=bias)
        if sha(checkpoint)!=identity["checkpoints_sha256"][arm]:raise ValueError("immutable old checkpoint changed during audit")
    torch.save(retained,out/"condition_projections.pt")
    write(out/"report.json",dict(identity=identity,summaries=summaries,
        retained_sha256=sha(out/"condition_projections.pt"),cuda_initialized=torch.cuda.is_initialized(),
        gpu_neural_forward_replayed=False,optimizer_or_sampler_called=False,new_answers_generated=False,
        semantic_accuracy_claim=False,language_quality_claim=False,general_ilm_goal_complete=False))
    for arm,s in summaries.items():
        print(json.dumps(dict(arm=arm,blocks=s["blocks"],collisions=s["collisions"],
            cpu_reduction_differences=s["cpu_bf16_vs_rounded_operands_differing_values"]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/writer_condition_precision_v82_20261107")
    args=parser.parse_args();run(args.out.resolve())
