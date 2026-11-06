#!/usr/bin/env python3
"""Independently check retained condition precision counts using NumPy."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, sha, committed_identity


def rounded_bf16(value):
    bits=np.ascontiguousarray(value,dtype=np.float32).view(np.uint32)
    even=(bits>>16)&1
    return ((bits+np.uint32(0x7fff)+even)&np.uint32(0xffff0000)).view(np.float32)


def close(actual,expected):
    if isinstance(actual,dict):
        if actual.keys()!=expected.keys():raise ValueError("count keys differ")
        for k in actual:close(actual[k],expected[k])
    elif isinstance(actual,list):
        if len(actual)!=len(expected):raise ValueError("row counts differ")
        for a,b in zip(actual,expected):close(a,b)
    elif isinstance(actual,float):
        if not np.isclose(actual,expected,rtol=1e-10,atol=1e-12):raise ValueError("numeric statistic differs")
    elif actual!=expected:raise ValueError("discrete statistic differs")


def verify(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("CPU-only verification required")
    torch.set_num_threads(2)
    report=json.loads((out/"report.json").read_text())
    for name,digest in report["identity"]["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("producer source changed")
    payload=out/"condition_projections.pt"
    if sha(payload)!=report["retained_sha256"]:raise ValueError("retained projection arrays changed")
    values=torch.load(payload,map_location="cpu",weights_only=True)
    result_path=ROOT/"references/evidence/adaptive_readout_v81_result/result.json"
    if sha(result_path)!=report["identity"]["source_result_sha256"]:raise ValueError("source result changed")
    public=json.loads(result_path.read_text());checks={}
    for arm in ("control","adapted"):
        data=values[arm];rows=data["rows"];summary=report["summaries"][arm]
        if len(rows)!=107 or rows!=summary["rows"]:raise ValueError("complete retained block cohort changed")
        arrays={k:v.numpy() for k,v in data["values"].items()}
        target_hashes=[];source_states=[]
        for i in range(32):
            folder=ROOT/f"artifacts/adaptive_readout_v81eval_20261107/{arm}/predictions/{i:02}"
            header=public["source_prediction_headers"][arm][i]
            for name in ("reference.png","full_history_trace.pt"):
                if sha(folder/name)!=header["files_sha256"][name]:raise ValueError("original source file changed")
            image=np.asarray(Image.open(folder/"reference.png"))
            n=image.shape[1]//64
            trace=torch.load(folder/"full_history_trace.pt",map_location="cpu",weights_only=True)
            source_states.append(trace["states"][0,:n].numpy())
            for j in range(n):
                row=rows[len(target_hashes)]
                if (row["case"],row["block"],row["source"])!=(i,j,header["source"]):raise ValueError("source block order differs")
                digest=hashlib.sha256(image[:,j*64:(j+1)*64].copy().tobytes()).hexdigest()
                if digest!=row["target_sha256"]:raise ValueError("owning target hash differs")
                target_hashes.append(digest)
        if not np.array_equal(np.concatenate(source_states),arrays["states"]):raise ValueError("retained condition differs from original trace")
        if not np.array_equal(rounded_bf16(arrays["states"]).view(np.uint32),arrays["rounded_input"].view(np.uint32)):
            raise ValueError("independent BF16 rounding bits differ")
        for name,array in arrays.items():
            groups={}
            for i,row in enumerate(array):groups.setdefault(row.tobytes(),[]).append(i)
            repeated=[dict(members=indices,distinct_target_images=len({target_hashes[j] for j in indices}))
                for indices in groups.values() if len(indices)>1]
            expected=dict(unique_rows=len(groups),exact_collision_groups=repeated,
                groups_with_different_targets=sum(v["distinct_target_images"]>1 for v in repeated))
            close(expected,summary["collisions"][name])
        errors={}
        for label,name,reference in (("input_rounding","rounded_input","states"),
                ("input_only_projection","input_only_projection","fp32_projection"),
                ("rounded_operands_projection","rounded_operands_projection","fp32_projection"),
                ("cpu_bf16_projection","cpu_bf16_projection","fp32_projection")):
            # Match the reported FP32 subtraction, with FP64 norm accumulation.
            difference=(arrays[name]-arrays[reference]).astype(np.float64)
            denominator=np.sqrt(np.sum(arrays[reference].astype(np.float64)**2,axis=1))
            errors[label]=(np.sqrt(np.sum(difference**2,axis=1))/np.maximum(denominator,1e-30)).tolist()
        close(errors,summary["relative_errors"])
        expected_pairs={(a,b) for a,ra in enumerate(rows) for b,rb in enumerate(rows)
            if ra["case"]%2==0 and rb["case"]==ra["case"]+1 and rb["block"]==ra["block"]}
        actual_pairs=[(p["row_a"],p["row_b"]) for p in summary["matching_pair_blocks"]]
        if set(actual_pairs)!=expected_pairs or len(actual_pairs)!=len(expected_pairs):
            raise ValueError("complete matching-pair block population differs")
        for pair in summary["matching_pair_blocks"]:
            a,b=pair["row_a"],pair["row_b"]
            if (pair["case_a"],pair["case_b"],pair["block"])!=(rows[a]["case"],rows[b]["case"],rows[a]["block"]):
                raise ValueError("paired metadata differ from owning conditions")
            distances={name:float(np.sqrt(np.sum((v[a].astype(np.float64)-v[b].astype(np.float64))**2))) for name,v in arrays.items()}
            close(distances,pair["distances"])
            if pair["different_targets"]!=(target_hashes[a]!=target_hashes[b]):raise ValueError("paired target identity differs")
            for key,numerator,denominator in (("input_rounding_distance_ratio","rounded_input","states"),
                    ("projection_bf16_distance_ratio","cpu_bf16_projection","fp32_projection")):
                close(distances[numerator]/distances[denominator] if distances[denominator] else None,pair[key])
        spectral_errors={}
        for name,expected in summary["numerical_spectra"].items():
            centered=arrays[name].astype(np.float64);centered-=centered.mean(0)
            eigen=np.maximum(np.linalg.eigvalsh(centered@centered.T),0)[::-1]
            probability=eigen[eigen>0]/eigen.sum()
            stable=float(eigen.sum()/eigen[0]);entropy=float(np.exp(-np.sum(probability*np.log(probability))))
            if not np.isclose(stable,expected["stable_rank"],rtol=1e-8,atol=1e-8) or not np.isclose(entropy,expected["entropy_effective_rank"],rtol=1e-8,atol=1e-8):
                raise ValueError("independent Gram spectral summaries differ")
            spectral_errors[name]=dict(stable_absolute_error=abs(stable-expected["stable_rank"]),
                entropy_absolute_error=abs(entropy-expected["entropy_effective_rank"]))
        checkpoint=ROOT/f"artifacts/adaptive_readout_v81r2_20261106/{arm}/checkpoint_latest.pt"
        if sha(checkpoint)!=report["identity"]["checkpoints_sha256"][arm]:raise ValueError("completed checkpoint changed")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        for name in ("weight","bias"):
            if not torch.equal(saved["model"]["flow.condition."+name],data[name]):raise ValueError("retained projection weights differ")
        del saved;gc.collect()
        weight=data["weight"].numpy();bias=data["bias"].numpy();projection_errors={}
        for name,inputs in (("fp32_projection",arrays["states"]),("input_only_projection",arrays["rounded_input"])):
            independent=inputs@weight.T+bias
            difference=independent.astype(np.float64)-arrays[name].astype(np.float64)
            relative=float(np.linalg.norm(difference)/np.linalg.norm(arrays[name].astype(np.float64)))
            maximum=float(np.max(np.abs(difference)))
            if relative>4e-6 or maximum>2e-4:raise ValueError("independent FP32 projection exceeds declared verification tolerance")
            projection_errors[name]=dict(relative_l2=relative,maximum_absolute=maximum)
        differing=int(np.count_nonzero(arrays["cpu_bf16_projection"]!=arrays["rounded_operands_projection"]))
        if differing!=summary["cpu_bf16_vs_rounded_operands_differing_values"]:raise ValueError("CPU reduction difference count differs")
        checks[arm]=dict(blocks=107,source_traces=32,input_bf16_rounding_bit_exact=True,
            all_collision_groups_and_target_identities_exact=True,all_pair_and_error_statistics_verified=True,
            gram_spectral_checks=spectral_errors,fp32_projection_checks=projection_errors,
            cpu_reduction_differences=differing,numerical_rank_not_independently_estimated=True)
    write(out/"verification.json",dict(**committed_identity(("scripts/verify_writer_condition_precision_v82.py",)),
        report_sha256=sha(out/"report.json"),retained_sha256=sha(payload),checks=checks,
        independent_numpy_checks_pass=True,gpu_reduction_neural_forward_replayed=False,
        full_flow_or_sampler_replayed=False,language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/writer_condition_precision_v82_20261107")
    args=parser.parse_args();verify(args.out.resolve())
