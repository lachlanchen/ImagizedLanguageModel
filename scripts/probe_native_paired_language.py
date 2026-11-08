#!/usr/bin/env python3
"""Actual full-page codec and two-update gradient probes, never qualification."""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, save_json, sha

INITIAL = "fc031796540a4b6ca4c86774a5033e94ae32b415e825b6a221128756748d3575"


def implementation():
    return {p: sha(ROOT/p) for p in ("scripts/probe_native_paired_language.py",
        "references/paired_language_native_preflight_protocol.md", "ilm/image_edit/baseline.py",
        "ilm/image_edit/paired.py", "ilm/image_edit/autonomous.py", "ilm/image_edit/recognition.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("codec","backward"),required=True)
    parser.add_argument("--snapshot",type=Path,default=ROOT/"artifacts/external/image_edit_learning/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d")
    parser.add_argument("--condition",type=Path,default=ROOT/"artifacts/image_edit_baseline/condition/condition.pt")
    parser.add_argument("--teacher",type=Path,default=ROOT/"artifacts/external/writing_reader/hub/models--stepfun-ai--GOT-OCR-2.0-hf/snapshots/d3017ef2c2c1395888c8d635c5e0508bcb0ac78d")
    parser.add_argument("--initial",type=Path,default=ROOT/"artifacts/balanced_autonomous_writer_v1/training/checkpoint.pt")
    parser.add_argument("--pages",type=Path,default=ROOT/"artifacts/image_edit_learning/pages_v2")
    parser.add_argument("--controls",type=Path,default=ROOT/"artifacts/paired_language_controls_v1")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import subprocess
    import numpy as np
    from PIL import Image
    import torch
    from ilm.image_edit.paired import (adapter_state,add_adapters,answer,load_pipeline,read_manifest,
        restore_adapters,save_answer,verify_source)
    from ilm.image_edit.autonomous import encode_reference,final_image_loss,native_implementation,sample_page
    from ilm.image_edit.recognition import RecognitionTeacher
    code=implementation()
    if subprocess.check_output(["git","status","--porcelain","--",*code],cwd=ROOT,text=True):
        raise ValueError("commit the exact probe and protocol before GPU execution")
    source=verify_source(args.snapshot)
    preparation=json.loads((args.controls/"preparation.json").read_text())
    index=args.controls/"preflight_normal_inputs.json"
    if (preparation["manifest_sha256"]!=sha(args.pages/"pairs.jsonl")
            or preparation["index_files_sha256"][index.name]!=sha(index)):
        raise ValueError("unchanged full-page preparation required")
    cases=json.loads(index.read_text())
    manifest=read_manifest(args.pages/"pairs.jsonl")
    first_train=[r for r in manifest if r["task"]=="qa" and r["split"]=="train"][:2]
    first_held=[r for r in manifest if r["task"]=="qa" and r["split"]=="validation"][:2]
    if [r["id"] for r in cases]!=[r["id"] for r in first_train+first_held]:
        raise ValueError("fixed first training/held-out pairs required")
    for r in cases:
        if set(r)!={"id","input","input_sha256","seed"} or sha(args.controls/r["input"])!=r["input_sha256"]:
            raise ValueError("input-only source changed")
        with Image.open(args.controls/r["input"]) as im:
            if im.size!=(1024,1024):raise ValueError("complete native page must remain 1024-square")
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/"status.json",dict(stage="starting",pid=os.getpid()))
    started=time.perf_counter();observed=[]
    try:
        with gpu_zero() as device:
            torch.manual_seed(20268801);torch.cuda.manual_seed_all(20268801)
            pipe=load_pipeline(args.snapshot,device,codec_only=args.stage=="codec")
            native=native_implementation(pipe)
            metadata=dict(stage=args.stage,source=source,implementation=code,native_implementation=native,
                input_index_sha256=sha(index),manifest_sha256=sha(args.pages/"pairs.jsonl"),
                original_input_canvas=[1024,1024],answer_canvas=[512,256],independent_language_goal_complete=False)
            if args.stage=="codec":
                with torch.inference_mode():
                    for i,r in enumerate(cases):
                        features=encode_reference(pipe,args.controls/r["input"])
                        mean=pipe.vae.bn.running_mean.view(1,-1,1,1).to(features)
                        std=(pipe.vae.bn.running_var.view(1,-1,1,1)+pipe.vae.config.batch_norm_eps).sqrt().to(features)
                        native_features=pipe._unpatchify_latents(features*std+mean)
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            decoded=pipe.vae.decode(native_features,return_dict=False)[0]
                        rgb=pipe.image_processor.postprocess(decoded,output_type="pt").detach().cpu()
                        if rgb.shape!=(1,3,1024,1024) or not bool(torch.isfinite(rgb).all()):
                            raise ValueError("actual full-page reconstruction invalid")
                        if torch.cuda.max_memory_allocated()>=22*1024**3:
                            raise RuntimeError("registered full-page memory limit exceeded")
                        output=save_answer(pipe,rgb,args.output/f"{i:03}_reconstruction")
                        original=np.asarray(Image.open(args.controls/r["input"]).convert("RGB"),dtype=np.float32)/255
                        reconstructed=rgb[0].permute(1,2,0).float().numpy()
                        a=original.mean(axis=-1)<.5;b=reconstructed.mean(axis=-1)<.5
                        observed.append(dict(id=r["id"],input_sha256=r["input_sha256"],input_feature_shape=list(features.shape),
                            reconstruction=output,mean_absolute_error=float(np.abs(original-reconstructed).mean()),
                            thresholded_ink_f1=2*int((a&b).sum())/int(a.sum()+b.sum())))
                        print("Reconstructed full page",r["id"],flush=True)
                        del features,native_features,decoded,rgb
                metadata.update(used_adapter_checkpoint=False,optimizer_updates=0)
            else:
                if sha(args.initial)!=INITIAL:raise ValueError("fixed complete balanced checkpoint required")
                state=torch.load(args.initial,map_location="cpu",weights_only=True)
                if state["metadata"]["source"]!=source or state["metadata"]["condition_sha256"]!=sha(args.condition):
                    raise ValueError("fixed source/condition differs")
                add_adapters(pipe.transformer,16);restore_adapters(pipe.transformer,state["adapters"])
                beginning=state["adapters"];del state
                parameters=[p for p in pipe.transformer.parameters() if p.requires_grad]
                optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01)
                fixed=condition_cache(args.condition)
                reader=RecognitionTeacher(args.teacher,device);reader.model.train()
                metadata.update(initial_checkpoint_sha256=INITIAL,condition_sha256=sha(args.condition),
                    training_only_reader_source=reader.source,objective="unchanged final-image recognition CE + 10 balanced L1",
                    parameters=sum(p.numel() for p in parameters),updates_discarded=True)
                pipe.transformer.enable_gradient_checkpointing()
                for i,row in enumerate(first_train):
                    seed=20268802+i
                    if i==0:
                        pipe.transformer.eval()
                        native_pixels=answer(pipe,row["input_path"],fixed,seed=seed,width=512,height=256)
                        save_answer(pipe,native_pixels,args.output/"first_native_forward")
                    pipe.transformer.train();optimizer.zero_grad(set_to_none=True)
                    features=encode_reference(pipe,row["input_path"])
                    result=sample_page(pipe,features,fixed["positive"],seed=seed,differentiable=True,capture=i==0)
                    if i==0 and not torch.equal(native_pixels,result["rgb"].detach().cpu()):
                        raise ValueError("first actual differentiable full-page forward differs from native inference")
                    target=torch.from_numpy(np.array(Image.open(row["target_path"]).convert("RGB")).copy()).permute(2,0,1)[None].to(device).float()/255
                    loss,components=final_image_loss(result,target,row["reference_text"],reader)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite full-page loss")
                    loss.backward()
                    probes={str(k):float(v.grad.float().norm()) if v.grad is not None and bool(torch.isfinite(v.grad).all()) else 0.
                        for k,v in result["probes"]}
                    if i==0 and (set(probes)!={"0","24","49"} or not all(v>0 for v in probes.values())):
                        raise ValueError("missing full-trajectory gradients")
                    if any(p.grad is not None for m in (pipe.vae,reader.model) for p in m.parameters()) or any(
                            p.grad is not None for p in pipe.transformer.parameters() if not p.requires_grad):
                        raise ValueError("frozen parameter received gradient")
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    if not bool(norm>0):raise ValueError("zero adapter gradient")
                    output=save_answer(pipe,result["rgb"].detach().cpu(),args.output/f"update_{i+1:03}_before")
                    optimizer.step()
                    if any(int(s["step"])!=i+1 for s in optimizer.state.values()):raise ValueError("optimizer steps differ")
                    if torch.cuda.max_memory_allocated()>=22*1024**3:raise RuntimeError("registered full-page memory limit exceeded")
                    observed.append(dict(id=row["id"],seed=seed,input_sha256=row["input_sha256"],target_sha256=row["target_sha256"],
                        loss=float(loss.detach()),**components,gradient_norm=float(norm),prediction_gradient_norms=probes,
                        output=output,initial_native_forward_equal=True if i==0 else None))
                    save_json(args.output/"updates.json",observed);print(json.dumps(observed[-1]),flush=True)
                    del result,target,features,loss
                final=adapter_state(pipe.transformer)
                changed=sum(not torch.equal(value,beginning[name]) for name,value in final.items())
                if not changed:raise ValueError("no actual parameter change")
                torch.save(dict(adapters=final,metadata=metadata,not_for_inference=True),args.output/"discarded_adapters.pt")
                metadata.update(optimizer_updates=2,changed_adapter_tensors=changed,all_optimizer_steps=2,
                    discarded_adapters_sha256=sha(args.output/"discarded_adapters.pt"))
            if implementation()!=code:raise ValueError("actual probe implementation changed while live")
            report=dict(metadata=metadata,observations=observed,elapsed_seconds=time.perf_counter()-started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,model_execution_completed=True,
                raw_review_completed=False,semantic_learning_validated=False,language_quality_qualified=False,general_ilm_goal_complete=False)
            save_json(args.output/"report.json",report);save_json(args.output/"status.json",dict(stage="completed",pid=None))
    except BaseException as error:
        save_json(args.output/"status.json",dict(stage="failed",pid=None,reason=str(error)));raise


if __name__=="__main__":
    main()
