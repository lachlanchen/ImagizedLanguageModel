#!/usr/bin/env python3
"""Execute the complete real image-language actor with a separate codec output."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.blind_vision_transcription import check_host_memory,gpu_memory
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.codec_block_writer import CodecBlockWriter,load_qualified_writing_codec
from ilm.visual_lm.codec_visual_language_actor import CodecVisualLanguageActor
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor,file_sha
from scripts.check_grounded_image_actor import ROOT,WRITING_SHA,SOURCE_NAMES,atomic_json,cpu_tree,exact

SOURCES=sorted(set(SOURCE_NAMES)|{"scripts/check_codec_visual_language_actor.py",
    "ilm/visual_lm/codec_block_writer.py","ilm/visual_lm/continuous_glyph_codec.py",
    "ilm/visual_lm/codec_visual_language_actor.py","references/codec_visual_language_actor_preflight_plan.md"})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/codec_visual_actor_preflight_20261105/attempt01")
    args=parser.parse_args();out=args.out
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise ValueError("explicit physical GPU 0 and deterministic cuBLAS required")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"report.json").exists():raise FileExistsError("completed preflight is immutable")
    lock=(out/"operation.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begun=time.perf_counter();checks={};evidence={}
    sources={name:file_sha(ROOT/name) for name in SOURCES}
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    prerequisites={
        "artifacts/codec_output_v67_20261105/report.json":"058cab147a71ce40ae9fc8a327a7265ef145fa1247a9915ab9e60d29b3901a01",
        "references/evidence/codec_output_v67_result.json":"361d877eec1cb817105e82c81f4bf508272595b6a106adbe39293759f9cfb32b",
        "artifacts/grounded_image_actor_preflight_20261105/attempt03/report.json":"2f1cfa22dbdd177a8baf4eca5fe0617fa8a03fbeabe4c51a4fac8c8f825d50fb"}
    def status(stage,**values):
        row=dict(stage=stage,pid=os.getpid(),physical_gpu=0,elapsed_seconds=time.perf_counter()-begun,**values)
        atomic_json(out/"status.json",row);print(json.dumps(row),flush=True)
    try:
        for name,digest in sources.items():
            committed=subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)
            import hashlib
            if hashlib.sha256(committed).hexdigest()!=digest:
                raise ValueError("commit complete-actor preflight sources before execution")
        check_host_memory()
        if gpu_memory()>256:raise RuntimeError("GPU 0 occupied; preserve the existing job")
        for path,digest in prerequisites.items():
            if file_sha(ROOT/path)!=digest:raise ValueError("qualified prerequisite changed: "+path)
        qualified=json.loads((ROOT/"references/evidence/codec_output_v67_result.json").read_text())
        if not qualified["output_interface_pass"] or not all(qualified["gates"].values()):
            raise ValueError("faithful output interface must qualify first")
        torch.set_num_threads(4);torch.manual_seed(20266801);torch.cuda.manual_seed_all(20266801)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        device=torch.device("cuda:0")
        data=CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_train_20261105",
            purpose="train",expected_report_sha256=WRITING_SHA)
        prior=json.loads((ROOT/"artifacts/grounded_image_actor_preflight_20261105/attempt03/report.json").read_text())
        resources=prior["evidence"]["resource_rows"]
        def example(index):
            image,boundary=data[index]
            prompt=fold_writing_prompt(image[...,:boundary*64]).pixels
            targets=[b.clone() for b in image[...,boundary*64:-128].split(64,dim=-1)]
            return prompt,targets
        for row in resources.values():
            prompt,targets=example(row["index"])
            if data.names[row["index"]]!=row["file"] or list(prompt.shape)!=row["page_shape"] or len(targets)!=row["answer_blocks"]:
                raise ValueError("actual largest resource witness differs")
        status("loading_complete_actor")
        base,provenance=load_grounded_image_actor(ollama_root=ROOT.parent/"LocalLLM/.local/models/ollama",
            gguf_source=ROOT.parent/"LocalLLM/.local/tools/llama.cpp/gguf-py",device=device,
            progress=lambda loaded,total:status("loading_complete_actor",loaded=loaded,total=total))
        codec=load_qualified_writing_codec(ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt",device)
        base.writer=CodecBlockWriter(codec)
        actor=CodecVisualLanguageActor(base)
        setup=time.perf_counter()-begun
        checks["all_native_layers_retained"]=len(base.vision.blocks)==27 and len(base.context.layers)==36
        checks["no_language_vocabulary"]=base.context.embed_tokens is None and base.context.config.vocab_size==0
        checks["all_real_weights_finite"]=all(bool(torch.isfinite(p).all()) for p in actor.parameters())
        trainable=[p for p in actor.parameters() if p.requires_grad]
        evidence["parameters"]=dict(total=sum(p.numel() for p in actor.parameters()),
            trainable=sum(p.numel() for p in trainable),codec=sum(p.numel() for p in codec.parameters()))
        checks["fixed_renderer_frozen"]=not any(p.requires_grad for p in codec.parameters())
        checks["actual_trainable_count"]=evidence["parameters"]["trainable"]==9351937
        base.context.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        attention_autocast=[]
        hook=base.context.layers[0].self_attn.register_forward_pre_hook(
            lambda module,inputs:attention_autocast.append(torch.is_autocast_enabled()))
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.95),weight_decay=.01,fused=True)
        generator=torch.Generator(device=device).manual_seed(20266802)
        def update(index,commit=True):
            actor.train();optimizer.zero_grad(set_to_none=True)
            prompt,targets=example(index)
            loss,values=actor.training_loss(prompt,targets,generator=generator,writer_blocks=16)
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise FloatingPointError("complete real-model gradients missing or nonfinite")
            if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):
                raise ValueError("a frozen base or codec parameter received gradients")
            parameters=dict(actor.named_parameters())
            selected=("actor.queries","bridge.projection.weight","actor.stop.weight",
                "actor.context.layers.0.self_attn.q_proj.up","actor.context.layers.35.self_attn.v_proj.up")
            norms={name:float(parameters[name].grad.float().norm()) for name in selected}
            if not all(value>0 for value in norms.values()):raise ValueError("required zero gradients: "+str(norms))
            before=float(torch.nn.utils.clip_grad_norm_(trainable,1.))
            if commit:optimizer.step()
            result=dict(index=index,file=data.names[index],target_blocks=len(targets),loss=float(loss.detach()),
                state=float(values["state"]),image=float(values["image"]),stop=float(values["stop"]),
                selected_blocks=values["selected_blocks"].tolist(),gradient_norms=norms,gradient_norm_before_clip=before)
            del loss,values
            optimizer.zero_grad(set_to_none=True)
            return result
        status("initializing_actual_optimizer")
        first=update(0)
        resource_updates=[]
        for row in {v["index"]:v for v in resources.values()}.values():
            status("complete_resource_backward",**row)
            resource_updates.append(update(row["index"],commit=False))
        checks["full_resource_backward_with_adam_moments"]=True
        evidence["resource_backward"]=resource_updates
        status("exact_optimizer_rng_replay")
        def snapshot():
            return dict(trainable=actor.trainable_state(),optimizer=cpu_tree(optimizer.state_dict()),
                generator=generator.get_state().cpu(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device))
        saved=snapshot();torch.save(saved,out/"resume_checkpoint.pt")
        continued=update(len(data)//2);reference=snapshot()
        actor.load_trainable_state(saved["trainable"]);optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"]);torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"],device)
        replay=update(len(data)//2);recovered=snapshot()
        for key in reference:checks["exact_resume_"+key]=exact(reference[key],recovered[key])
        checks["exact_resume_update_metrics"]=continued==replay
        evidence["updates"]=[first,continued,replay]
        del saved,reference,recovered
        actor.eval();prompt,targets=example(0)
        if len(targets)<2:raise ValueError("two actual future blocks required for causal intervention")
        status("causal_and_cached_image_path")
        with torch.no_grad():
            ordinary,_,_=base.conditions(prompt,targets[:2])
            altered,_,_=base.conditions(prompt,(targets[0],-targets[1]))
            future=float((ordinary[:2]-altered[:2]).abs().max())
            plan_future=float((actor.bridge(ordinary[:2])-actor.bridge(altered[:2])).abs().max())
            stop_future=float((actor.stop_logits(ordinary[:2])-actor.stop_logits(altered[:2])).abs().max())
            a,deep,_=base.encode_images(targets[:2]);b,other,_=base.encode_images((targets[0],-targets[1]))
            isolation=max(float((x-y).abs().max()) for x,y in zip([a[0],*[v[0] for v in deep]],[b[0],*[v[0] for v in other]]))
            full,_,_=base.conditions(prompt,targets[:1]);_,past,offset=base.conditions(prompt,use_cache=True)
            cached,_,_=base.append_generated(targets[0],past,offset=offset)
            cache_abs=float((full[-1:]-cached).abs().max())
            cache_rel=cache_abs/max(float(full[-1:].abs().max()),1e-12)
            x,y=actor.bridge(full[-1:]),actor.bridge(cached)
            plan_abs=float((x-y).abs().max());plan_rel=plan_abs/max(float(x.abs().max()),1e-12)
            result=actor.generate(prompt,maximum_blocks=2)
            repeated=actor.generate(prompt,maximum_blocks=2)
        checks["future_query_exact"]=future==0
        checks["future_output_plan_exact"]=plan_future==0
        checks["future_stop_exact"]=stop_future==0
        checks["encoder_isolation_exact"]=isolation==0
        checks["cached_context_relative_error"]=cache_rel<=.02
        checks["cached_plan_relative_error"]=plan_rel<=.02
        checks["context_attention_fp32"]=bool(attention_autocast) and not any(attention_autocast)
        checks["deterministic_image_generation"]=exact(result,repeated)
        checks["actual_image_generation_exercised"]=len(result["images"])>0
        checks["finite_image_generation"]=all(bool(torch.isfinite(image).all()) for image in result["images"])
        hook.remove()
        if result["images"]:
            pixels=torch.cat(result["images"],dim=-1)[0,0]
            Image.fromarray(((pixels+1)*127.5).numpy().astype(np.uint8)).save(out/"untrained_execution.png")
        evidence.update(future_query_max_abs=future,future_plan_max_abs=plan_future,future_stop_max_abs=stop_future,
            encoder_isolation_max_abs=isolation,cached_context_max_abs=cache_abs,cached_context_relative=cache_rel,
            cached_plan_max_abs=plan_abs,cached_plan_relative=plan_rel,generation=dict(blocks=len(result["images"]),
                events=result["events"],meaning_not_evaluated=True,training_updates=2),
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/1024**3)
        checks["memory_below_22_gib"]=evidence["peak_allocated_vram_gib"]<22
        checks["sources_unchanged"]=sources=={name:file_sha(ROOT/name) for name in SOURCES}
        report=dict(checks=checks,evidence=evidence,source_commit=revision,
            prerequisites_sha256=prerequisites,source_sha256=sources,
            provenance=provenance,writing_report_sha256=data.report_sha256,setup_seconds=setup,
            elapsed_seconds=time.perf_counter()-begun,physical_gpu=0,development_examples=0,teacher_calls=0,
            language_quality_claim=False,overall_ilm_goal_complete=False)
        atomic_json(out/("report.json" if all(checks.values()) else "failed_verification.json"),report)
        if not all(checks.values()):raise AssertionError(checks)
        status("complete",all_execution_checks_passed=True,language_quality_claim=False)
    except BaseException as error:
        atomic_json(out/"failure.json",dict(error=repr(error),checks=checks,evidence=evidence,sources=sources))
        status("failed",error=repr(error));raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=="__main__":main()
