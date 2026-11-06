#!/usr/bin/env python3
"""Qualify actual inherited-body image continuation, gradients and feedback."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.codec_block_writer import CodecBlockWriter,load_qualified_writing_codec,CODEC_SHA
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor,MODEL_SHA
from ilm.visual_lm.native_image_continuation import NativeImagePrefix,NativeImageLanguageModel
from scripts.cache_semantic_images_v72 import ROOT,CODEC,CONTROLS,CONTROL_SHA,sha,atomic_json,committed_identity
from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer,embed_rows
from scripts.probe_suffix_kv_v78 import native_identity
from scripts.train_clean_corpus_v75 import setup

V77=ROOT/"artifacts/reader_content_v77_20261106"
V78=ROOT/"artifacts/suffix_kv_v78_20261106"
SOURCES=("scripts/check_native_image_execution_v79.py","ilm/visual_lm/native_image_continuation.py",
    "ilm/visual_lm/suffix_attention_mask.py","ilm/visual_lm/clean_writing_flow.py",
    "ilm/visual_lm/full_rank_writing_flow.py","ilm/visual_lm/semantic_writing_model.py",
    "ilm/visual_lm/codec_block_writer.py","ilm/visual_lm/grounded_image_weights.py",
    "ilm/visual_lm/grounded_image_actor.py","scripts/check_native_visual_transfer_v71.py",
    "references/native_image_execution_v79_protocol.md","references/native_image_continuation_v79_design.md")


def digest(values):
    value=hashlib.sha256()
    for name,tensor in sorted(values):
        value.update(name.encode());value.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        value.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def source_identity():
    v78=json.loads((V78/"verification.json").read_text())
    v77=json.loads((V77/"verification.json").read_text())
    if (not v78["offline_checks_pass"] or not v78["masked_poison_states_identical"]
            or v78["generation_report_sha256"]!=sha(V78/"generation_report.json")
            or v77["generation_report_sha256"]!=sha(V77/"generation_report.json")
            or v78["source_verification_sha256"]!=sha(V77/"verification.json")
            or v78["preflight_sha256"]!=sha(V78/"preflight.pt") or sha(CONTROLS)!=CONTROL_SHA):
        raise ValueError("completed authenticated source-reading/visibility fixtures required")
    return dict(**committed_identity(SOURCES),native=native_identity(),
        source_visibility_verification_sha256=sha(V78/"verification.json"),
        source_preflight_sha256=sha(V78/"preflight.pt"),
        source_reader_verification_sha256=sha(V77/"verification.json"),source_model_sha256=MODEL_SHA,
        codec_sha256=CODEC_SHA,seed=20267910,optimizer_updates=0,quality_claim=False)


def save_answer(path,images):
    if len(images):
        pixels=torch.cat(list(images),dim=-1)[0].add(1).mul(127.5).round().to(torch.uint8).numpy()
        Image.fromarray(pixels).save(path)


def evaluate(out):
    if out.exists():
        raise FileExistsError("choose a fresh immutable execution directory")
    identity=source_identity()
    out.mkdir(parents=True)
    atomic_json(out/"identity.json",identity)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup();torch.manual_seed(20267910);torch.cuda.manual_seed_all(20267910)
        begun=time.perf_counter();local=ROOT.parent/"LocalLLM/.local"
        base,provenance=load_grounded_image_actor(ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        base.writer=torch.nn.Identity();base.requires_grad_(False).eval()
        writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,device))
        controls=torch.load(CONTROLS,map_location="cpu",weights_only=True)
        input_scale=float(controls["after"].square().mean().sqrt())
        model=NativeImageLanguageModel(base.context,writer,input_scale=input_scale).to(device).eval()
        fixture=torch.load(V78/"preflight.pt",map_location="cpu",weights_only=True)
        prefill=torch.load(V77/"outputs/copy_00_normal.pt",map_location="cpu",weights_only=True)
        prefix=NativeImagePrefix(fixture["original"],prefill["prefill_memory"][:,-1:],
            torch.tensor([fixture["metadata"]["rotary_offset"]])).to(device)
        reference=fixture["states"]["full_original"].to(device)
        archive,tokenizer,markers=archive_and_tokenizer()
        numeric=embed_rows(archive,[fixture["source_first_id"]]).to(device)[None]
        del archive,tokenizer,markers
        atomic_json(out/"status.json",dict(stage="native_forward_and_gradient",pid=os.getpid()))
        setup_seconds=time.perf_counter()-begun
        frozen_versions={n:p._version for n,p in model.continuation.context.named_parameters()}
        prefix_hash=digest((f"{i}/{kind}",t) for i,pair in enumerate(prefix.key_values) for kind,t in zip(("k","v"),pair))
        with torch.no_grad():
            ordinary,_=model.continuation(numeric,prefix,checkpoint_layers=False)
        forward_error=float((ordinary-reference).abs().max())
        grad_input=numeric.detach().clone().requires_grad_(True)
        checkpointed,_=model.continuation(grad_input,prefix,checkpoint_layers=True)
        grad_forward_error=float((checkpointed-reference).abs().max())
        functional=torch.randn(checkpointed.shape,device=device,generator=torch.Generator(device=device).manual_seed(20267911))
        input_gradient=torch.autograd.grad((checkpointed*functional).mean(),grad_input)[0]
        if (max(forward_error,grad_forward_error)>1e-5 or not torch.isfinite(input_gradient).all()
                or not bool(input_gradient.abs().max()>0)):
            raise AssertionError("actual native forward or all-layer input-gradient propagation failed")
        torch.save(dict(reference=reference.cpu(),ordinary=ordinary.cpu(),checkpointed=checkpointed.detach().cpu(),
            numerical_input=numeric.cpu(),input_gradient=input_gradient.cpu(),functional=functional.cpu()),out/"native_numeric.pt")
        del checkpointed,grad_input,input_gradient
        padded=NativeImagePrefix(tuple((F.pad(k,(0,0,0,17)),F.pad(v,(0,0,0,17))) for k,v in prefix.key_values),
            prefix.initial_state,prefix.rotary_offsets,valid_lengths=torch.tensor([prefix.length],device=device))
        with torch.no_grad():
            padding_state,_=model.continuation(numeric,padded,checkpoint_layers=False)
        padding_error=float((padding_state-reference).abs().max())
        if padding_error>1e-4:
            raise AssertionError("prefix padding changed the native semantic continuation")
        torch.save(padding_state.cpu(),out/"padding_state.pt")
        del padded,padding_state
        reference_image=ROOT/"data/general_instructions/copy_transfer_v75_20261106/references/00.png"
        pixels=torch.from_numpy(np.array(Image.open(reference_image),copy=True)).float().div(127.5).sub(1)[None,None]
        physical=torch.cat(list(pixels.split(64,dim=-1)),dim=0).to(device)
        with torch.no_grad():
            codes=writer.encode_targets(physical)[None]
            parallel=model.conditions(prefix,codes,checkpoint_layers=False)
            altered=codes.clone()
            altered[:,1:]=torch.randn(altered[:,1:].shape,device=device,generator=torch.Generator(device=device).manual_seed(20267912))
            alternate=model.conditions(prefix,altered,checkpoint_layers=False)
            current=prefix;incremental=[prefix.initial_state.float()]
            for i in range(codes.shape[1]):
                hidden,cache=model.continuation(model.embed_images(codes[:,i:i+1]),current,
                    checkpoint_layers=False,collect_cache=True)
                current=NativeImagePrefix(cache,hidden[:,-1:],current.rotary_offsets+2)
                incremental.append(current.initial_state)
            incremental=torch.cat(incremental,dim=1)
        future_error=float((parallel[:,:2]-alternate[:,:2]).abs().max())
        later_effect=float((parallel[:,2:]-alternate[:,2:]).abs().max())
        incremental_error=float((parallel-incremental).abs().max())
        if future_error>1e-5 or later_effect<=1e-5 or incremental_error>1e-4:
            raise AssertionError("causal future exclusion or parallel/incremental alignment failed")
        torch.save(dict(parallel=parallel.cpu(),alternate=alternate.cpu(),incremental=incremental.cpu(),
            source_codes=codes.cpu()),out/"causal_states.pt")
        del current,cache,hidden,incremental,parallel,alternate,altered
        atomic_json(out/"status.json",dict(stage="image_adapter_gradient",pid=os.getpid()))
        model.train()
        states=model.conditions(prefix,codes)
        probe=torch.randn(states[:,1:].shape,device=device,generator=torch.Generator(device=device).manual_seed(20267913))
        auxiliary=(states[:,1:]*probe).mean()
        auxiliary.backward()
        input_grads={n:p.grad.detach().cpu().clone() for n,p in model.image_input.named_parameters() if p.grad is not None}
        if not input_grads or not all(bool(torch.isfinite(g).all()) for g in input_grads.values()) or not any(bool(g.abs().max()>0) for g in input_grads.values()):
            raise AssertionError("actual image-input adapter cannot learn through the inherited body")
        torch.save(input_grads,out/"image_input_gradients.pt")
        del states,auxiliary
        model.zero_grad(set_to_none=True)
        loss,metrics=model.training_loss(prefix,codes,torch.tensor([codes.shape[1]],device=device),
            generator=torch.Generator(device=device).manual_seed(20267914))
        loss.backward()
        head_grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()
            if n.startswith(("flow.","stop.")) and p.grad is not None}
        if not torch.isfinite(loss) or not head_grads or not all(bool(torch.isfinite(g).all()) for g in head_grads.values()) or not any(bool(g.abs().max()>0) for g in head_grads.values()):
            raise AssertionError("supervised image-code/stop head gradients failed")
        torch.save(head_grads,out/"head_gradients.pt")
        flow_input_grads={n:float(p.grad.abs().max()) for n,p in model.image_input.named_parameters() if p.grad is not None}
        loss_value=float(loss.detach());metric_values={k:float(v) if isinstance(v,torch.Tensor) else v for k,v in metrics.items()}
        del loss
        model.zero_grad(set_to_none=True);model.eval()
        patch=model.trainable_state_dict()
        patch_hash=digest(patch.items())
        torch.save(dict(model_kind="native_continuous_image_v79",trainable=patch,identity=identity,
            frozen_weights_saved=False,trained=False),out/"initial_adapters.pt")
        with torch.no_grad():
            model.image_input[1].weight[0,0].add_(.125)
        model.load_trainable_state_dict(patch)
        restored=digest(model.trainable_state_dict().items())==patch_hash
        if not restored:
            raise AssertionError("image-only adapter checkpoint round-trip failed")
        atomic_json(out/"status.json",dict(stage="physical_image_feedback",pid=os.getpid()))
        generated=model.generate(prefix,generator=torch.Generator(device=device).manual_seed(20267915),maximum_blocks=2)
        torch.save(generated,out/"untrained_answer.pt")
        save_answer(out/"untrained_answer.png",generated["images"])
        feedback=[hashlib.sha256(writer.encode_targets(image[None].to(device)).cpu().numpy().tobytes()).hexdigest()
            for image in generated["images"]]
        if len(feedback)!=2 or feedback!=generated["feedback_sha256"] or generated["stopped"] or not generated["capped"]:
            raise AssertionError("unforced physical image feedback execution failed")
        saved_bias=model.stop.bias.detach().clone()
        with torch.no_grad():model.stop.bias.fill_(30)
        forced=model.generate(prefix,generator=torch.Generator(device=device).manual_seed(20267915),maximum_blocks=2)
        with torch.no_grad():model.stop.bias.copy_(saved_bias)
        if len(forced["images"]) or not forced["stopped"] or forced["capped"]:
            raise AssertionError("forced empty stop branch failed")
        torch.save(forced,out/"forced_empty.pt")
        if digest(model.trainable_state_dict().items())!=patch_hash:
            raise AssertionError("execution changed the untrained new parameters")
        if (any(p.requires_grad or p.grad is not None or p._version!=frozen_versions[n]
                for n,p in model.continuation.context.named_parameters())
                or model.continuation.context.embed_tokens is not None or model.continuation.context.config.vocab_size!=0):
            raise AssertionError("frozen vocabulary-free source boundary changed")
        final_prefix_hash=digest((f"{i}/{kind}",t) for i,pair in enumerate(prefix.key_values) for kind,t in zip(("k","v"),pair))
        if final_prefix_hash!=prefix_hash:
            raise AssertionError("checkpointed forward/backward mutated the prompt prefix")
        peak=torch.cuda.max_memory_allocated()/1024**3
        if peak>=24:
            raise AssertionError("execution exceeded the physical24GiB budget")
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in (".pt",".png")}
        atomic_json(out/"report.json",dict(identity=identity,provenance=provenance,files_sha256=files,
            forward_max_abs_error=forward_error,checkpoint_forward_max_abs_error=grad_forward_error,
            padding_max_abs_error=padding_error,future_max_abs_error=future_error,later_image_effect=later_effect,
            incremental_max_abs_error=incremental_error,prefix_sha256=prefix_hash,prefix_unchanged=True,
            frozen_context_unchanged=True,new_parameter_checkpoint_exact=restored,new_parameter_sha256=patch_hash,
            source_layers_exercised=len(base.context.layers),source_prefix_positions=prefix.length,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            whole_model_parameters=sum(p.numel() for p in model.parameters()),input_scale=input_scale,
            finite_image_input_gradients=True,finite_head_gradients=True,loss_without_update=loss_value,
            supervised_metrics=metric_values,initial_flow_loss_image_input_gradient_max=flow_input_grads,
            physical_feedback_exact=True,forced_empty_stop=True,forced_branch_quality_claim=False,
            elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
            peak_allocated_vram_gib=peak,optimizer_updates=0,trained=False,
            language_quality_claim=False,general_ilm_goal_complete=False))
        atomic_json(out/"status.json",dict(stage="execution_complete_awaiting_cpu_verification",pid=None))
        print(json.dumps(dict(peak_gib=peak,forward_error=forward_error,incremental_error=incremental_error,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)),indent=2))


def verify(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only independent execution verification required")
    torch.set_num_threads(4)
    report=json.loads((out/"report.json").read_text());identity=source_identity()
    if {k:v for k,v in identity.items() if k!="source_commit"}!={k:v for k,v in report["identity"].items() if k!="source_commit"}:
        raise ValueError("registered execution implementation or source fixture changed")
    for name,digest_value in report["files_sha256"].items():
        if sha(out/name)!=digest_value:
            raise ValueError("execution numeric/physical artifact changed")
    native=torch.load(out/"native_numeric.pt",map_location="cpu",weights_only=True)
    fixture=torch.load(V78/"preflight.pt",map_location="cpu",weights_only=True)
    if not torch.equal(native["reference"],fixture["states"]["full_original"]):
        raise ValueError("execution used a different source next-state reference")
    for name in ("ordinary","checkpointed"):
        if float((native[name]-native["reference"]).abs().max())>1e-5:
            raise ValueError("retained native forward differs")
    if not torch.isfinite(native["input_gradient"]).all() or not bool(native["input_gradient"].abs().max()>0):
        raise ValueError("retained complete-body input gradient failed")
    padded=torch.load(out/"padding_state.pt",map_location="cpu",weights_only=True)
    if float((padded-native["reference"]).abs().max())>1e-4:
        raise ValueError("retained padding comparison failed")
    states=torch.load(out/"causal_states.pt",map_location="cpu",weights_only=True)
    if (float((states["parallel"][:,:2]-states["alternate"][:,:2]).abs().max())>1e-5
            or float((states["parallel"][:,2:]-states["alternate"][:,2:]).abs().max())<=1e-5
            or float((states["parallel"]-states["incremental"]).abs().max())>1e-4):
        raise ValueError("retained causal or incremental source states failed")
    for name in ("image_input_gradients.pt","head_gradients.pt"):
        gradients=torch.load(out/name,map_location="cpu",weights_only=True)
        if not gradients or not all(bool(torch.isfinite(v).all()) for v in gradients.values()) or not any(bool(v.abs().max()>0) for v in gradients.values()):
            raise ValueError("retained trainable gradients failed")
    patch=torch.load(out/"initial_adapters.pt",map_location="cpu",weights_only=True)
    if patch["frozen_weights_saved"] or patch["trained"] or digest(patch["trainable"].items())!=report["new_parameter_sha256"]:
        raise ValueError("untrained image-only checkpoint boundary failed")
    writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    result=torch.load(out/"untrained_answer.pt",map_location="cpu",weights_only=True)
    feedback=[hashlib.sha256(writer.encode_targets(image[None]).cpu().numpy().tobytes()).hexdigest() for image in result["images"]]
    if len(feedback)!=2 or feedback!=result["feedback_sha256"] or result["stopped"] or not result["capped"]:
        raise ValueError("independent physical feedback re-encoding failed")
    forced=torch.load(out/"forced_empty.pt",map_location="cpu",weights_only=True)
    if len(forced["images"]) or not forced["stopped"]:
        raise ValueError("forced stop branch failed")
    if not report["prefix_unchanged"] or not report["frozen_context_unchanged"] or report["peak_allocated_vram_gib"]>=24:
        raise ValueError("frozen state or device budget failed")
    receipt=dict(report_sha256=sha(out/"report.json"),identity=identity,offline_checks_pass=True,
        retained_forward_and_gradient_vectors_checked=True,independent_feedback_reencodings=2,
        whole_native_forward_backward_independently_replayed=False,source_layers_exercised=report["source_layers_exercised"],
        trainable_parameters=report["trainable_parameters"],peak_allocated_vram_gib=report["peak_allocated_vram_gib"],
        optimizer_updates=0,trained=False,language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(out/"verification.json",receipt);atomic_json(out/"status.json",dict(stage="completed_verified",pid=None))
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/native_image_execution_v79_20261106")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    (verify if args.verify else evaluate)(args.out)
