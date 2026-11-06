#!/usr/bin/env python3
"""Offline visibility intervention on the source reader's retained suffix KV."""

import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
import transformers
from transformers.cache_utils import DynamicCache

from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor
from ilm.visual_lm.image_context_interface import ImageContextInterface
from ilm.visual_lm.suffix_attention_mask import causal_suffix_mask
from scripts.cache_semantic_images_v72 import ROOT,CONTROLS,CONTROL_SHA,sha,atomic_json,committed_identity
from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer,compile_controls,embed_rows
from scripts.probe_reader_content_v77 import cpu,load_image
from scripts.train_clean_corpus_v75 import setup
from scripts.verify_native_visual_baseline_v70 import distance

SOURCE=ROOT/"artifacts/reader_content_v77_20261106"
SOURCES=("scripts/probe_suffix_kv_v78.py","ilm/visual_lm/suffix_attention_mask.py",
    "references/suffix_kv_visibility_v78_protocol.md","scripts/probe_reader_content_v77.py",
    "scripts/check_native_visual_transfer_v71.py","scripts/train_clean_corpus_v75.py",
    "ilm/visual_lm/grounded_image_weights.py","ilm/visual_lm/grounded_image_actor.py",
    "ilm/visual_lm/image_context_interface.py","scripts/verify_native_visual_baseline_v70.py")


def native_identity():
    import transformers.cache_utils as cache
    import transformers.masking_utils as masking
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    return dict(torch_version=torch.__version__,transformers_version=transformers.__version__,
        source_sha256={name:sha(inspect.getfile(module)) for name,module in
            (("cache",cache),("masking",masking),("qwen3_vl",native))})


def source():
    prep=json.loads((SOURCE/"preparation.json").read_text())
    generated=json.loads((SOURCE/"generation_report.json").read_text())
    verified=json.loads((SOURCE/"verification.json").read_text())
    if (verified["generation_report_sha256"]!=sha(SOURCE/"generation_report.json")
            or verified["preparation_sha256"]!=sha(SOURCE/"preparation.json")
            or not verified["offline_checks_pass"] or verified["normal_memory_matches"]!=44
            or verified["helper_projection_argmax_agreements"]!=725 or sha(CONTROLS)!=CONTROL_SHA):
        raise ValueError("the complete verified V77 comparator is required")
    for name,digest in generated["identity"]["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("frozen V77 source changed")
    for row in prep["cases"]:
        normal=row["images"]["normal"]
        name=f"outputs/{row['group']}_{row['position']:02}_normal.pt"
        if sha(SOURCE/normal["file"])!=normal["sha256"] or sha(SOURCE/name)!=generated["raw_files_sha256"][name]:
            raise ValueError("source normal input/output changed")
    return prep,generated,verified


def mask_geometry_check():
    value=causal_suffix_mask(7,3,4,device="cpu")
    allowed=value[0,0]==0
    expected=torch.tensor([[0,0,0,0,1,1,1,1,0,0],
        [0,0,0,0,1,1,1,1,1,0],[0,0,0,0,1,1,1,1,1,1]],dtype=torch.bool)
    if value.shape!=(1,1,3,10) or not torch.equal(allowed,expected) or not bool((value[0,0][~allowed]==torch.finfo(torch.float32).min).all()):
        raise AssertionError("suffix/history visibility or future exclusion failed")


def prepare(out):
    cpu();mask_geometry_check()
    if out.exists():
        raise FileExistsError("choose a fresh diagnostic directory")
    prep,_,_=source()
    out.mkdir(parents=True)
    atomic_json(out/"preparation.json",dict(identity=committed_identity(SOURCES),native=native_identity(),
        source_preparation_sha256=sha(SOURCE/"preparation.json"),
        source_generation_sha256=sha(SOURCE/"generation_report.json"),
        source_verification_sha256=sha(SOURCE/"verification.json"),cases=44,
        suffix_positions=52,new_helper_generations=44,forced_preflight_forwards=4,
        cpu_future_exclusion_check=True,optimizer_updates=0,language_quality_claim=False,
        general_ilm_goal_complete=False))
    print("Registered44 masked helper calls; parallel-query future exclusion passes.")


def cache_geometry(cache):
    layers=[]
    for k,v in cache.to_legacy_cache():
        if k.shape!=v.shape or k.dtype!=v.dtype or k.shape[-2]!=cache.get_seq_length():
            raise ValueError("unexpected native dynamic KV geometry")
        layers.append(dict(shape=list(k.shape),dtype=str(k.dtype),element_size=k.element_size(),
            full_bytes=(k.numel()+v.numel())*k.element_size(),
            suffix_bytes=(k[...,-52:,:].numel()+v[...,-52:,:].numel())*k.element_size()))
    return dict(layers=layers,full_bytes=sum(r["full_bytes"] for r in layers),
        suffix_bytes=sum(r["suffix_bytes"] for r in layers),positions=cache.get_seq_length())


def clone_cache(cache,*,poison=False):
    result=[]
    for k,v in cache.to_legacy_cache():
        k,v=k.clone(),v.clone()
        if poison:
            k[...,:-52,:].fill_(7.);v[...,:-52,:].fill_(-11.)
        result.append((k,v))
    return DynamicCache.from_legacy_cache(tuple(result))


@torch.inference_mode()
def generate(out):
    mask_geometry_check()
    prep=json.loads((out/"preparation.json").read_text())
    identity=committed_identity(SOURCES)
    original,_,_=source()
    if (prep["identity"]["sources"]!=identity["sources"] or prep["native"]!=native_identity()
            or prep["source_verification_sha256"]!=sha(SOURCE/"verification.json")
            or (out/"generation_report.json").exists()):
        raise ValueError("registered sources differ or completed run already exists")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup();torch.manual_seed(20267710);torch.cuda.manual_seed_all(20267710)
        begun=time.perf_counter();local=ROOT.parent/"LocalLLM/.local"
        actor,provenance=load_grounded_image_actor(ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        actor.writer=torch.nn.Identity();actor.requires_grad_(False).eval()
        if any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith(".up")):
            raise ValueError("source adapters must remain zero")
        controls=torch.load(CONTROLS,map_location="cpu",weights_only=True)
        interface=ImageContextInterface(actor,**controls).requires_grad_(False).eval()
        archive,tokenizer,markers=archive_and_tokenizer()
        compiled,compiler=compile_controls(archive,tokenizer,markers)
        if any(not torch.equal(v,controls[k]) for k,v in compiled.items()):
            raise ValueError("numeric source controls changed")
        source_head=archive.omitted["output.weight"]
        head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),copy=True)).to(device,torch.bfloat16)
        eog=set(compiler["eog_ids"]);setup_seconds=time.perf_counter()-begun
        atomic_json(out/"compiler.json",compiler)

        def step(cache,embedding,offset,blocked):
            length=cache.get_seq_length()
            positions=torch.tensor([length,offset,offset,offset],device=device).reshape(4,1,1)
            mask=None if blocked is None else causal_suffix_mask(length,1,blocked,device=device)
            result=actor.context(inputs_embeds=embedding,position_ids=positions,
                past_key_values=cache,attention_mask=mask,use_cache=True)
            return result.last_hidden_state[:,-1:],result.past_key_values,mask

        inventory={}
        for i,row in enumerate(original["cases"]):
            path=out/"outputs"/f"{row['group']}_{row['position']:02}.pt"
            path.parent.mkdir(exist_ok=True)
            source_name=f"outputs/{row['group']}_{row['position']:02}_normal.pt"
            baseline=torch.load(SOURCE/source_name,map_location="cpu",weights_only=True)
            expected=dict(position=row["position"],group=row["group"],
                preparation_sha256=sha(out/"preparation.json"),source_call_sha256=sha(SOURCE/source_name))
            if not path.exists():
                arguments,metadata=interface.assemble(load_image(SOURCE/row["images"]["normal"]["file"]))
                result=actor.context(**arguments,use_cache=True)
                memory=result.last_hidden_state[:,-52:].to(torch.bfloat16).cpu()
                if tensor_digest(memory)!=row["expected_memory_sha256"]:
                    raise ValueError("normal prefill no longer matches V77 and original ILM")
                state=result.last_hidden_state[:,-1:].to(torch.bfloat16)
                past=result.past_key_values
                geometry=cache_geometry(past)
                if geometry["positions"]!=metadata["sequence_length"]:
                    raise ValueError("cached prefill positions differ")
                blocked=geometry["positions"]-52
                first=int(F.linear(state[:,-1].float(),head.float()).argmax(-1))
                if first!=baseline["result"]["selected_ids"][0]:
                    raise ValueError("first unmodified source decision changed")
                if i==0 and not (out/"preflight.pt").exists():
                    embedding=embed_rows(archive,[first]).to(device)[None]
                    stored={"original":tuple((k.cpu().clone(),v.cpu().clone()) for k,v in past.to_legacy_cache()),
                        "poisoned":tuple((k.cpu().clone(),v.cpu().clone()) for k,v in clone_cache(past,poison=True).to_legacy_cache()),
                        "states":{},"source_first_id":first,"metadata":metadata,"blocked":blocked,
                        "source_call_sha256":expected["source_call_sha256"]}
                    for route,masked,poison in (("masked_original",True,False),("masked_poisoned",True,True),
                            ("full_original",False,False),("full_poisoned",False,True)):
                        cache=clone_cache(past,poison=poison)
                        vector,_,mask=step(cache,embedding,metadata["rotary_offset"],blocked if masked else None)
                        stored["states"][route]=vector.cpu()
                        if masked:stored["mask"]=mask.cpu()
                        del cache,vector
                    vectors=stored["states"]
                    if (not torch.equal(vectors["masked_original"],vectors["masked_poisoned"])
                            or float((vectors["full_original"]-vectors["full_poisoned"]).abs().max())<=1e-5
                            or float((vectors["full_original"][0,0]-baseline["result"]["hidden_states"][1]).abs().max())>1e-5):
                        raise AssertionError("native masking/pre-poison or original continuation replay failed")
                    torch.save(stored,out/"preflight.pt")
                    del stored,vectors,embedding
                ids=[];states=[];selected=[];offset=metadata["rotary_offset"];stopped=False
                for index in range(64):
                    states.append(state[0,-1].float().cpu())
                    logits=F.linear(state[:,-1].float(),head.float())
                    token=int(logits.argmax(-1));ids.append(token);selected.append(float(logits[0,token]));del logits
                    if token in eog:
                        stopped=True;break
                    if index<63:
                        embedding=embed_rows(archive,[token]).to(device)[None]
                        state,past,_=step(past,embedding,offset,blocked)
                        offset+=1
                saved=dict(**expected,prefill_memory=memory,cache_geometry=geometry,blocked_prefix=blocked,
                    selected_ids=ids,selected_logits=selected,hidden_states=torch.stack(states),
                    text=tokenizer.decode(ids[:-1] if stopped else ids,skip_special_tokens=False),
                    stopped=stopped,capped=not stopped,metadata=metadata,source_helper_only=True)
                partial=path.with_suffix(".partial");torch.save(saved,partial);partial.replace(path)
                del result,state,past,memory
            saved=torch.load(path,map_location="cpu",weights_only=True)
            if any(saved[k]!=v for k,v in expected.items()):
                raise ValueError("retained masked call identity changed")
            inventory[str(path.relative_to(out))]=sha(path)
            atomic_json(out/"status.json",dict(stage="masked_suffix_helper",pid=os.getpid(),completed=i+1,total=44))
            print(json.dumps(dict(completed=i+1,group=row["group"],position=row["position"],text=saved["text"]),ensure_ascii=False),flush=True)
        if actor.context.embed_tokens is not None or actor.context.config.vocab_size!=0:
            raise ValueError("source helper vocabulary entered the actor")
        atomic_json(out/"generation_report.json",dict(identity=identity,native=native_identity(),
            preparation_sha256=sha(out/"preparation.json"),compiler_sha256=sha(out/"compiler.json"),
            preflight_sha256=sha(out/"preflight.pt"),raw_files_sha256=inventory,provenance=provenance,
            helper_generations=44,forced_preflight_forwards=4,optimizer_updates=0,
            elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            cache_keys_physically_deleted=False,independent_image_actor_calls=0,
            language_quality_claim=False,general_ilm_goal_complete=False))
        atomic_json(out/"status.json",dict(stage="awaiting_cpu_verification",pid=None))


def verify(out):
    cpu();mask_geometry_check();begun=time.perf_counter()
    prep=json.loads((out/"preparation.json").read_text());report=json.loads((out/"generation_report.json").read_text())
    original,_,full_verified=source();identity=committed_identity(SOURCES)
    if (prep["identity"]["sources"]!=identity["sources"] or report["identity"]["sources"]!=identity["sources"]
            or prep["native"]!=native_identity() or report["native"]!=native_identity()
            or report["preparation_sha256"]!=sha(out/"preparation.json")
            or report["preflight_sha256"]!=sha(out/"preflight.pt")):
        raise ValueError("source, native implementation or preflight changed")
    preflight=torch.load(out/"preflight.pt",map_location="cpu",weights_only=True)
    for (k,v),(pk,pv) in zip(preflight["original"],preflight["poisoned"]):
        if (not torch.equal(k[...,-52:,:],pk[...,-52:,:]) or not torch.equal(v[...,-52:,:],pv[...,-52:,:])
                or not bool((pk[...,:-52,:]==7).all()) or not bool((pv[...,:-52,:]==-11).all())):
            raise ValueError("poison changed retained suffix or failed to cover excluded keys")
    length=preflight["metadata"]["sequence_length"]
    mask=preflight["mask"]
    if (mask.shape!=(1,1,1,length+1) or not bool((mask[...,:length-52]==torch.finfo(torch.float32).min).all())
            or not bool((mask[...,length-52:]==0).all())):
        raise ValueError("preflight additive mask differs from the declared visible regions")
    states=preflight["states"]
    if not torch.equal(states["masked_original"],states["masked_poisoned"]):
        raise ValueError("excluded values affected masked native continuation")
    poison_effect=float((states["full_original"]-states["full_poisoned"]).abs().max())
    if poison_effect<=1e-5:
        raise ValueError("unmasked poison positive control is ineffective")
    baseline=torch.load(SOURCE/"outputs/copy_00_normal.pt",map_location="cpu",weights_only=True)
    replay_error=float((states["full_original"][0,0]-baseline["result"]["hidden_states"][1]).abs().max())
    if replay_error>1e-5:
        raise ValueError("original unmasked continuation did not replay")
    archive,tokenizer,markers=archive_and_tokenizer()
    controls,compiler=compile_controls(archive,tokenizer,markers)
    if (sha(out/"compiler.json")!=report["compiler_sha256"] or compiler!=json.loads((out/"compiler.json").read_text())
            or any(not torch.equal(v,torch.load(CONTROLS,map_location="cpu",weights_only=True)[k]) for k,v in controls.items())):
        raise ValueError("source compiler/control frame changed")
    eog=set(compiler["eog_ids"])
    items=[];vectors=[];ids=[];recorded=[]
    for row in original["cases"]:
        name=f"outputs/{row['group']}_{row['position']:02}.pt"
        if sha(out/name)!=report["raw_files_sha256"][name]:
            raise ValueError("masked response changed")
        value=torch.load(out/name,map_location="cpu",weights_only=True)
        source_name=f"outputs/{row['group']}_{row['position']:02}_normal.pt"
        full=torch.load(SOURCE/source_name,map_location="cpu",weights_only=True)
        tokens=value["selected_ids"]
        if (value["source_call_sha256"]!=sha(SOURCE/source_name) or value["preparation_sha256"]!=report["preparation_sha256"]
                or value["position"]!=row["position"] or value["group"]!=row["group"]
                or tokens[0]!=full["result"]["selected_ids"][0]
                or tensor_digest(value["prefill_memory"])!=row["expected_memory_sha256"]
                or not torch.equal(value["hidden_states"][0],value["prefill_memory"][0,-1].float())
                or value["hidden_states"].shape!=(len(tokens),4096) or not torch.isfinite(value["hidden_states"]).all()):
            raise ValueError("masked identity, first decision or reader memory changed")
        stopped=tokens[-1] in eog
        if stopped!=value["stopped"] or stopped==value["capped"] or not 1<=len(tokens)<=64 or (not stopped and len(tokens)!=64):
            raise ValueError("masked termination differs from source end IDs")
        text=tokenizer.decode(tokens[:-1] if stopped else tokens,skip_special_tokens=False)
        if text!=value["text"]:
            raise ValueError("masked raw text differs from retained source choices")
        g=value["cache_geometry"]
        calculated_full=calculated_suffix=0
        for layer in g["layers"]:
            b,h,n,d=layer["shape"]
            if n!=value["metadata"]["sequence_length"] or b!=1 or n<=52:
                raise ValueError("original cache dimensions differ")
            calculated_full+=2*b*h*n*d*layer["element_size"]
            calculated_suffix+=2*b*h*52*d*layer["element_size"]
        if (g["full_bytes"]!=calculated_full or g["suffix_bytes"]!=calculated_suffix
                or value["blocked_prefix"]!=value["metadata"]["sequence_length"]-52):
            raise ValueError("KV footprint or visibility boundary differs")
        partner=next(r for r in original["cases"] if r["group"]==row["group"] and r["position"]==(row["position"]^1))
        reference=row["reference"].strip();other=partner["reference"].strip();trimmed=text.strip()
        old=next(r for r in full_verified["measurements"] if r["group"]==row["group"] and r["position"]==row["position"] and r["route"]=="normal")
        items.append(dict(group=row["group"],position=row["position"],answer=text,full_answer=old["answer"],stopped=stopped,
            exact=bool(stopped and trimmed==reference),full_exact=old["exact"],
            character_accuracy=max(0.,1-distance(trimmed,reference)/max(len(trimmed),len(reference),1)),
            owning_reference_included=reference in trimmed,selective_reference_included=reference in trimmed and other not in trimmed,
            identical_source_ids=tokens==full["result"]["selected_ids"],full_kv_bytes=calculated_full,suffix_kv_bytes=calculated_suffix))
        vectors.append(value["hidden_states"]);ids.extend(tokens);recorded.extend(value["selected_logits"])
    if len(items)!=44:
        raise ValueError("entire44-case cohort required")
    source_head=archive.omitted["output.weight"]
    head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),copy=True)).to(torch.bfloat16).float()
    all_states=torch.cat(vectors);selected=torch.tensor(ids);recorded=torch.tensor(recorded)
    agreements=0;max_error=0.;max_regret=0.
    with torch.inference_mode():
        for start in range(0,len(ids),16):
            logits=F.linear(all_states[start:start+16],head);choice=selected[start:start+16]
            actual=logits.gather(1,choice[:,None]).flatten()
            agreements+=int((logits.argmax(1)==choice).sum())
            max_error=max(max_error,float((actual-recorded[start:start+16]).abs().max()))
            max_regret=max(max_regret,float((logits.max(1).values-actual).max()))
    if max_error>1e-3 or max_regret>1e-4:
        raise ValueError("temporary masked source projection fails replay")
    summaries={}
    for group in ("copy","training"):
        subset=[r for r in items if r["group"]==group]
        summaries[group]=dict(cases=len(subset),exact=sum(r["exact"] for r in subset),full_exact=sum(r["full_exact"] for r in subset),
            stopped=sum(r["stopped"] for r in subset),character_accuracy=float(np.mean([r["character_accuracy"] for r in subset])),
            owning_reference_included=sum(r["owning_reference_included"] for r in subset),
            selective_reference_included=sum(r["selective_reference_included"] for r in subset),
            identical_source_ids=sum(r["identical_source_ids"] for r in subset),
            retained_full_correct=sum(r["full_exact"] and r["exact"] for r in subset),
            lost_full_correct=sum(r["full_exact"] and not r["exact"] for r in subset),
            newly_exact=sum(not r["full_exact"] and r["exact"] for r in subset))
    result=dict(identity=identity,native=native_identity(),generation_report_sha256=sha(out/"generation_report.json"),
        source_verification_sha256=sha(SOURCE/"verification.json"),measurements=items,summaries=summaries,
        first_decisions_equal=44,normal_reader_memories_equal=44,masked_poison_states_identical=True,
        unmasked_poison_max_abs_effect=poison_effect,original_full_next_state_max_abs_error=replay_error,
        preflight_sha256=sha(out/"preflight.pt"),helper_projection_positions=len(ids),
        helper_projection_argmax_agreements=agreements,numerical_near_tie_disagreements=len(ids)-agreements,
        projection_selected_logit_error_max=max_error,projection_choice_regret_max=max_regret,
        suffix_layerwise_bytes=sorted({r["suffix_kv_bytes"] for r in items}),
        original_full_bytes_mean=float(np.mean([r["full_kv_bytes"] for r in items])),
        source_kv_physically_deleted=False,whole_context_forward_independently_replayed=False,
        final_layer_memory_sufficiency_proven=False,independent_image_actor_calls=0,optimizer_updates=0,
        offline_checks_pass=True,language_quality_claim=False,general_ilm_goal_complete=False,
        elapsed_seconds=time.perf_counter()-begun)
    atomic_json(out/"verification.json",result);atomic_json(out/"status.json",dict(stage="completed_verified",pid=None))
    print(json.dumps(dict(summaries=summaries,suffix_bytes=result["suffix_layerwise_bytes"],
        projection_positions=len(ids),argmax_agreements=agreements),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","generate","verify"),required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/suffix_kv_v78_20261106")
    args=parser.parse_args()
    {"prepare":prepare,"generate":generate,"verify":verify}[args.stage](args.out)
