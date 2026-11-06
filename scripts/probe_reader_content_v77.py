#!/usr/bin/env python3
"""Offline source-reader content probe on the exact failed ILM prompt images."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor
from ilm.visual_lm.image_context_interface import ImageContextInterface
from ilm.visual_lm.semantic_image_memory import native_budget_image
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, CONTROLS, CONTROL_SHA, sha, atomic_json, committed_identity, pixels_sha
from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer, compile_controls, embed_rows
from scripts.native_visual_baseline_v70 import TASK
from scripts.prepare_copy_transfer_v75 import CASES
from scripts.probe_corpus_fit_v75 import fixed_cases
from scripts.train_clean_corpus_v75 import setup
from scripts.verify_codec_language_v68 import independent_actor_prompt
from scripts.verify_native_visual_baseline_v70 import distance

COPY=ROOT/"data/general_instructions/copy_transfer_v75_20261106"
COPY_RUN=ROOT/"artifacts/copy_transfer_v75_20261106"
FIT=ROOT/"artifacts/corpus_fit_v75_20261106"
TEXT=ROOT/"data/visual_qa/cmrc2018_short/train.jsonl"
TEXT_SHA="742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c"
SOURCES=("scripts/probe_reader_content_v77.py","references/reader_content_v77_protocol.md",
    "scripts/check_native_visual_transfer_v71.py","scripts/verify_native_visual_baseline_v70.py",
    "scripts/native_visual_baseline_v70.py","scripts/prepare_copy_transfer_v75.py",
    "scripts/probe_corpus_fit_v75.py","scripts/train_clean_corpus_v75.py",
    "ilm/visual_lm/grounded_image_weights.py","ilm/visual_lm/grounded_image_actor.py",
    "ilm/visual_lm/gguf_visual_language.py","ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/semantic_image_memory.py","ilm/visual_lm/complete_writing_data.py",
    "ilm/visual_lm/full_text_renderer.py","ilm/visual_lm/causal_visual_prefix.py",
    "scripts/verify_codec_language_v68.py","references/evidence/clean_learning_v74_result.json")


def cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only stage required")
    torch.set_num_threads(4)


def source_rows():
    if sha(TEXT)!=TEXT_SHA or sha(CONTROLS)!=CONTROL_SHA:
        raise ValueError("source text or structural controls changed")
    copy=json.loads((COPY/"report.json").read_text())
    cv=json.loads((COPY_RUN/"verification.json").read_text())
    fv=json.loads((FIT/"verification.json").read_text())
    if (cv["report_sha256"]!=sha(COPY_RUN/"report.json") or not cv["all_12_outputs_independently_scored"]
            or fv["report_sha256"]!=sha(FIT/"report.json")
            or not fv["all_32_free_and_assisted_outputs_independently_scored"]):
        raise ValueError("completed failed actor comparisons required")
    rows=[]
    for i,(script,prompt,reference) in enumerate(CASES):
        record=copy["cases"][i]
        for key in ("input_file","source_strip_file","reference_file"):
            if sha(COPY/record[key])!=copy["files_sha256"][record[key]]:
                raise ValueError("original copy prompt changed")
        physical=torch.load(COPY/record["input_file"],map_location="cpu",weights_only=True)
        header=json.loads((COPY_RUN/"predictions"/f"{i:02}"/"header.json").read_text())
        answer_path=COPY_RUN/"predictions"/f"{i:02}"/"answer.pt"
        if sha(answer_path)!=header["output"]["raw_sha256"] or pixels_sha(physical)!=header["prompt_sha256"]:
            raise ValueError("original copy call changed")
        answer=torch.load(answer_path,map_location="cpu",weights_only=True)
        rows.append(dict(group="copy",position=i,pair=i//2,script=script,source_prompt=prompt,
            reference=reference,physical=physical,strip=np.array(Image.open(COPY/record["source_strip_file"])),
            target=np.array(Image.open(COPY/record["reference_file"])),
            expected_memory_sha256=answer["reader_memory_sha256"],
            original_actor_answer_sha256=sha(answer_path),source=record["input_file"]))
    texts={r["id"]:r for r in map(json.loads,TEXT.read_text().splitlines())}
    data=CompleteWritingImageData(TRAIN,purpose="train",expected_report_sha256=TRAIN_SHA)
    index={name:i for i,name in enumerate(data.names)}
    for i,case in enumerate(fixed_cases()):
        source=texts[Path(case["source"]).stem]
        strip=((data.prompt(index[case["source"]])[0,0]+1)*127.5).to(torch.uint8)
        physical=fold_writing_prompt(data.prompt(index[case["source"]])).pixels
        if not torch.equal(physical,independent_actor_prompt(dict(prompt=strip))):
            raise ValueError("training input fold changed")
        folder=FIT/"predictions"/f"{i:02}"
        header=json.loads((folder/"header.json").read_text())
        if pixels_sha(physical)!=header["prompt_sha256"]:
            raise ValueError("training prompt differs from the original actor call")
        for name,digest in header["files_sha256"].items():
            if sha(folder/name)!=digest:
                raise ValueError("verified training call changed")
        memory=torch.load(folder/"memory.pt",map_location="cpu",weights_only=True)
        rows.append(dict(group="training",position=i,pair=i//2,script="zh",source_prompt=source["prompt"],
            reference=source["references"][0]["text"],physical=physical,strip=strip.numpy(),
            target=((data.target(index[case["source"]])[0,0,:,:-128]+1)*127.5).to(torch.uint8).numpy(),
            expected_memory_sha256=tensor_digest(memory),original_actor_answer_sha256=sha(folder/"answer.pt"),
            source=case["source"]))
    if len(rows)!=44:
        raise ValueError("all fixed 44 cases required")
    return rows


def input_array(value):
    return ((value[0,0]+1)*127.5).round().to(torch.uint8).numpy()


def prepare(out):
    cpu()
    if out.exists():
        raise FileExistsError("choose a fresh immutable diagnostic directory")
    identity=committed_identity(SOURCES)
    rows=source_rows()
    renderer=CompleteLineRenderer()
    out.mkdir(parents=True)
    records=[]
    for row in rows:
        prompt,_=renderer.render(row["source_prompt"],maximum_blocks=240)
        target,_=renderer.render(row["reference"],maximum_blocks=128)
        if not np.array_equal(prompt,row["strip"]) or not np.array_equal(target,row["target"]):
            raise ValueError("source text is not the exact original physical content")
        name=f"{row['group']}_{row['position']:02}"
        folder=out/"inputs"/name
        folder.mkdir(parents=True)
        physical=input_array(row["physical"])
        Image.fromarray(physical).save(folder/"physical.png")
        images={}
        for route in (["normal","blank"] if row["group"]=="copy" else ["normal"]):
            original=row["physical"] if route=="normal" else torch.ones_like(row["physical"])
            transformed=input_array(native_budget_image(original))
            Image.fromarray(transformed).save(folder/(route+".png"))
            images[route]=dict(file=str((folder/(route+".png")).relative_to(out)),
                sha256=sha(folder/(route+".png")),shape=list(transformed.shape))
        records.append(dict(**{k:v for k,v in row.items() if k not in ("physical","strip","target")},
            physical_file=str((folder/"physical.png").relative_to(out)),physical_png_sha256=sha(folder/"physical.png"),
            physical_tensor_sha256=pixels_sha(row["physical"]),images=images))
    atomic_json(out/"preparation.json",dict(identity=identity,cases=records,renderer=renderer.provenance,
        structural_controls_sha256=CONTROL_SHA,source_text_sha256=TEXT_SHA,
        copy_preparation_sha256=sha(COPY/"report.json"),copy_verification_sha256=sha(COPY_RUN/"verification.json"),
        fit_verification_sha256=sha(FIT/"verification.json"),planned_generations=100,
        optimizer_updates=0,sealed_examples=0,source_text_and_rasters_match=True,
        source_helper_only=True,language_quality_claim=False,general_ilm_goal_complete=False))
    print("Prepared 44 authenticated cases and 100 fixed offline helper calls.")


def load_image(path):
    return torch.from_numpy(np.array(Image.open(path).convert("L"),copy=True)).float().div(127.5).sub(1)[None,None]


@torch.inference_mode()
def generate(out):
    identity=committed_identity(SOURCES)
    prep=json.loads((out/"preparation.json").read_text())
    if prep["identity"]["sources"]!=identity["sources"] or (out/"generation_report.json").exists():
        raise ValueError("prepared source changed or completed generations already exist")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup()
        torch.manual_seed(20267710);torch.cuda.manual_seed_all(20267710)
        begun=time.perf_counter()
        local=ROOT.parent/"LocalLLM/.local"
        actor,provenance=load_grounded_image_actor(ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        actor.writer=torch.nn.Identity()
        actor.requires_grad_(False).eval()
        if any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith(".up")):
            raise ValueError("contextual adapters must have zero outputs")
        controls=torch.load(CONTROLS,map_location="cpu",weights_only=True)
        interface=ImageContextInterface(actor,**controls).requires_grad_(False).eval()
        archive,tokenizer,markers=archive_and_tokenizer()
        compiled,compiler=compile_controls(archive,tokenizer,markers)
        if any(not torch.equal(v,controls[k]) for k,v in compiled.items()):
            raise ValueError("source compiled controls differ from the actor")
        source_head=archive.omitted["output.weight"]
        head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),copy=True)).to(device,torch.bfloat16)
        eog=set(compiler["eog_ids"])
        setup_seconds=time.perf_counter()-begun
        atomic_json(out/"compiler.json",compiler)

        def prefill(row,route):
            if route=="text":
                encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
                ids=[markers["im_start"],*encode("user\n"),*encode(TASK+"\n\n"+row["source_prompt"]),
                    markers["im_end"],*encode("\n"),markers["im_start"],*encode("assistant\n")]
                embedding=embed_rows(archive,ids).to(device)[None]
                positions=torch.arange(len(ids),device=device)[None,None].expand(4,1,-1)
                result=actor.context(inputs_embeds=embedding,position_ids=positions,use_cache=True)
                return result.last_hidden_state[:,-1:].to(torch.bfloat16),result.past_key_values,dict(
                    sequence_length=len(ids),rotary_offset=len(ids),visual_positions=0),None
            entry=row["images"][route]
            if sha(out/entry["file"])!=entry["sha256"]:
                raise ValueError("prepared helper pixels changed")
            arguments,metadata=interface.assemble(load_image(out/entry["file"]))
            result=actor.context(**arguments,use_cache=True)
            memory=result.last_hidden_state[:,-len(interface.after):].to(torch.bfloat16).cpu()
            if route=="normal" and tensor_digest(memory)!=row["expected_memory_sha256"]:
                raise ValueError("helper prefill does not match the original ILM memory")
            return result.last_hidden_state[:,-1:].to(torch.bfloat16),result.past_key_values,metadata,memory

        def decode(state,past,metadata):
            ids,states,logits_selected=[],[],[]
            offset=metadata["rotary_offset"]
            stopped=False
            for step in range(64):
                states.append(state[0,-1].float().cpu())
                logits=F.linear(state[:,-1].float(),head.float())
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite source helper projection")
                token=int(logits.argmax(-1))
                ids.append(token);logits_selected.append(float(logits[0,token]));del logits
                if token in eog:
                    stopped=True;break
                if step<63:
                    embedding=embed_rows(archive,[token]).to(device)[None]
                    positions=torch.tensor([past.get_seq_length(),offset,offset,offset],device=device).reshape(4,1,1)
                    result=actor.context(inputs_embeds=embedding,position_ids=positions,past_key_values=past,use_cache=True)
                    state,past=result.last_hidden_state[:,-1:],result.past_key_values
                    offset+=1
            return dict(selected_ids=ids,selected_logits=logits_selected,hidden_states=torch.stack(states),
                text=tokenizer.decode(ids[:-1] if stopped else ids,skip_special_tokens=False),
                stopped=stopped,capped=not stopped,metadata=metadata,helper_only_symbolic_decoding=True)

        inventory={};completed=0
        for row in prep["cases"]:
            for route in (["normal","blank","text"] if row["group"]=="copy" else ["normal","text"]):
                name=f"outputs/{row['group']}_{row['position']:02}_{route}.pt"
                path=out/name
                path.parent.mkdir(exist_ok=True)
                expected=dict(group=row["group"],position=row["position"],route=route,
                    preparation_sha256=sha(out/"preparation.json"))
                if not path.exists():
                    state,past,metadata,memory=prefill(row,route)
                    result=decode(state,past,metadata)
                    del state,past
                    partial=path.with_suffix(".partial")
                    torch.save(dict(**expected,result=result,prefill_memory=memory if route=="normal" else None),partial)
                    partial.replace(path)
                value=torch.load(path,map_location="cpu",weights_only=True)
                if {k:value[k] for k in expected}!=expected:
                    raise ValueError("retained helper call identity changed")
                inventory[name]=sha(path);completed+=1
                atomic_json(out/"status.json",dict(stage="offline_reader_probe",pid=os.getpid(),completed=completed,total=100))
                print(json.dumps(dict(completed=completed,group=row["group"],position=row["position"],route=route,
                    text=value["result"]["text"]),ensure_ascii=False),flush=True)
        if completed!=100 or actor.context.embed_tokens is not None or actor.context.config.vocab_size!=0:
            raise ValueError("helper call inventory or vocabulary-free actor boundary changed")
        atomic_json(out/"generation_report.json",dict(identity=identity,preparation_sha256=sha(out/"preparation.json"),
            compiler_sha256=sha(out/"compiler.json"),raw_files_sha256=inventory,provenance=provenance,
            elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            helper_generations=100,normal_memory_matches=44,source_helper_uses_full_kv=True,
            vocabulary_module_absent=True,source_output_head_helper_only=True,optimizer_updates=0,
            image_actor_generated_images=0,language_quality_claim=False,general_ilm_goal_complete=False))
        atomic_json(out/"status.json",dict(stage="generation_complete_awaiting_cpu_verification",pid=None))


def independent_resize(physical,blank):
    source=np.array(Image.open(physical).convert("L"),copy=True)
    if blank:
        source[:]=255
    h,w=source.shape
    nh,nw=h,w
    if h*w<1048576:
        factor=np.sqrt(np.float32(1048576)/(np.float32(h)*np.float32(w)))
        nh=int(np.ceil(np.float32(h)*factor/32))*32
        nw=int(np.ceil(np.float32(w)*factor/32))*32
    elif h*w>4194304:
        factor=np.sqrt(np.float32(h*w)/np.float32(4194304))
        nh=max(32,int(np.floor(np.float32(h)/factor/32))*32)
        nw=max(32,int(np.floor(np.float32(w)/factor/32))*32)
    return np.array(Image.fromarray(source).convert("RGB").resize((nw,nh),Image.Resampling.BILINEAR).convert("L"))


def verify(out):
    cpu();begun=time.perf_counter()
    identity=committed_identity(SOURCES)
    prep=json.loads((out/"preparation.json").read_text())
    report=json.loads((out/"generation_report.json").read_text())
    if (prep["identity"]["sources"]!=identity["sources"] or report["identity"]["sources"]!=identity["sources"]
            or sha(out/"preparation.json")!=report["preparation_sha256"]
            or sha(out/"compiler.json")!=report["compiler_sha256"]):
        raise ValueError("registered preparation or helper implementation changed")
    rows=source_rows()
    renderer=CompleteLineRenderer()
    archive,tokenizer,markers=archive_and_tokenizer()
    controls,compiler=compile_controls(archive,tokenizer,markers)
    original_controls=torch.load(CONTROLS,map_location="cpu",weights_only=True)
    if compiler!=json.loads((out/"compiler.json").read_text()) or any(not torch.equal(v,original_controls[k]) for k,v in controls.items()):
        raise ValueError("source task/control compiler no longer matches")
    eog=set(compiler["eog_ids"])
    items=[];states=[];tokens=[];recorded=[];memories=0;images=0
    for row,original in zip(prep["cases"],rows):
        for key in ("group","position","source_prompt","reference","expected_memory_sha256","original_actor_answer_sha256"):
            if row[key]!=original[key]:
                raise ValueError("fixed cohort or comparison changed")
        physical=out/row["physical_file"]
        if (sha(physical)!=row["physical_png_sha256"] or not np.array_equal(input_array(original["physical"]),np.array(Image.open(physical)))
                or pixels_sha(original["physical"])!=row["physical_tensor_sha256"]):
            raise ValueError("original physical input no longer matches")
        p,_=renderer.render(row["source_prompt"],maximum_blocks=240)
        r,_=renderer.render(row["reference"],maximum_blocks=128)
        if not np.array_equal(p,original["strip"]) or not np.array_equal(r,original["target"]):
            raise ValueError("text controls no longer identify the physical prompts/answers")
        for route in (["normal","blank","text"] if row["group"]=="copy" else ["normal","text"]):
            name=f"outputs/{row['group']}_{row['position']:02}_{route}.pt"
            if sha(out/name)!=report["raw_files_sha256"][name]:
                raise ValueError("retained helper output changed")
            saved=torch.load(out/name,map_location="cpu",weights_only=True)
            if (saved["group"]!=row["group"] or saved["position"]!=row["position"] or saved["route"]!=route
                    or saved["preparation_sha256"]!=report["preparation_sha256"]):
                raise ValueError("helper output identity changed")
            result=saved["result"];ids=result["selected_ids"]
            if (not 1<=len(ids)<=64 or result["hidden_states"].shape!=(len(ids),4096)
                    or not torch.isfinite(result["hidden_states"]).all()):
                raise ValueError("retained projection geometry differs")
            stopped=ids[-1] in eog
            if stopped!=result["stopped"] or result["capped"]==stopped or (not stopped and len(ids)!=64):
                raise ValueError("source termination differs")
            text=tokenizer.decode(ids[:-1] if stopped else ids,skip_special_tokens=False)
            if text!=result["text"]:
                raise ValueError("helper string does not decode from retained IDs")
            if route!="text":
                entry=row["images"][route]
                transformed=independent_resize(physical,route=="blank")
                if sha(out/entry["file"])!=entry["sha256"] or not np.array_equal(transformed,np.array(Image.open(out/entry["file"]))):
                    raise ValueError("declared observer pixels differ")
                h,w=transformed.shape
                metadata=result["metadata"]
                if (metadata["visual_positions"]!=(h//32)*(w//32)
                        or metadata["sequence_length"]!=len(controls["before"])+(h//32)*(w//32)+len(controls["after"])
                        or metadata["rotary_offset"]!=len(controls["before"])+max(h//32,w//32)+len(controls["after"])):
                    raise ValueError("visual contextual positions differ from the declared geometry")
                images+=1
            if route=="normal":
                memory=saved["prefill_memory"]
                if (memory.shape!=(1,52,4096) or memory.dtype!=torch.bfloat16
                        or tensor_digest(memory)!=row["expected_memory_sha256"]
                        or not torch.equal(memory[0,-1].float(),result["hidden_states"][0])):
                    raise ValueError("normal helper and original ILM memory differ")
                memories+=1
            partner=next(r for r in prep["cases"] if r["group"]==row["group"] and r["position"]==(row["position"]^1))
            reference=row["reference"].strip();other=partner["reference"].strip();trimmed=text.strip()
            piece="" if ids[0] in eog else tokenizer.decode(ids[:1],skip_special_tokens=False).strip()
            prefix=bool(piece and reference.startswith(piece))
            items.append(dict(group=row["group"],position=row["position"],route=route,answer=text,stopped=stopped,
                exact=bool(stopped and trimmed==reference),character_accuracy=max(0.,1-distance(trimmed,reference)/max(len(trimmed),len(reference),1)),
                owning_reference_included=bool(reference in trimmed),selective_reference_included=bool(reference in trimmed and other not in trimmed),
                first_piece=piece,first_piece_matches_prefix=prefix,
                selective_first_piece_prefix=bool(prefix and not other.startswith(piece))))
            states.append(result["hidden_states"]);tokens.extend(ids);recorded.extend(result["selected_logits"])
    if len(items)!=100 or memories!=44 or images!=56:
        raise ValueError("entire fixed diagnostic is required")
    source_head=archive.omitted["output.weight"]
    head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),copy=True)).to(torch.bfloat16).float()
    all_states=torch.cat(states);selected=torch.tensor(tokens);recorded=torch.tensor(recorded)
    agreements=0;max_error=0.;max_regret=0.
    with torch.inference_mode():
        for start in range(0,len(selected),16):
            logits=F.linear(all_states[start:start+16],head)
            choice=selected[start:start+16]
            actual=logits.gather(1,choice[:,None]).flatten()
            max_error=max(max_error,float((actual-recorded[start:start+16]).abs().max()))
            max_regret=max(max_regret,float((logits.max(1).values-actual).max()))
            agreements+=int((logits.argmax(1)==choice).sum())
    if max_error>1e-3 or max_regret>1e-4:
        raise ValueError("source helper output projection fails numerical replay")
    summaries={}
    for group,routes in (("copy",["normal","blank","text"]),("training",["normal","text"])):
        for route in routes:
            subset=[r for r in items if r["group"]==group and r["route"]==route]
            summaries[group+"/"+route]=dict(cases=len(subset),exact=sum(r["exact"] for r in subset),
                stopped=sum(r["stopped"] for r in subset),character_accuracy=float(np.mean([r["character_accuracy"] for r in subset])),
                owning_reference_included=sum(r["owning_reference_included"] for r in subset),
                selective_reference_included=sum(r["selective_reference_included"] for r in subset),
                first_piece_matches_prefix=sum(r["first_piece_matches_prefix"] for r in subset),
                selective_first_piece_prefix=sum(r["selective_first_piece_prefix"] for r in subset))
    value=dict(identity=identity,generation_report_sha256=sha(out/"generation_report.json"),
        preparation_sha256=sha(out/"preparation.json"),measurements=items,summaries=summaries,
        normal_memory_matches=memories,independently_reconstructed_images=images,
        helper_projection_positions=len(selected),helper_projection_argmax_agreements=agreements,
        numerical_near_tie_disagreements=len(selected)-agreements,projection_selected_logit_error_max=max_error,
        projection_choice_regret_max=max_regret,whole_vision_context_forward_replayed=False,
        source_helper_uses_full_kv=True,compact_memory_sufficiency_proven=False,
        independent_image_actor_calls=0,optimizer_updates=0,offline_checks_pass=True,
        language_quality_claim=False,general_ilm_goal_complete=False,
        elapsed_seconds=time.perf_counter()-begun)
    atomic_json(out/"verification.json",value)
    atomic_json(out/"status.json",dict(stage="completed_verified",pid=None))
    print(json.dumps(dict(summaries=summaries,normal_memory_matches=memories,
        projection_positions=len(selected),argmax_agreements=agreements),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/reader_content_v77_20261106")
    parser.add_argument("--stage",choices=("prepare","generate","verify"),required=True)
    args=parser.parse_args()
    {"prepare":prepare,"generate":generate,"verify":verify}[args.stage](args.out)


if __name__=="__main__":
    main()
