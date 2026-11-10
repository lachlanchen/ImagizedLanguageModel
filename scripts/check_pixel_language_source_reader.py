#!/usr/bin/env python3
"""Helper-only source reading baseline on the current Chinese image cohort.

This diagnostic does not generate ILM images or train on development cases.
The temporary source vocabulary stays outside the image actor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import (
    diagnostic_gpu, host_resources, reserve_check, owner_gone, require)

TRAINING = ROOT/'artifacts/pixel_language_diversity_training_v2'
DIAGNOSTIC = ROOT/'artifacts/pixel_first_block_likelihood_final_v3'


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def sources():
    previous = load(DIAGNOSTIC/'preparation.json')
    for name, digest in previous['code_sha256'].items():
        require(sha(ROOT/name) == digest, 'completed diagnostic source changed: '+name)
    names = set(previous['code_sha256']) | {
        rel(__file__), 'scripts/check_native_visual_transfer_v71.py',
        'scripts/native_visual_baseline_v70.py',
        'references/pixel_language_source_reader_protocol.md'}
    result = {name: sha(ROOT/name) for name in names}
    for name, digest in result.items():
        require(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)).hexdigest() == digest,
                'commit the reader diagnostic before preparation: '+name)
    return result


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for preparation')
    require(not output.exists(), 'preserve existing and partial reader diagnostics')
    code = sources()
    main = load(TRAINING/'preparation.json')
    corpus = ROOT/main['corpus']
    require(sha(corpus/'manifest.json') == main['corpus_sha256']['manifest.json'], 'owning corpus changed')
    manifest = load(corpus/'manifest.json')
    diagnostic = load(DIAGNOSTIC/'preparation.json')
    source = ROOT/'data/visual_qa/cmrc2018_short/development.jsonl'
    records = {r['id']: r for r in map(json.loads, source.read_text().splitlines())}
    controls = next(ROOT/n for n in main['bound_sha256'] if Path(n).name == 'structural_vectors.pt')
    compiler = controls.parent/'compiler.json'
    require(sha(controls) == main['bound_sha256'][rel(controls)], 'owning numeric task vectors changed')
    bindings = {rel(TRAINING/'preparation.json'): sha(TRAINING/'preparation.json'),
        rel(corpus/'manifest.json'): sha(corpus/'manifest.json'), rel(source): sha(source),
        rel(DIAGNOSTIC/'preparation.json'): sha(DIAGNOSTIC/'preparation.json'),
        rel(controls): sha(controls), rel(compiler): sha(compiler)}
    jobs = []
    for i, row in enumerate(manifest['development'][:16]):
        require(row['source'] == 'cmrc', 'fixed Chinese cohort required')
        item = records[row['id']]
        images, fingerprints = {}, {}
        for route in ('normal', 'blank'):
            path = corpus/row[route]
            require(sha(path) == row[route+'_sha256'], 'original PNG changed')
            bindings[rel(path)] = row[route+'_sha256']
            images[route] = rel(path)
            old = next(v for v in diagnostic['jobs'] if v['key'] == f'development_{i:04}_{route}')
            fingerprints[route] = old['saved']['0']['reader_memory_sha256']
        jobs.append(dict(index=i, id=row['id'], pair=row['pair'], images=images,
            expected_reader_memory=fingerprints, source_prompt=item['prompt'],
            references=[r['text'] for r in item['references']]))
    output.mkdir(parents=True)
    save_json(output/'preparation.json', dict(source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), code_sha256=code,
        bindings_sha256=bindings, controls=rel(controls), compiler=rel(compiler),
        jobs=jobs, routes=['normal','blank','text'], maximum_tokens=64,
        maximum_allocated_vram_gib=22., maximum_resource_wait_seconds=300., model_calls=0, optimizer_updates=0,
        helper_only_symbolic_decoding=True, ILM_image_answers_generated=0,
        development_sealed=False, development_training_allowed=False))
    print(json.dumps(dict(prepared=True,cases=len(jobs),routes=3,model_calls=0)))


def run(output):
    prepared = load(output/'preparation.json')
    owner_gone(TRAINING)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'original deterministic GPU0 environment required')
    require(sources() == prepared['code_sha256'], 'prepared source changed')
    for name, digest in prepared['bindings_sha256'].items():
        require(sha(ROOT/name) == digest, 'prepared input changed: '+name)
    execution = output/'execution'
    require(not execution.exists(), 'preserve earlier reader executions')
    execution.mkdir()
    began = time.perf_counter()
    identity = dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    def status(phase, **extra):
        save_json(execution/'status.json',dict(phase=phase,**identity,seconds=time.perf_counter()-began,**extra))
    try:
        import torch
        from torch.nn import functional as F
        from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer,compile_controls,embed_rows
        from scripts.native_visual_baseline_v70 import TASK
        from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor
        from ilm.visual_lm.image_context_interface import ImageContextInterface
        from ilm.visual_lm.semantic_image_memory import native_budget_image
        from ilm.visual_lm.clean_image_actor import tensor_digest
        with diagnostic_gpu() as (device, admission), torch.no_grad():
            save_json(execution/'admission.json',admission)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            status('loading_source_reader')
            local=ROOT.parent/'LocalLLM/.local'
            actor,provenance=load_grounded_image_actor(ollama_root=local/'models/ollama',
                gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
            actor.writer=torch.nn.Identity()
            actor.requires_grad_(False).eval()
            require(not any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith('.up')),
                    'original source adapters must have zero output')
            controls=torch.load(ROOT/prepared['controls'],map_location='cpu',weights_only=True)
            interface=ImageContextInterface(actor,**controls).requires_grad_(False).eval()
            archive,tokenizer,markers=archive_and_tokenizer()
            compiled,compiler=compile_controls(archive,tokenizer,markers)
            require(compiler==load(ROOT/prepared['compiler']) and
                    all(torch.equal(compiled[k],v) for k,v in controls.items()), 'numeric task controls changed')
            source_head=archive.omitted['output.weight']
            head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),
                copy=True)).to(device,torch.bfloat16)
            eog=set(compiler['eog_ids'])
            measurements=[]
            resources=[]
            resource_wait_seconds=0.
            runtime_position={}
            def guard():
                nonlocal resource_wait_seconds
                require(torch.cuda.max_memory_allocated()/1024**3 <= prepared['maximum_allocated_vram_gib'],
                        'reader diagnostic exceeds22GiB allocation')
                while True:
                    v=host_resources()
                    resources.append(v)
                    require(v['available_gib']>=24,'reader runtime RAM reserve reached')
                    pressure_ok=v['some_pressure_avg10']<=.5 and v['full_pressure_avg10']<=.1
                    if admission['mode']!='post_terminal_RAM_reserve_and_PSI' or pressure_ok:
                        break
                    remaining=prepared['maximum_resource_wait_seconds']-resource_wait_seconds
                    require(remaining>0,'reader total memory-pressure wait budget exhausted')
                    status('waiting_for_memory_pressure',completed=len(measurements),total=48,
                        resource_wait_seconds=resource_wait_seconds,resources=v,**runtime_position)
                    waited=time.perf_counter()
                    time.sleep(min(10.,remaining))
                    resource_wait_seconds+=time.perf_counter()-waited
            for job in prepared['jobs']:
                for route in prepared['routes']:
                    runtime_position=dict(index=job['index'],route=route,decoded_steps=0)
                    guard()
                    if route=='text':
                        encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
                        ids=[markers['im_start'],*encode('user\n'),*encode(TASK+'\n\n'+job['source_prompt']),
                            markers['im_end'],*encode('\n'),markers['im_start'],*encode('assistant\n')]
                        embedding=embed_rows(archive,ids).to(device)[None]
                        positions=torch.arange(len(ids),device=device)[None,None].expand(4,1,-1)
                        response=actor.context(inputs_embeds=embedding,position_ids=positions,use_cache=True)
                        metadata=dict(sequence_length=len(ids),rotary_offset=len(ids),visual_positions=0)
                    else:
                        array=np.asarray(Image.open(ROOT/job['images'][route]).convert('L')).copy()
                        prompt=torch.from_numpy(array).float()[None,None]/127.5-1
                        arguments,metadata=interface.assemble(native_budget_image(prompt))
                        response=actor.context(**arguments,use_cache=True)
                        fingerprint=tensor_digest(response.last_hidden_state[:,-len(interface.after):].to(torch.bfloat16))
                        require(fingerprint==job['expected_reader_memory'][route],
                                'source reader fingerprint differs from actual ILM image path')
                        metadata=dict(metadata,reader_memory_sha256=fingerprint)
                        del arguments,prompt
                    state,past=response.last_hidden_state[:,-1:],response.past_key_values
                    del response
                    ids,states,selected=[],[],[]
                    offset=metadata['rotary_offset']
                    stopped=False
                    for step in range(prepared['maximum_tokens']):
                        states.append(state[0,-1].float().cpu())
                        logits=F.linear(state[:,-1].float(),head.float())
                        require(bool(torch.isfinite(logits).all()),'nonfinite source projection')
                        token=int(logits.argmax(-1)); ids.append(token); selected.append(float(logits[0,token]))
                        del logits
                        runtime_position['decoded_steps']=step+1
                        guard()
                        if token in eog:
                            stopped=True
                            break
                        if step+1<prepared['maximum_tokens']:
                            embedding=embed_rows(archive,[token]).to(device)[None]
                            positions=torch.tensor([past.get_seq_length(),offset,offset,offset],device=device).reshape(4,1,1)
                            response=actor.context(inputs_embeds=embedding,position_ids=positions,past_key_values=past,use_cache=True)
                            state,past=response.last_hidden_state[:,-1:],response.past_key_values
                            del response
                            offset+=1
                    answer=tokenizer.decode(ids[:-1] if stopped else ids,skip_special_tokens=False)
                    value=dict(index=job['index'],route=route,selected_ids=ids,selected_logits=selected,
                        hidden_states=torch.stack(states),text=answer,stopped=stopped,capped=not stopped,
                        metadata=metadata,helper_only_symbolic_decoding=True)
                    name=f"case_{job['index']:04}_{route}.pt"
                    torch.save(value,execution/name)
                    references=[s.strip() for s in job['references']]
                    measurements.append(dict(index=job['index'],id=job['id'],pair=job['pair'],route=route,
                        text=answer,stopped=stopped,capped=not stopped,
                        literal_exact=bool(stopped and answer.strip() in references),
                        reference_included=any(s and s in answer for s in references),file=name,sha256=sha(execution/name)))
                    del state,past
                    status('reading_source',completed=len(measurements),total=48)
                    print(json.dumps(dict(index=job['index'],route=route,text=answer),ensure_ascii=False),flush=True)
            require(actor.context.embed_tokens is None,'source vocabulary entered image interface')
            summaries={route:dict(cases=16,literal_exact=sum(v['literal_exact'] for v in measurements if v['route']==route),
                stopped=sum(v['stopped'] for v in measurements if v['route']==route)) for route in prepared['routes']}
            save_json(execution/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),
                provenance=provenance,measurements=measurements,summaries=summaries,
                all32_image_reader_fingerprints_match_ILM=True,helper_only_symbolic_decoding=True,
                source_output_head_registered_on_image_actor=False,ILM_image_answers_generated=0,
                optimizer_updates=0,development_training_allowed=False,formal_semantic_accuracy_measured=False,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                elapsed_seconds=time.perf_counter()-began,resource_admission=admission,
                resource_wait_seconds=resource_wait_seconds,
                minimum_available_gib=min(v['available_gib'] for v in resources),
                maximum_memory_pressure_some_avg10=max(v['some_pressure_avg10'] for v in resources),
                maximum_memory_pressure_full_avg10=max(v['full_pressure_avg10'] for v in resources)))
            status('completed')
    except BaseException as error:
        status('failed',reason=repr(error))
        raise
    finally:
        if (execution/'status.json').exists():
            save_json(execution/'status.json',dict(load(execution/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.stage](args.output.resolve())
