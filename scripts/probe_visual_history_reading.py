#!/usr/bin/env python3
"""Probe native visual rereading of complete answer prefixes before new training."""
import argparse
from collections import Counter, defaultdict
import fcntl
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor, MODEL_SHA
from ilm.visual_lm.image_context_interface import ImageContextInterface
from ilm.visual_lm.semantic_image_memory import native_budget_image
from scripts.cache_semantic_images_v72 import ROOT, atomic_json, sha, committed_identity
from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer, embed_rows
from scripts.train_clean_corpus_v75 import setup

SOURCE=ROOT/'artifacts/reader_content_v77_20261106'
TASK=('The upper panel contains a question or a copying request. The lower panel, '
      'labelled Answer already written, contains the answer prefix already written. '
      'Output only the missing continuation of that answer. Do not repeat the prefix. '
      'If the answer is already complete, output nothing. Do not add any explanation.')
LABEL='Answer already written:'
SEED=2026110803
SOURCES=('scripts/probe_visual_history_reading.py','references/visual_history_reading_protocol.md',
    'ilm/visual_lm/grounded_image_actor.py','ilm/visual_lm/grounded_image_weights.py',
    'ilm/visual_lm/gguf_visual_language.py','ilm/visual_lm/image_context_interface.py',
    'ilm/visual_lm/semantic_image_memory.py','ilm/visual_lm/causal_visual_prefix.py',
    'ilm/visual_lm/full_text_renderer.py','scripts/cache_semantic_images_v72.py',
    'scripts/check_native_visual_transfer_v71.py','scripts/train_clean_corpus_v75.py')


def read(path):
    return json.loads(Path(path).read_text())


def status(out,stage,**fields):
    row=dict(stage=stage,pid=os.getpid(),**fields);atomic_json(out/'status.json',row)
    print(json.dumps(row),flush=True)


def cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('explicit CPU-only stage required')
    torch.set_num_threads(4)


def pixels(path):
    return np.asarray(Image.open(path).convert('L')).copy()


def image_tensor(a):
    return torch.from_numpy(a.copy()).float().div(127.5).sub(1)[None,None]


def compile_controls(archive,tokenizer,markers):
    encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
    before=[markers['im_start'],*encode('user\n'),markers['vision_start']]
    after=[markers['vision_end'],*encode(TASK),markers['im_end'],*encode('\n'),
        markers['im_start'],*encode('assistant\n')]
    return dict(before=embed_rows(archive,before),after=embed_rows(archive,after)),dict(
        before_ids=before,after_ids=after,task=TASK,source_model_sha256=MODEL_SHA,
        eog_ids=[int(x) for x in archive.field('tokenizer.ggml.eos_token_ids')],
        tokenizer_compiler_and_readout_only=True)


def source_inventory():
    prep=read(SOURCE/'preparation.json');report=read(SOURCE/'generation_report.json')
    verified=read(SOURCE/'verification.json');public=read(ROOT/'references/evidence/reader_content_v77_result.json')
    if (sha(SOURCE/'preparation.json')!=report['preparation_sha256']
            or sha(SOURCE/'generation_report.json')!=verified['generation_report_sha256']
            or verified['generation_report_sha256']!=public['generation_report_sha256']):
        raise ValueError('completed source inventory provenance changed')
    if Counter(r['group'] for r in prep['cases'])!=Counter(copy=12,training=32):
        raise ValueError('complete 44-case inventory required')
    for row in prep['cases']:
        if sha(SOURCE/row['physical_file'])!=row['physical_png_sha256']:
            raise ValueError('original physical question changed')
    return prep


def prefix_image(renderer,text):
    if not text:return np.full((32,64),255,dtype=np.uint8)
    a,_=renderer.render(text,maximum_blocks=128)
    return ((fold_writing_prompt(image_tensor(a)).pixels[0,0]+1)*127.5).round().to(torch.uint8).numpy()


def page(question,prefix,label,*,erase=False):
    height=question.shape[0]+64+prefix.shape[0]
    width=max(question.shape[1],prefix.shape[1],label.shape[1])
    width=32*((width+31)//32)
    a=np.full((height,width),255,dtype=np.uint8);a[:question.shape[0],:question.shape[1]]=question
    y=question.shape[0];a[y+15:y+17]=0;a[y+32:y+64,:label.shape[1]]=label
    if not erase:a[y+64:,:prefix.shape[1]]=prefix
    return a


def prepare(out):
    cpu()
    if out.exists():raise FileExistsError('choose a fresh output directory')
    source=source_inventory();identity=committed_identity(SOURCES);renderer=CompleteLineRenderer()
    archive,tokenizer,markers=archive_and_tokenizer();controls,compiler=compile_controls(archive,tokenizer,markers)
    out.mkdir(parents=True);(out/'inputs').mkdir();(out/'outputs').mkdir()
    torch.save(controls,out/'controls.pt');atomic_json(out/'compiler.json',compiler)
    label,_=renderer.render(LABEL,maximum_blocks=128);Image.fromarray(label).save(out/'label.png')
    rows=[];files={'controls.pt':sha(out/'controls.pt'),'compiler.json':sha(out/'compiler.json'),'label.png':sha(out/'label.png')}
    for original in source['cases']:
        question=pixels(SOURCE/original['physical_file']);text=original['reference'];n=len(text)
        if not n:raise ValueError('original canonical answer is empty')
        positions=sorted({0,n//2,max(0,n-1),n})
        for k in positions:
            prefix=prefix_image(renderer,text[:k]);p=out/'inputs'/f'{original["group"]}_{original["position"]:02}_{k:03d}_prefix.png'
            Image.fromarray(prefix).save(p);files[str(p.relative_to(out))]=sha(p)
            middle=k==n//2 and 0<k<n
            routes=('visual','text','erased') if middle else ('visual','text')
            a=page(question,prefix,label)
            if not np.array_equal(a[:question.shape[0],:question.shape[1]],question):raise ValueError('question pixels were altered')
            for route in routes:
                index=len(rows);name=f'inputs/{index:04d}.png'
                value=page(question,prefix,label,erase=route=='erased')
                Image.fromarray(value).save(out/name);files[name]=sha(out/name)
                rows.append(dict(index=index,group=original['group'],case=original['position'],pair=original['pair'],
                    script=original['script'],position=k,characters=n,terminal=k==n,middle=middle,route=route,
                    input=name,input_sha256=sha(out/name),prefix_image=str(p.relative_to(out)),
                    source_question_file=original['physical_file'],source_question_sha256=original['physical_png_sha256'],
                    source_prompt=original['source_prompt'],prefix=text[:k],remaining=text[k:],reference=text,
                    prefix_pixel_shape=list(prefix.shape),physical_shape=list(value.shape)))
    atomic_json(out/'preparation.json',dict(identity=identity,rows=rows,files_sha256=files,renderer=renderer.provenance,
        source_preparation_sha256=sha(SOURCE/'preparation.json'),source_report_sha256=sha(SOURCE/'generation_report.json'),
        task=TASK,calls=len(rows),source_weights_duplicated=False,optimizer_updates=0,
        symbolic_readout_instrument_only=True,general_ilm_goal_complete=False))
    status(out,'prepared',calls=len(rows),routes=dict(Counter(x['route'] for x in rows)),cases=44)


def prepared(out):
    p=read(out/'preparation.json')
    for name,digest in p['identity']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('bound producer changed: '+name)
    for name,digest in p['files_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('prepared pixels or controls changed: '+name)
    source_inventory()
    return p


@torch.inference_mode()
def generate(out):
    prep=prepared(out)
    if (out/'generation.json').exists():raise FileExistsError('completed generation is immutable')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=setup()
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);started=time.monotonic()
        status(out,'loading_source',calls=len(prep['rows']))
        local=ROOT.parent/'LocalLLM/.local'
        actor,provenance=load_grounded_image_actor(ollama_root=local/'models/ollama',
            gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
        actor.writer=torch.nn.Identity();actor.requires_grad_(False).eval()
        if any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith('.up')):
            raise ValueError('source adapter outputs must be zero')
        numeric=torch.load(out/'controls.pt',map_location='cpu',weights_only=True)
        interface=ImageContextInterface(actor,**numeric).requires_grad_(False).eval()
        archive,tokenizer,markers=archive_and_tokenizer();again,compiler=compile_controls(archive,tokenizer,markers)
        if compiler!=read(out/'compiler.json') or any(not torch.equal(again[k],v) for k,v in numeric.items()):
            raise ValueError('fixed numeric controls changed')
        source=archive.omitted['output.weight']
        weights=archive.gguf.dequantize(source.data,source.tensor_type)
        head=torch.from_numpy(np.array(weights,copy=True)).to(torch.bfloat16).float().to(device)
        del weights;gc.collect()
        eog=set(compiler['eog_ids']);setup_seconds=time.monotonic()-started
        records=[];journal=out/f'calls_{time.time_ns()}.jsonl';active_start=time.monotonic()
        previous=read(out/'partial.json') if (out/'partial.json').exists() else {'outputs':[]}
        for r in previous['outputs']:
            if sha(out/r['file'])!=r['sha256']:raise ValueError('completed call changed')
            records.append(r)
        if any(r['index']!=i for i,r in enumerate(records)):raise ValueError('completed calls must form a prefix')
        encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
        with journal.open('w') as log:
            for row in prep['rows'][len(records):]:
                if time.monotonic()-started>7200:raise TimeoutError('two-hour probe bound reached; preserve partial calls')
                call_started=time.monotonic()
                if row['route']=='text':
                    content=TASK+'\n\nUpper panel:\n'+row['source_prompt']+'\n\nAnswer already written:\n'+row['prefix']
                    ids=[markers['im_start'],*encode('user\n'+content),markers['im_end'],*encode('\n'),
                        markers['im_start'],*encode('assistant\n')]
                    embedding=embed_rows(archive,ids).to(device)[None]
                    position_ids=torch.arange(len(ids),device=device)[None,None].expand(4,1,-1)
                    output=actor.context(inputs_embeds=embedding,position_ids=position_ids,use_cache=True)
                    meta=dict(sequence_length=len(ids),rotary_offset=len(ids),visual_positions=0)
                    memory=output.last_hidden_state[:,-1:].to(torch.bfloat16).cpu();del embedding
                else:
                    a=pixels(out/row['input']);arguments,meta=interface.assemble(native_budget_image(image_tensor(a)))
                    output=actor.context(**arguments,use_cache=True)
                    memory=output.last_hidden_state[:,-len(interface.after):].to(torch.bfloat16).cpu();del arguments
                state=output.last_hidden_state[:,-1:].to(torch.bfloat16);past=output.past_key_values;del output
                offset=meta['rotary_offset'];states=[];selected=[];scores=[];stopped=False
                for step in range(64):
                    states.append(state[0,-1].float().cpu())
                    logits=F.linear(state[:,-1].float(),head)
                    if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite source projection')
                    piece=int(logits.argmax(-1));selected.append(piece);scores.append(float(logits[0,piece]));del logits
                    if piece in eog:stopped=True;break
                    if step<63:
                        embedding=embed_rows(archive,[piece]).to(device)[None]
                        positions=torch.tensor([past.get_seq_length(),offset,offset,offset],device=device).reshape(4,1,1)
                        output=actor.context(inputs_embeds=embedding,position_ids=positions,past_key_values=past,use_cache=True)
                        state,past=output.last_hidden_state[:,-1:],output.past_key_values;offset+=1
                        del output,embedding
                text=tokenizer.decode(selected[:-1] if stopped else selected,skip_special_tokens=False)
                name=f'outputs/{row["index"]:04d}.pt';path=out/name
                value=dict(index=row['index'],input_sha256=row['input_sha256'],route=row['route'],metadata=meta,
                    memory=memory,hidden_states=torch.stack(states),selected_ids=selected,selected_logits=scores,
                    text=text,stopped=stopped,capped=not stopped,elapsed_seconds=time.monotonic()-call_started,
                    source_helper_only=True,reference_used_as_history=row['position']>0)
                temporary=path.with_suffix('.partial');torch.save(value,temporary)
                if path.exists():raise FileExistsError('unindexed call exists; preserve and audit it before any restart')
                temporary.replace(path);record=dict(index=row['index'],file=name,sha256=sha(path))
                records.append(record);log.write(json.dumps(record)+'\n');log.flush()
                atomic_json(out/'partial.json',dict(outputs=records,complete=len(records)==len(prep['rows'])))
                del past,state,states,memory
                if torch.cuda.max_memory_allocated()>22*1024**3:raise MemoryError('declared allocated-memory bound exceeded')
                if len(records)%16==0 or len(records)==len(prep['rows']):
                    status(out,'reading',completed=len(records),total=len(prep['rows']),route=row['route'])
        atomic_json(out/'generation.json',dict(preparation_sha256=sha(out/'preparation.json'),outputs=records,
            source=provenance,setup_seconds=setup_seconds,current_inference_seconds=time.monotonic()-active_start,
            current_total_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            optimizer_updates=0,source_helper_only=True,image_answer_generated=False,general_ilm_goal_complete=False))
        status(out,'generation_complete',calls=len(records))


def metrics(rows):
    result={}
    for group in ('copy','training'):
        for route in ('visual','text','erased'):
            chosen=[r for r in rows if r['group']==group and r['route']==route]
            nonempty=[r for r in chosen if r['expected']];terminal=[r for r in chosen if r['terminal']]
            result[group+'/'+route]=dict(calls=len(chosen),suffix_exact=sum(r['exact'] for r in chosen),
                stopped_suffix_exact=sum(r['exact'] and r['stopped'] for r in chosen),nonempty_suffixes=len(nonempty),
                first_codepoint_correct=sum(r['first_codepoint_correct'] for r in nonempty),
                first_codepoint_accuracy=sum(r['first_codepoint_correct'] for r in nonempty)/max(1,len(nonempty)),
                terminals=len(terminal),empty_stopped_terminals=sum(r['exact'] and r['stopped'] for r in terminal))
    middle={}
    for group in ('copy','training'):
        normal={(r['case'],r['position']):r for r in rows if r['group']==group and r['route']=='visual' and r['middle']}
        erased={(r['case'],r['position']):r for r in rows if r['group']==group and r['route']=='erased'}
        if normal.keys()!=erased.keys():raise ValueError('all middle erasure pairs required')
        middle[group]=dict(pairs=len(normal),normal_first_correct=sum(r['first_codepoint_correct'] for r in normal.values()),
            erased_first_correct=sum(r['first_codepoint_correct'] for r in erased.values()),
            first_codepoint_gain=sum(normal[k]['first_codepoint_correct']-erased[k]['first_codepoint_correct'] for k in normal)/max(1,len(normal)),
            response_changed=sum(normal[k]['response']!=erased[k]['response'] for k in normal))
    return result,middle


def verify(out):
    cpu();prep=prepared(out);generation=read(out/'generation.json')
    if generation['preparation_sha256']!=sha(out/'preparation.json') or len(generation['outputs'])!=len(prep['rows']):
        raise ValueError('complete generation for prepared inventory required')
    renderer=CompleteLineRenderer();label=pixels(out/'label.png')
    archive,tokenizer,markers=archive_and_tokenizer();controls,compiler=compile_controls(archive,tokenizer,markers)
    saved=torch.load(out/'controls.pt',map_location='cpu',weights_only=True)
    if compiler!=read(out/'compiler.json') or any(not torch.equal(v,controls[k]) for k,v in saved.items()):
        raise ValueError('source controls do not recompile')
    source=archive.omitted['output.weight']
    head=torch.from_numpy(np.array(archive.gguf.dequantize(source.data,source.tensor_type),copy=True)).to(torch.bfloat16).float()
    eog=set(compiler['eog_ids']);results=[];decision_count=0
    for row,record in zip(prep['rows'],generation['outputs']):
        if row['index']!=record['index'] or sha(out/record['file'])!=record['sha256']:raise ValueError('raw call identity changed')
        prefix=prefix_image(renderer,row['prefix']);question=pixels(SOURCE/row['source_question_file'])
        if (not np.array_equal(prefix,pixels(out/row['prefix_image']))
                or not np.array_equal(page(question,prefix,label,erase=row['route']=='erased'),pixels(out/row['input']))):
            raise ValueError('page or prefix pixels do not reconstruct')
        v=torch.load(out/record['file'],map_location='cpu',weights_only=True)
        if v['index']!=row['index'] or v['input_sha256']!=row['input_sha256']:raise ValueError('call input changed')
        if v['stopped']!=(v['selected_ids'][-1] in eog):raise ValueError('native EOG event changed')
        ids=v['selected_ids'][:-1] if v['stopped'] else v['selected_ids']
        if tokenizer.decode(ids,skip_special_tokens=False)!=v['text']:raise ValueError('native text decoding changed')
        h=v['hidden_states'];selected=torch.tensor(v['selected_ids'])
        if h.shape!=(len(selected),4096) or not torch.isfinite(h).all():raise ValueError('complete finite pre-decision states required')
        for start in range(0,len(h),16):
            logits=F.linear(h[start:start+16].float(),head)
            if not torch.equal(logits.argmax(-1),selected[start:start+16]):raise ValueError('retained source decision does not replay')
            decision_count+=len(logits);del logits
        response=v['text'].strip();expected=row['remaining'].strip()
        results.append(dict(**{k:row[k] for k in ('index','group','case','pair','script','position','characters','terminal','middle','route','input')},
            prefix=row['prefix'],expected=expected,response=response,raw_response=v['text'],stopped=v['stopped'],
            exact=response==expected,first_codepoint_correct=bool(expected and response and expected[0]==response[0]),
            raw_file=record['file'],raw_sha256=record['sha256'],input_sha256=row['input_sha256']))
    groups,middle=metrics(results)
    gates=dict(copy_first=groups['copy/visual']['first_codepoint_accuracy']>=.9,
        qa_first=groups['training/visual']['first_codepoint_accuracy']>=.8,
        copy_terminal=groups['copy/visual']['empty_stopped_terminals']/12>=.8,
        qa_terminal=groups['training/visual']['empty_stopped_terminals']/32>=.8,
        copy_prefix_gain=middle['copy']['first_codepoint_gain']>=.15,
        qa_prefix_gain=middle['training']['first_codepoint_gain']>=.15)
    atomic_json(out/'verification.json',dict(preparation_sha256=sha(out/'preparation.json'),generation_sha256=sha(out/'generation.json'),
        groups=groups,middle_prefix=middle,gates=gates,writer_trial_admitted=all(gates.values()),rows=results,
        source_decisions_replayed=decision_count,page_pixels_reconstructed=True,numeric_controls_recompiled=True,
        full_vision_body_replayed=False,teacher_assisted_history=True,optimizer_updates=0,
        independent_ILM_qualified=False,general_ilm_goal_complete=False))
    status(out,'verified',groups=groups,middle=middle,gates=gates,writer_trial_admitted=all(gates.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','generate','verify'),required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/visual_history_reading_v1');args=p.parse_args()
    {'prepare':prepare,'generate':generate,'verify':verify}[args.stage](args.output.resolve())
