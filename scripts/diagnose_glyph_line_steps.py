#!/usr/bin/env python3
"""Source-head-only test of character-sized framing; no learned image answer."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/diagnose_glyph_line_steps.py'
PROTOCOL='references/glyph_line_step_probe_protocol.md'
TASK=('\nRead the question or instruction in the supplied material. Answer it using any accompanying passage. '
      'Return only the requested answer. Write exactly one character on each line, with a newline after '
      'every character including the last character. Do not add numbering, spaces between characters, '
      'explanations, or any other text. If there is no question or instruction, use an empty answer.')
ROUTES=['native_units','exact_character_plus_LF','canonical_image_plus_LF']
PARENT=ROOT/'artifacts/glyph_feedback_ceiling_v1'


def load(path):return json.loads(Path(path).read_text())
def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def identity():
    names=[OWN,PROTOCOL,'scripts/check_native_visual_transfer_v71.py',
        'scripts/train_semantic_glyph_language.py','ilm/visual_lm/semantic_glyph_loop.py',
        'ilm/visual_lm/image_context_interface.py','ilm/visual_lm/glyph_runtime_resources.py',
        'ilm/visual_lm/glyph_reader_monitor.py','ilm/visual_lm/grounded_image_weights.py',
        'ilm/visual_lm/grounded_image_actor.py','ilm/visual_lm/gguf_visual_language.py']
    for name in names:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'commit line-step diagnostic sources first')
    return {n:sha(ROOT/n) for n in names}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU compilation requires hidden CUDA')
    require(not output.exists(),'preserve previous source probes')
    code=identity()
    require((ROOT/'artifacts/balanced_glyph_writer_v1/raw_review/scores.json').exists()
        and (ROOT/'artifacts/balanced_glyph_parent_v1/raw_review/scores.json').exists(),
        'finish the unchanged balanced-writer comparison first')
    import torch
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer,embed_rows
    torch.set_num_threads(2);archive,tokenizer,markers=archive_and_tokenizer()
    encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
    lf=encode('\n');require(len(lf)==1 and tokenizer.decode(lf)=='\n','one fixed newline unit required')
    before=[markers['im_start'],*encode('user\n'),markers['vision_start']]
    after=[markers['vision_end'],*encode(TASK),markers['im_end'],*encode('\n'),markers['im_start'],*encode('assistant\n')]
    require(tokenizer.decode(before,skip_special_tokens=False)=='<|im_start|>user\n<|vision_start|>'
        and tokenizer.decode(after,skip_special_tokens=False)=='<|vision_end|>'+TASK+'<|im_end|>\n<|im_start|>assistant\n',
        'fixed source controls must round-trip')
    parent=load(PARENT/'preparation.json');primitive=ROOT/parent['primitive']
    inventory=load(primitive/'inventory.json')['records'];exceptions=[];tested=0
    for row in inventory:
        for char,index in zip(row['chars'],row['source_ids']):
            require(tokenizer.decode([index])==char,'primitive single-form source ID changed')
            tested+=1;ids=encode(char+'\n')
            if ids!=[index,*lf]:exceptions.append(dict(char=char,source_id=index,combined_ids=ids))
    output.mkdir(parents=True)
    torch.save(dict(before=embed_rows(archive,before),after=embed_rows(archive,after),
        line_advance=embed_rows(archive,lf)),output/'structural_vectors.pt')
    bindings={rel(PARENT/'preparation.json'):sha(PARENT/'preparation.json')}
    for name in ['training.pt','inventory.json']:
        path=primitive/name;require(sha(path)==parent['bindings_sha256'][rel(path)],'parent physical primitive changed')
        bindings[rel(path)]=sha(path)
    checkpoint=ROOT/parent['checkpoint']
    require(sha(checkpoint)==parent['bindings_sha256'][rel(checkpoint)],'parent encoder checkpoint changed')
    bindings[rel(checkpoint)]=sha(checkpoint)
    jobs=[{k:r[k] for k in ['split','index','prompt']} for r in parent['jobs']]
    for row in jobs:
        digest=sha(ROOT/row['prompt'])
        require(digest==parent['bindings_sha256'][row['prompt']],'parent input pixels changed')
        bindings[row['prompt']]=digest
    for path in [ROOT/'artifacts/pixel_language_diversity_v1/manifest.json',
                 ROOT/'artifacts/semantic_glyph_language_v1/preparation.json',
                 *(ROOT/f'data/visual_qa/cmrc2018_short/{split}.jsonl' for split in ['train','development'])]:
        bindings[rel(path)]=sha(path)
    require(len(jobs)==24,'fixed earlier image cohort required')
    save_json(output/'compiler.json',dict(task=TASK,before_ids=before,after_ids=after,newline_ids=lf,
        eog_ids=list(map(int,archive.field('tokenizer.ggml.eos_token_ids'))),
        primitive_forms_tested=tested,two_unit_codec_exceptions=exceptions,compiler_only_symbolic_data=True))
    save_json(output/'preparation.json',dict(code_sha256=code,bindings_sha256=bindings,jobs=jobs,
        checkpoint=rel(checkpoint),primitive=rel(primitive),maximum_steps=64,routes=ROUTES,
        controls_sha256=sha(output/'structural_vectors.pt'),compiler_sha256=sha(output/'compiler.json'),
        maximum_seconds=2400.,helper_only_symbolic_decoding=True,optimizer_updates=0,
        training_data_reuse_allowed=False,learned_ILM_images_generated=0,
        reserved_question_cohorts_used=False,references_read=False))
    print(json.dumps(dict(prepared=True,forms_tested=tested,codec_exceptions=len(exceptions),neural_calls=0)))


def verify(output):
    m=load(output/'preparation.json');require(identity()==m['code_sha256'],'line-step source changed')
    for name,digest in m['bindings_sha256'].items():require(sha(ROOT/name)==digest,'fixed diagnostic input changed')
    require(sha(output/'structural_vectors.pt')==m['controls_sha256']
        and sha(output/'compiler.json')==m['compiler_sha256'],'fixed source framing changed')
    return m


def run(output):
    m=verify(output)
    import torch
    from torch.nn import functional as F
    from scripts.train_semantic_glyph_language import reader,bridge_model,numerics,image,atomic_tensor
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer,embed_rows
    from ilm.visual_lm.image_context_interface import ImageContextInterface
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    folder=output/'execution';folder.mkdir(exist_ok=False)
    monitor=GlyphReaderMonitor(folder,m['maximum_seconds']);monitor.status('admitting')
    try:
        controls=torch.load(output/'structural_vectors.pt',weights_only=True,map_location='cpu')
        compiler=load(output/'compiler.json');eog=set(compiler['eog_ids']);records=[]
        monitor.guard()
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard()
            original,provenance=reader(device)
            interface=ImageContextInterface(original.actor,before=controls['before'],after=controls['after'])
            del original
            bridge=bridge_model({},device,ROOT/m['checkpoint'])
            archive,tokenizer,_=archive_and_tokenizer();monitor.guard()
            source=archive.omitted['output.weight']
            array=np.array(archive.gguf.dequantize(source.data,source.tensor_type),dtype=np.float32,copy=True)
            head=torch.from_numpy(array).to(device,torch.bfloat16).float();del array
            inventory=load(ROOT/m['primitive']/'inventory.json')['records']
            forms={c:(i,index) for i,row in enumerate(inventory) for c,index in zip(row['chars'],row['source_ids'])}
            data=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')
            physical=(1-2*data['ink'].float()).to(device)
            encoded=torch.cat([bridge.feedback_embedding(physical[i:i+128]) for i in range(0,len(physical),128)])
            lf=controls['line_advance'].to(device)
            hook=interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            for job in m['jobs']:
                first_hash=None
                for route in ROUTES:
                    monitor.position=dict(split=job['split'],index=job['index'],route=route,completed=len(records),total=72,step=0)
                    args,geometry=interface.assemble(native_budget_image(image(ROOT/job['prompt'])))
                    value=interface.actor.context(**args,use_cache=True)
                    h,past=value.last_hidden_state[:,-1:],value.past_key_values;del value,args
                    digest=tensor_hash(h)
                    if first_hash is None:first_hash=digest
                    require(digest==first_hash,'within-case image prefills differ across feedback routes')
                    offset=geometry['rotary_offset'];consumed=[];emitted=[];decisions=[];states=[];feedback=[];reason='cap'
                    for step in range(m['maximum_steps']+1):
                        monitor.position['step']=step;monitor.guard();states.append(h[0,-1].float().cpu())
                        winner=int(F.linear(h[:,-1].float(),head).argmax(-1));text=tokenizer.decode([winner],skip_special_tokens=False)
                        decisions.append(dict(id=winner,source_unit=text))
                        if winner in eog:reason='native_terminal';break
                        if step==m['maximum_steps']:break
                        if route=='native_units':
                            embedded=embed_rows(archive,[winner]).to(device);consumed.append(winner)
                        else:
                            if not text or '\ufffd' in text or text[0] not in forms:
                                reason='unsupported_first_complete_form';break
                            char=text[0];label,index=forms[char];emitted.append(char)
                            charvec=embed_rows(archive,[index]).to(device) if route=='exact_character_plus_LF' else encoded[label:label+1]
                            embedded=torch.cat([charvec,lf])
                            feedback.append(dict(char=char,image_sha256=tensor_hash(physical[label:label+1]),
                                character_embedding_sha256=tensor_hash(charvec),newline_embedding_sha256=tensor_hash(lf)))
                        n=len(embedded);seq=torch.arange(past.get_seq_length(),past.get_seq_length()+n,device=device)
                        rotary=torch.arange(offset,offset+n,device=device)
                        positions=torch.stack([seq,rotary,rotary,rotary])[:,None]
                        value=interface.actor.context(inputs_embeds=embedded[None],position_ids=positions,past_key_values=past,use_cache=True)
                        h,past=value.last_hidden_state[:,-1:],value.past_key_values;del value,embedded;offset+=n
                    full=tokenizer.decode(consumed,skip_special_tokens=False) if route=='native_units' else ''.join(c+'\n' for c in emitted)
                    cleaned=full.replace('\n','')
                    native_format=(full.endswith('\n') and all(len(s)==1 for s in full.split('\n')[:-1])) if full else True
                    stem=f"{job['split']}_{job['index']:04}_{route}"
                    raw=dict(decisions=decisions,states=torch.stack(states),feedback=feedback,emitted_forms=emitted,
                        source_text=full,LF_removed_text=cleaned,reason=reason,observer_geometry=geometry)
                    atomic_tensor(folder/(stem+'.pt'),raw)
                    records.append(dict(**job,route=route,text=cleaned,source_text=full,reason=reason,
                        emitted_steps=len(consumed) if route=='native_units' else len(emitted),
                        native_single_codepoint_lines_with_final_LF=native_format,
                        selected_nonterminal_multicodepoint_units=sum(len(d['source_unit'])>1 for d in decisions if d['id'] not in eog),
                        prefill_sha256=digest,tensor=stem+'.pt',tensor_sha256=sha(folder/(stem+'.pt'))))
                    monitor.status('source_only_line_clock_diagnostic');print(json.dumps(records[-1],ensure_ascii=False),flush=True)
                    del h,past,states
            hook.remove();require(interface.actor.context.embed_tokens is None,'lexical table entered contextual actor')
            save_json(folder/'report.json',dict(records=records,preparation_sha256=sha(output/'preparation.json'),
                provenance=provenance,all_within_case_initial_states_bit_identical=True,
                helper_only_symbolic_decoding=True,learned_ILM_images_generated=0,optimizer_updates=0,
                source_head_attached_to_actor=False,training_data_reuse_allowed=False,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU reference scoring')
    verify(output);require(load(output/'execution/status.json')['phase']=='completed','finish all diagnostic routes')
    rows=load(output/'execution/report.json')['records'];require(len(rows)==72,'retain all24 cases and three routes')
    corpus=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    controls=load(ROOT/'artifacts/semantic_glyph_language_v1/preparation.json')['jobs'][:8]
    source={split:{r['id']:r for r in map(json.loads,(ROOT/f'data/visual_qa/cmrc2018_short/{split}.jsonl').read_text().splitlines())}
            for split in ['train','development']}
    scored=[]
    for row in rows:
        index=row['index'];split=row['split'];key=Path(controls[index]['source']).stem if split=='train' else corpus['development'][index]['id']
        references=[v['text'] for v in source[split][key]['references']]
        scored.append(dict(**row,references=references,literal_reference_exact=row['reason']=='native_terminal' and row['text'] in references))
    groups={}
    for split in ['train','development']:
        for route in ROUTES:
            part=[r for r in scored if r['split']==split and r['route']==route]
            groups[split+'/'+route]=dict(cases=len(part),literal_reference_exact=sum(r['literal_reference_exact'] for r in part),
                native_terminal=sum(r['reason']=='native_terminal' for r in part),
                unsupported=sum(r['reason']=='unsupported_first_complete_form' for r in part),capped=sum(r['reason']=='cap' for r in part),
                single_codepoint_line_format=sum(r['native_single_codepoint_lines_with_final_LF'] for r in part))
    native=groups['development/native_units']['literal_reference_exact']
    exact=groups['development/exact_character_plus_LF']['literal_reference_exact']
    save_json(output/'scores.json',dict(records=scored,groups=groups,
        prespecified_mechanical_alignment_signal=native>=10 and exact>=max(10,native-1),
        helper_only_symbolic_decoding=True,learned_ILM_language_accuracy=False,
        runtime_solution_demonstrated=False,training_data_reuse_allowed=False))
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','run','score'],required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/glyph_line_step_probe_v1'))
    a=p.parse_args();{'prepare':prepare,'run':run,'score':score}[a.stage](a.output.resolve())
