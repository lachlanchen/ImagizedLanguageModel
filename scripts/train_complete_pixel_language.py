#!/usr/bin/env python3
"""Learn complete natural question-answer images with the actual native actor."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT,gpu_zero,save_json,sha
from scripts.preflight_native_pixel_actor import (
    NATIVE,RASTER,CONTROLS,CODEC,AUDIT,read_png,save_output,prefix_identity,FILES as ACTOR_FILES,
)
from scripts.train_conditional_pixel_density import atomic_tensor,equal_tree,require

SEED=202611093
UPDATES=8000
BATCH=4
QUALIFICATION=ROOT/'artifacts/native_pixel_actor_preflight_v2'
INITIAL=ROOT/'artifacts/native_pixel_numerics_v1/execution/initial_actor.pt'
INITIAL_SHA='e3949fc68937affb19e0ec22386b577f702dfb9e45b07f10c6d6cca735a160d6'
FILES=tuple(dict.fromkeys(('scripts/train_complete_pixel_language.py','scripts/audit_complete_pixel_language.py',
    'ilm/visual_lm/complete_pixel_training.py','references/complete_pixel_language_protocol.md',
    'scripts/probe_native_corpus_fit_v80.py','scripts/probe_corpus_fit_v75.py',
    'scripts/train_native_chinese_v80.py','ilm/visual_lm/semantic_writing_data.py',
    'ilm/visual_lm/codec_language_audit.py','ilm/visual_lm/complete_writing_data.py',*ACTOR_FILES)))


def load(path):return json.loads(Path(path).read_text())


def identity():
    code={name:sha(ROOT/name) for name in FILES}
    for name,digest in code.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()==digest,
            'commit complete-answer learning source first: '+name)
    return code


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for preparation')
    code=identity();qualified=load(QUALIFICATION/'execution/report.json');audit=load(QUALIFICATION/'audit.json')
    require(load(QUALIFICATION/'execution/status.json')['phase']=='completed'
        and all(qualified['checks'].values()) and audit['full_PNG_outputs_match_actual_tensors'],
        'completed real native learning qualification required')
    for name,digest in load(QUALIFICATION/'preparation.json')['code_sha256'].items():
        require(sha(ROOT/name)==digest,'qualified actor implementation changed: '+name)
    require(sha(INITIAL)==INITIAL_SHA,'untrained native pixel initialization changed')
    import numpy as np
    import torch
    from PIL import Image
    from scripts.probe_native_corpus_fit_v80 import cases_and_data,prompt,targets,COHORT
    from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
    torch.set_num_threads(4)
    cases,data,indices=cases_and_data();rows=[];bound={}
    def bind(path):bound[str(path.relative_to(ROOT))]=sha(path)
    for p in (COHORT,INITIAL,NATIVE/'training/checkpoint_latest.pt',RASTER/'execution/final.pt',CONTROLS,CODEC,
            QUALIFICATION/'preparation.json',QUALIFICATION/'execution/report.json',QUALIFICATION/'audit.json',
            QUALIFICATION/'execution/status.json',AUDIT/'report.json',data.images.path/'report.json'):bind(p)
    def blocks(strip):
        if strip.ndim==2:strip=strip.float().div(127.5).sub(1)[None,None]
        n=strip.shape[-1]//64
        require(1<=n<=128 and strip.shape[-1]==n*64,'complete reference exceeds physical capacity')
        return torch.cat([strip[...,64*i:64*(i+1)].clone() for i in range(n)])
    for i,case in enumerate(cases):
        index=indices[case['source']];pixels=prompt(data,index,case)
        require(torch.equal(pixels,prompt(data,index,case,independent=True)),'independent training prompt differs')
        _,_,reference=targets(data,index,case)
        rows.append(dict(id=case['source'],source='cmrc',split='train',pair=i//2,prompt=pixels,
            images=blocks(reference),references=[reference],source_file=str(data.images.path/ case['source'])))
        bind(data.images.path/case['source'])
    development=CodecLanguageAuditData(AUDIT)
    chinese=[i for i,r in enumerate(development.cases) if r['source']=='cmrc'][:16]
    categories=set();english=[]
    for i,row in enumerate(development.cases):
        if row['source']=='dolly' and row['category'] not in categories:
            english.append(i);categories.add(row['category'])
    require(len(chinese)==16 and len(english)==8,'registered development source counts differ')
    require(all(development.cases[chinese[i]]['article']==development.cases[chinese[i+1]]['article'] for i in range(0,16,2))
        and len({development.cases[i]['article'] for i in chinese})==8,'development pairs are not eight owning articles')
    for position,index in enumerate(chinese+english):
        case=development.cases[index];payload=development.payload(index)
        normal=development.prompt(index);blank=development.prompt(index,blank=True)
        require(normal.shape==blank.shape and bool((blank>=normal).all()),'erasure changed layout or added ink')
        rows.append(dict(id=case['id'],source=case['source'],split='development',
            pair=position//2 if case['source']=='cmrc' else None,category=case['category'],
            prompt=normal,blank_prompt=blank,images=blocks(payload['references'][0]),
            references=payload['references'],source_file=str(AUDIT/case['file'])))
        bind(AUDIT/case['file'])
    require(len({hashlib.sha256(r['prompt'].numpy().tobytes()).hexdigest() for r in rows})==56,
        'training and development prompt images must be distinct')
    output.mkdir(parents=True,exist_ok=False);(output/'inputs').mkdir();(output/'references').mkdir()
    manifest=[]
    for i,row in enumerate(rows):
        for route,key in [('normal','prompt')]+([('blank','blank_prompt')] if row['split']=='development' else []):
            p=output/'inputs'/f'{i:02}_{route}.png'
            Image.fromarray((row[key][0,0].numpy()>0).astype('uint8')*255).save(p)
            require(torch.equal(read_png(p),row[key]),'owning PNG does not preserve actual input bits')
        refs=[]
        for j,reference in enumerate(row['references']):
            p=output/'references'/f'{i:02}_{j:02}.png';Image.fromarray(reference.numpy()).save(p)
            refs.append(dict(path=str(p.relative_to(output)),sha256=sha(p)))
        record={k:row[k] for k in ('id','source','split','pair')}
        record.update(index=i,blocks=len(row['images']),references=refs,
            normal=f'inputs/{i:02}_normal.png',normal_sha256=sha(output/'inputs'/f'{i:02}_normal.png'))
        if row['split']=='development':record.update(blank=f'inputs/{i:02}_blank.png',blank_sha256=sha(output/'inputs'/f'{i:02}_blank.png'))
        manifest.append(record)
    atomic_tensor(output/'data.pt',rows);save_json(output/'cases.json',manifest)
    save_json(output/'preparation.json',dict(code_sha256=code,bound_sha256=bound,
        data_sha256=sha(output/'data.pt'),cases_sha256=sha(output/'cases.json'),seed=SEED,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        train_cases=32,train_pairs=16,development_cmrc=16,development_dolly=8,development_sealed=False,
        maximum_updates=UPDATES,batch_size=BATCH,maximum_draws=UPDATES*BATCH,
        train_only_early_stop='32/32 exact autonomous complete training answers at updates1000,2000,4000,8000',
        maximum_seconds=21600,maximum_allocated_vram_gib=22.,maximum_generated_blocks=128,
        density_peak_learning_rate=3e-4,other_peak_learning_rate=1e-4,warmup_updates=100,
        training_target_truncation=False,all_complete_target_pieces_scored=True,
        initial_state_sha256=INITIAL_SHA,initial_preflight_updates_retained=False,
        language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps(dict(prepared=str(output),train_cases=32,train_blocks=sum(len(r['images']) for r in rows[:32]),
        development_cases=24,development_blocks=[len(r['images']) for r in rows[32:]]),indent=2))


def unchanged(output):
    meta=load(output/'preparation.json');require(identity()==meta['code_sha256'],'learning source changed')
    for name,digest in meta['bound_sha256'].items():require(sha(ROOT/name)==digest,'bound source changed: '+name)
    require(sha(output/'data.pt')==meta['data_sha256'] and sha(output/'cases.json')==meta['cases_sha256'],'complete prepared data changed')
    for case in load(output/'cases.json'):
        for route in ('normal','blank'):
            if route in case:require(sha(output/case[route])==case[route+'_sha256'],'input PNG changed')
        for ref in case['references']:require(sha(output/ref['path'])==ref['sha256'],'reference PNG changed')
    return meta


def physical_score(images,references,stopped,capped):
    import numpy as np
    actual=np.concatenate([x[0].numpy()<0 for x in images],axis=1) if len(images) else np.zeros((32,0),bool)
    scores=[]
    for reference in references:
        ref=reference.numpy()==0;overlap=min(actual.shape[1],ref.shape[1]);tp=int((actual[:,:overlap]&ref[:,:overlap]).sum())
        den=int(actual.sum()+ref.sum())
        scores.append(dict(foreground_f1=2*tp/den if den else 1.,exact=bool(stopped and not capped and actual.shape==ref.shape and np.array_equal(actual,ref))))
    return dict(foreground_f1=max(s['foreground_f1'] for s in scores),exact=any(s['exact'] for s in scores))


def run(output, *, resume=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID','physical GPU0 required')
    meta=unchanged(output);execution=output/'execution'
    if resume:
        prior=load(execution/'status.json')
        require(prior.get('phase')!='completed' and not (prior.get('pid') and Path('/proc',str(prior['pid'])).exists()),
            'a live or completed learning execution may not be restarted')
        require(any((execution/name).exists() for name in ('checkpoint_latest.pt','checkpoint_previous.pt')),
            'an actual complete checkpoint is required for explicit resume')
    else:execution.mkdir(exist_ok=False)
    started=time.perf_counter();step=0;executed=0;optimizer_seconds=0.
    def status(phase,**extra):
        event=dict(phase=phase,pid=os.getpid(),update=step,elapsed_seconds=time.perf_counter()-started,**extra)
        save_json(execution/'status.json',event)
        with (execution/'events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    def deadline(signum,frame):raise TimeoutError('registered6-hour complete learning execution deadline')
    old_handler=signal.signal(signal.SIGALRM,deadline);signal.alarm(meta['maximum_seconds'])
    try:
        with gpu_zero() as device:
            import gc
            import torch
            from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
            from ilm.visual_lm.native_pixel_language import NativePixelLanguageModel
            from ilm.visual_lm.complete_pixel_training import complete_pixel_loss,cpu_tree
            from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
            from scripts.train_native_chinese_v80 import batch
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            rows=torch.load(output/'data.pt',map_location='cpu',weights_only=True)
            local=ROOT.parent/'LocalLLM/.local';status('loading_qualified_native_pixel_actor')
            actor,provenance=load_trained_native_image_actor(checkpoint=NATIVE/'training/checkpoint_latest.pt',
                controls=CONTROLS,codec=CODEC,ollama_root=local/'models/ollama',gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
            previous=actor.writing
            raster=torch.load(RASTER/'execution/final.pt',map_location='cpu',weights_only=True)['model']
            writer=NativePixelLanguageModel(previous.continuation.context,previous.writer,input_scale=float(previous.input_scale),query_seed=SEED+1,raster_state=raster).to(device)
            writer.load_trainable_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True));actor.writing=writer
            del previous,raster;gc.collect();torch.cuda.empty_cache();actor.eval()
            def forbidden(*args,**kwargs):raise RuntimeError('the old codec decoder entered output generation')
            writer.writer.predict=forbidden;writer.writer.codec.decode=forbidden
            parameters=[p for p in writer.parameters() if p.requires_grad]
            groups=[dict(params=[p for n,p in writer.named_parameters() if p.requires_grad and n.startswith('density.')],peak=3e-4),
                dict(params=[p for n,p in writer.named_parameters() if p.requires_grad and not n.startswith('density.')],peak=1e-4)]
            optimizer=torch.optim.AdamW(groups,lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
            sampler=ExactEpochSampler(32,SEED+1)
            pool=[];prefix_hashes=[]
            for i,row in enumerate(rows[:32]):
                prefix,geometry=actor.prefill(read_png(output/f'inputs/{i:02}_normal.png'))
                require(geometry['all_prompt_keys_retained'] and prefix.blocked_before==0,'full native prompt KV required')
                prefix_hashes.append(prefix_identity(prefix));pool.append((prefix.to('cpu'),writer.image_codes(row['images']).cpu()))
                del prefix;status('native_training_PNG_prefill',completed=i+1,total=32)
            prefix_file=execution/'prefix_identities.json'
            if prefix_file.exists():require(load(prefix_file)==prefix_hashes,'native source prefixes changed before resume')
            else:save_json(prefix_file,prefix_hashes)
            fixed={n:(id(p),p._version) for n,p in actor.named_parameters() if not p.requires_grad}
            def guard():
                require(torch.cuda.max_memory_allocated()/1024**3<=22.,'complete learning exceeds registered22GiB allocation')
                require(all((id(p),p._version)==fixed[n] for n,p in actor.named_parameters() if n in fixed),'a frozen source parameter changed')
            def loss(indices, *, checkpoint=True):
                prefix,codes,lengths=batch(pool,indices,device)
                return complete_pixel_loss(writer,prefix,[rows[i]['images'] for i in indices],codes,lengths,checkpoint_layers=checkpoint)
            def update(next_step):
                nonlocal executed,optimizer_seconds
                update_started=time.perf_counter()
                indices=sampler.take(BATCH);actor.train();optimizer.zero_grad(set_to_none=True)
                factor=min(next_step/100,1.)*(.1+.9*.5*(1+math.cos(math.pi*(next_step-1)/(UPDATES-1))))
                for group in optimizer.param_groups:group['lr']=group['peak']*factor
                value,parts=loss(indices);require(bool(torch.isfinite(value)),'nonfinite complete-image objective')
                value.backward();norms={k:0. for k in ('density','image_input','start_query','suffix','stop')}
                for name,p in writer.named_parameters():
                    if p.grad is None:continue
                    require(bool(torch.isfinite(p.grad).all()),'nonfinite gradient: '+name)
                    key='suffix' if name.startswith('continuation.context.') else name.split('.')[0]
                    if key in norms:norms[key]+=float(p.grad.double().square().sum())
                norms={k:v**.5 for k,v in norms.items()}
                if next_step<=2:require(all(v>0 for v in norms.values()),'a required initial complete-image gradient is zero')
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();executed+=1
                seconds=time.perf_counter()-update_started;optimizer_seconds+=seconds
                with (execution/'optimizer_calls.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(call=executed,update=next_step,indices=indices,seconds=seconds))+'\n')
                guard()
                return dict(update=next_step,indices=indices,loss=float(value.detach()),image_nll=float(parts['image_nll']),stop=float(parts['stop']),
                    gradient_norm=float(norm),gradient_components=norms,complete_blocks=parts['complete_blocks'],
                    density_learning_rate=optimizer.param_groups[0]['lr'],other_learning_rate=optimizer.param_groups[1]['lr'])
            def checkpoint(path):
                atomic_tensor(path,dict(model=writer.trainable_state_dict(),optimizer=cpu_tree(optimizer.state_dict()),
                    sampler=sampler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                    update=step,executed_optimizer_calls=executed,preparation_sha256=sha(output/'preparation.json'),
                    training_complete=False,frozen_weights_saved=False))
            def restore(value):
                writer.load_trainable_state_dict(value['model']);optimizer.load_state_dict(value['optimizer']);sampler.load_state_dict(value['sampler'])
                torch.set_rng_state(value['cpu_rng']);torch.cuda.set_rng_state(value['cuda_rng'])
            if resume:
                checkpoint_file=execution/'checkpoint_latest.pt'
                if not checkpoint_file.exists():checkpoint_file=execution/'checkpoint_previous.pt'
                value=torch.load(checkpoint_file,map_location='cpu',weights_only=True)
                require(value['preparation_sha256']==sha(output/'preparation.json'),'resume source identity changed')
                restore(value);step=value['update'];del value
                calls=[json.loads(line) for line in (execution/'optimizer_calls.jsonl').read_text().splitlines()]
                require(all(row['call']==i+1 for i,row in enumerate(calls)),'actual optimizer-call ledger is incomplete')
                executed=len(calls);optimizer_seconds=sum(row['seconds'] for row in calls)
                history=execution/'history.jsonl';lines=history.read_text().splitlines(keepends=True)
                require(len(lines)>=step and all(json.loads(line)['update']==i+1 for i,line in enumerate(lines[:step])),
                    'committed checkpoint history is incomplete or out of order')
                if len(lines)>step:
                    orphan=execution/f'uncommitted_history_before_resume_{time.time_ns()}.jsonl'
                    orphan.write_text(''.join(lines[step:]));history.write_text(''.join(lines[:step]))
            else:
                status('validating_actual_variable_length_batch')
                with torch.no_grad():
                    selected=[0,2,3,17];prefix,codes,lengths=batch(pool,selected,device)
                    full=writer.conditions(prefix,codes,checkpoint_layers=False)
                    independent=[];maximum_relative=0.;maximum_absolute=0.
                    for b,i in enumerate(selected):
                        own=writer.conditions(pool[i][0].to(device),pool[i][1][None].to(device),checkpoint_layers=False)[0]
                        delta=full[b,:len(own)]-own
                        maximum_absolute=max(maximum_absolute,float(delta.abs().max()))
                        maximum_relative=max(maximum_relative,float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(own).clamp_min(1e-12)))
                        independent.append(float(writer.loss_from_codes(pool[i][0].to(device),rows[i]['images'].to(device),pool[i][1].to(device),writer_blocks=128)[0]))
                    combined=float(loss(selected,checkpoint=False)[0]);mean=sum(independent)/4
                    require(maximum_relative<=1e-4 and abs(combined-mean)<=1e-4*max(1.,abs(mean)),'variable-length batch differs from independent full-answer objective')
                    changed=codes.clone()
                    for b,i in enumerate(selected):
                        if len(rows[i]['images'])>1:changed[b,1]=writer.image_codes(-rows[i]['images'][1:2])[0]
                    future=writer.conditions(prefix,changed,checkpoint_layers=False)
                    require(torch.equal(full[:,:2],future[:,:2]),'future physical answer image leaked into its query')
                    for b,n in enumerate(lengths.tolist()):changed[b,n:]=19.
                    padded=writer.conditions(prefix,changed,checkpoint_layers=False)
                    require(all(torch.equal(future[b,:n+1],padded[b,:n+1]) for b,n in enumerate(lengths.tolist())),
                        'padded future image codes contaminate valid queries')
                    save_json(execution/'batch_preflight.json',dict(selected=selected,actual_lengths=lengths.tolist(),
                        native_batch_independent_normwise_error=maximum_relative,absolute_error=maximum_absolute,
                        batch_loss=combined,mean_independent_loss=mean,future_image_exclusion_exact=True,
                        future_padding_exclusion_exact=True,all_target_pieces_scored=True,passed=True))
                del prefix,codes,lengths,full,future,padded,changed,own
            collector=None;encode=writer.writer.encode_targets
            def traced(images):
                values=encode(images)
                if collector is not None:collector.append(dict(images=images.detach().cpu().clone(),codes=values.detach().cpu().clone()))
                return values
            writer.writer.encode_targets=traced
            def generate_case(folder, *, index, route, actual_png):
                nonlocal collector
                receipt=folder/'receipt.json'
                expected=dict(index=index,route=route,update=step,actual_PNG_prefill=actual_png,
                    preparation_sha256=sha(output/'preparation.json'))
                if receipt.exists():
                    saved=load(receipt)
                    require(saved['identity']==expected,'existing output belongs to a different model state')
                    for name,digest in saved['hashes'].items():require(sha(folder/name)==digest,'saved complete output changed')
                    return torch.load(folder/'result.pt',map_location='cpu',weights_only=True),saved['output']
                if folder.exists():
                    require(resume,'preserve an unexpected partial output')
                    folder.rename(folder.with_name(folder.name+f'_uncommitted_{time.time_ns()}'))
                collector=[]
                result=(actor.generate(read_png(output/f'inputs/{index:02}_{route}.png'),maximum_blocks=128) if actual_png
                    else writer.generate(pool[index][0].to(device),maximum_blocks=128))
                saved=save_output(folder,result);atomic_tensor(folder/'feedback.pt',collector);collector=None
                save_json(receipt,dict(identity=expected,output=saved,
                    hashes={n:sha(folder/n) for n in ('result.pt','answer.png','feedback.pt')}))
                return result,saved
            def generate_train(label, *, actual_png=False):
                nonlocal collector
                actor.eval();folder=execution/label;folder.mkdir(exist_ok=True);scores=[]
                for i in range(32):
                    result,saved=generate_case(folder/f'{i:02}',index=i,route='normal',actual_png=actual_png)
                    scores.append(dict(index=i,**physical_score(result['images'],rows[i]['references'],result['stopped'],result['capped']),output=saved))
                    guard();status(label,completed=i+1,total=32)
                result=dict(update=step,cases=scores,exact=sum(s['exact'] for s in scores),
                    foreground_f1=sum(s['foreground_f1'] for s in scores)/32,actual_PNG_prefill=actual_png)
                save_json(folder/'report.json',result);return result
            def monitor():
                actor.eval()
                with torch.no_grad():
                    values=[loss(list(range(i,i+4)),checkpoint=False)[1] for i in range(0,32,4)]
                result=dict(update=step,image_nll=sum(float(x['image_nll']) for x in values)/8,
                    stop=sum(float(x['stop']) for x in values)/8,complete_training_questions=32)
                with (execution/'monitor.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
                return result
            if not resume:
                monitor();generate_train('initial_train')
                first=update(1);step=1;checkpoint(execution/'resume_check.pt')
                snapshot=torch.load(execution/'resume_check.pt',map_location='cpu',weights_only=True)
                second=update(2);step=2;expected_model=writer.trainable_state_dict();expected_optimizer=cpu_tree(optimizer.state_dict());expected_sampler=sampler.state_dict()
                restore(snapshot);repeated=update(2)
                require(second==repeated and equal_tree(expected_model,writer.trainable_state_dict())
                    and equal_tree(expected_optimizer,cpu_tree(optimizer.state_dict())) and equal_tree(expected_sampler,sampler.state_dict()),
                    'actual complete-answer batch optimizer/sampler replay differs')
                save_json(execution/'update_replay.json',dict(first=first,second=second,repeated=repeated,
                    model_optimizer_sampler_loss_gradients_bit_exact=True,retained_updates=2,actual_optimizer_calls=3))
                with (execution/'history.jsonl').open('a') as stream:stream.write(json.dumps(first)+'\n'+json.dumps(second)+'\n')
                del snapshot,expected_model,expected_optimizer,expected_sampler
                checkpoint(execution/'checkpoint_latest.pt')
            stop_reason='maximum_registered_updates'
            training_started=time.perf_counter();status('complete_answer_training',retained_draws=sampler.draws)
            if resume and step in (1000,2000,4000,8000):
                sample=generate_train(f'train_update_{step:05}')
                if sample['exact']==32:stop_reason='all32_complete_training_answers_exact_at_registered_milestone'
            planned=range(step+1,UPDATES+1) if stop_reason=='maximum_registered_updates' else ()
            for next_step in planned:
                record=update(next_step);step=next_step
                with (execution/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                if step%25==0:status('complete_answer_training',retained_draws=sampler.draws,last=record,training_elapsed_seconds=time.perf_counter()-training_started)
                if step%250==0:
                    latest=execution/'checkpoint_latest.pt'
                    if latest.exists():latest.replace(execution/'checkpoint_previous.pt')
                    checkpoint(latest)
                if step in (1000,2000,4000,8000):
                    monitor();sample=generate_train(f'train_update_{step:05}')
                    if sample['exact']==32:stop_reason='all32_complete_training_answers_exact_at_registered_milestone';break
            for i in range(32):
                prefix,_=actor.prefill(read_png(output/f'inputs/{i:02}_normal.png'))
                require(prefix_identity(prefix)==prefix_hashes[i],'learning changed native prompt KV; cached training prefixes became stale')
                del prefix
            final_train=generate_train('final_train',actual_png=True)
            development=[];actor.eval()
            for i in range(32,56):
                for route in ('normal','blank'):
                    folder=execution/'development'/f'{i:02}_{route}'
                    result,saved=generate_case(folder,index=i,route=route,actual_png=True)
                    development.append(dict(index=i,route=route,output=saved));guard();status('held_out_complete_answer_generation',completed=len(development),total=48)
            checkpoint(execution/'checkpoint_latest.pt')
            final=torch.load(execution/'checkpoint_latest.pt',map_location='cpu',weights_only=True);final['training_complete']=True
            atomic_tensor(execution/'checkpoint_latest.pt',final);del final
            save_json(execution/'report.json',dict(updates=step,retained_draws=sampler.draws,actual_optimizer_calls=executed,
                stop_reason=stop_reason,initial_state_sha256=INITIAL_SHA,final_checkpoint_sha256=sha(execution/'checkpoint_latest.pt'),
                complete_parameters=sum(p.numel() for p in actor.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
                source=provenance,all32_native_prefills_bit_exact_after_learning=True,final_training=final_train,
                development_outputs=development,codec_decoder_calls=0,feedback_from_emitted_pixels=True,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.perf_counter()-started,
                active_optimizer_seconds=optimizer_seconds,training_and_final_assessment_seconds=time.perf_counter()-training_started,
                explicit_resume_used=resume,language_quality_not_yet_audited=True,general_ilm_goal_complete=False))
            status('completed',retained_draws=sampler.draws)
    except BaseException as error:
        status('failed',reason=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        s=load(execution/'status.json') if (execution/'status.json').exists() else {}
        if 'torch' in locals() and torch.cuda.is_initialized():s['peak_allocated_vram_gib']=torch.cuda.max_memory_allocated()/1024**3
        save_json(execution/'status.json',dict(s,pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('prepare','run'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve(),resume=args.resume)
