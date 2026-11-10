#!/usr/bin/env python3
"""Refine the writer on actual fixed-reader states with balanced positions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/train_balanced_glyph_writer.py'
PROTOCOL = 'references/balanced_glyph_writer_protocol.md'
PARENT_SHA = '5c434a9a9b0c6ddc0c3fe4005265e4f8ca5b7d0add9a74a80ed371806fa71085'


def load(path): return json.loads(Path(path).read_text())
def rel(path): return str(Path(path).resolve().relative_to(ROOT))


def identity():
    result = {}
    for name in [OWN, PROTOCOL]:
        require(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
                == (ROOT/name).read_bytes(), 'commit balanced writer sources before execution')
        result[name] = sha(ROOT/name)
    return result


def verify(run):
    m = load(run/'preparation.json')
    for name, digest in {**m['code_sha256'], **m['bindings_sha256']}.items():
        require(sha(ROOT/name) == digest, 'balanced writer input changed: '+name)
    require(identity() == {n:m['code_sha256'][n] for n in [OWN, PROTOCOL]}, 'balanced writer code changed')
    return m


def prepare(parent, run, companion):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only data preparation')
    require(not run.exists() and not companion.exists(), 'preserve previous runs')
    from scripts.assess_glyph_tail_adapter import candidate
    m, checkpoint = candidate(parent)
    require(sha(checkpoint) == PARENT_SHA, 'fixed completed contextual parent required')
    require((parent/'raw_review/scores.json').exists(), 'finish previous fixed comparison first')
    code = identity()
    import torch
    from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    torch.set_num_threads(2)
    dataset = CompleteWritingImageData(ROOT/'data/general_instructions/complete_lines_development_20261105',
                                      purpose='development')
    rows = [json.loads(line) for line in (ROOT/'data/visual_qa/cmrc2018_short/development.jsonl').read_text().splitlines()]
    by_id = {r['id']:r for r in rows}
    old = load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    excluded = {r['context_hash'] for r in map(json.loads,
        (ROOT/'data/visual_qa/cmrc2018_short/train.jsonl').read_text().splitlines())}
    excluded |= {r['context_hash'] for r in m['fresh_cases']}
    excluded |= {by_id[r['id']]['context_hash'] for r in old['development'] if r['source'] == 'cmrc'}
    groups = {}
    for row in rows:
        if row['context_hash'] not in excluded and 'cmrc/'+row['id']+'.pt' in dataset.names:
            groups.setdefault(row['context_hash'], []).append(row)
    order = lambda s:hashlib.sha256(('balanced-glyph-writer-fresh32-v1\0'+s).encode()).hexdigest()
    contexts = sorted((k for k,v in groups.items() if len(v)>=2), key=order)[:16]
    require(len(contexts) == 16, 'need sixteen new contexts')
    run.mkdir(parents=True); companion.mkdir(parents=True); (run/'inputs').mkdir()
    bindings = dict(m['bindings_sha256'])
    def bind(path): bindings[rel(path)] = sha(path); return rel(path)
    for path in [parent/'preparation.json', parent/'cache/report.json', checkpoint,
                 parent/'qualify/report.json', parent/'qualify/records.json', parent/'qualify/visual_review.json',
                 parent/'raw_review/scores.json']:
        bind(path)
    fresh, references = [], {}
    for pair, context in enumerate(contexts):
        for row in sorted(groups[context], key=lambda x:order(x['id']))[:2]:
            name = 'cmrc/'+row['id']+'.pt'; full,boundary = dataset.get(name)
            pixels = fold_writing_prompt(full[...,:boundary*64]).pixels
            path = run/'inputs'/('fresh_'+row['id']+'.png')
            Image.fromarray(((pixels[0,0].numpy()+1)*127.5).astype(np.uint8)).save(path)
            index = len(fresh)
            fresh.append(dict(index=index,id=row['id'],pair=pair,context_hash=context,prompt=bind(path),
                source_tensor_sha256=dataset.report['files_sha256'][name]))
            references[f'fresh/{index:04}/normal'] = [v['text'] for v in row['references']]
    boards = []
    for start in range(0,32,2):
        a,b = [Image.open(ROOT/r['prompt']).convert('L') for r in fresh[start:start+2]]
        canvas = Image.new('L',(a.width+b.width+24,max(a.height,b.height)+28),245)
        d = ImageDraw.Draw(canvas); d.text((6,4),f'new fresh {start:02} and {start+1:02}; complete inputs',fill=0)
        canvas.paste(a,(4,24));canvas.paste(b,(a.width+16,24))
        name = f'input_preview_{start//2:02}.png';canvas.save(run/name)
        boards.append(dict(file=name,sha256=sha(run/name)))
    save_json(run/'evaluation_references.json',references)
    metadata = dict(m,code_sha256={**m['code_sha256'],**code},bindings_sha256=bindings,
        balanced_writer_refinement=True,comparison_parent=rel(parent),initial_checkpoint=rel(checkpoint),
        inherited_checkpoint_sha256=PARENT_SHA,inherited_optimizer_updates=18000,new_optimizer_updates=18000,
        fit_steps=18000,seed=202611121,readout_learning_rate=.0001,
        fresh_cases=fresh,input_preview_boards=boards,
        evaluation_references_sha256=sha(run/'evaluation_references.json'),
        previous_fresh_cohort_now_development=True,new_contexts_exclude_prior_reserved=True,
        body_adapters_frozen=True,cache_seconds=3600.,fit_seconds=3600.,
        cache_parent_report=rel(parent/'cache/report.json'),qualification_parent=rel(parent/'qualify/report.json'))
    save_json(run/'preparation.json',metadata)
    # The unchanged baseline genuinely reuses its completed fit and physical
    # qualification. No duplicate model, invented fit or new gate is created.
    for name in ['fit','qualify']:(companion/name).symlink_to((parent/name).resolve(),target_is_directory=True)
    (companion/'cache').mkdir();shutil.copy2(parent/'cache/report.json',companion/'cache/report.json')
    for name in ['evaluation_references.json']:shutil.copy2(run/name,companion/name)
    save_json(companion/'preparation.json',dict(metadata,unchanged_parent_baseline=True,
        new_optimizer_updates=0,physical_qualification_reused_with_identical_weights=True))
    for first,second in [(run,companion),(companion,run)]:
        save_json(first/'comparison_registration.json',dict(companion_run=rel(second),
            all_outputs_reviewed_before_reference_scoring=True,protocol_sha256=sha(ROOT/PROTOCOL)))
    print(json.dumps(dict(training_questions=len(m['training_jobs']),new_questions=len(fresh),
        prior_held_context_overlap=0,model_calls=0,CUDA_initialized=torch.cuda.is_initialized())))


def cache(run):
    m = verify(run)
    review = load(run/'input_review.json')
    require(review['all32_fresh_input_images_actually_viewed']
            and review['preparation_sha256']==sha(run/'preparation.json'), 'inspect new physical inputs before cache/fit')
    import torch
    from ilm.visual_lm.glyph_tail_adapter import load_tail
    from scripts.train_semantic_glyph_language import bridge_model,numerics,atomic_tensor
    from scripts.train_semantic_glyph_bridge import bridge_gpu
    from scripts import distill_semantic_glyph_readout as base
    from scripts.assess_glyph_tail_adapter import f1
    folder = run/'cache'; folder.mkdir(exist_ok=False)
    monitor = base.Monitor(folder,dict(maximum_stage_seconds=m['cache_seconds'],maximum_vram_gib=12.,
        minimum_runtime_ram_gib=32.,maximum_pressure_wait_seconds=1800.));monitor.status('admitting')
    try:
        rows = load(ROOT/m['cache_parent_report'])['files']
        for r in rows:require(sha(ROOT/r['file'])==r['sha256'],'actual prefix cache changed')
        monitor.guard()
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard()
            initial=torch.load(ROOT/m['initial_checkpoint'],weights_only=True,map_location='cpu')
            tail,start=load_tail(device);tail.eval()
            parameters=dict(tail.named_parameters())
            require(start==32,'tail boundary changed')
            for n,v in initial['body_tail_adapters'].items():parameters[n].copy_(v.to(device))
            model=bridge_model({},device,ROOT/m['initial_checkpoint'])
            ink=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'][:,0].numpy()
            old=load(ROOT/m['comparison_parent']/'qualify/records.json')['training'];oldpos=0
            pools={name:[] for name in ['first','later','terminal']};labels={n:[] for n in ['first','later']}
            terminal_ok=continue_ok=0
            for index,row in enumerate(rows):
                monitor.position=dict(completed=index,total=len(rows));monitor.guard()
                v=torch.load(ROOT/row['file'],weights_only=True,map_location='cpu',mmap=True)
                h=tail(inputs_embeds=v['lower'].to(device),position_ids=v['position_ids'].to(device),
                    use_cache=False).last_hidden_state[0,v['boundary']-1:].float()
                logits,_,stops=model.predict(h);actual=(logits[:-1]>=0).to(torch.uint8)[:,0].cpu().numpy()
                y=v['labels'][:-1].cpu();require(len(h)==len(y)+1,'causal state count differs')
                require(y.tolist()==m['training_jobs'][index]['labels'],'training-label ordering differs')
                for j,(a,label) in enumerate(zip(actual,y.tolist())):
                    r=old[oldpos];require((r['case'],r['position'],r['label'])==(index,j,label),'qualification order differs')
                    require(abs(f1(a,ink[label])-r['f1'])<1e-12
                        and bool(np.array_equal(a,ink[label]))==r['exact'],'new cache does not reproduce actual parent pixels')
                    oldpos+=1
                terminal_ok+=int(stops[-1]>=0);continue_ok+=int((stops[:-1]<0).sum())
                h=h.cpu();pools['first'].append(h[:1]);labels['first'].append(y[:1])
                pools['later'].append(h[1:-1]);labels['later'].append(y[1:]);pools['terminal'].append(h[-1:])
                if index%100==0:monitor.status('caching_actual_adapted_training_states')
            q=load(ROOT/m['qualification_parent'])
            require(oldpos==len(old) and terminal_ok==q['terminal_correct']
                and continue_ok==q['continue_correct'],'parent stop or score reproduction differs')
            value={n:torch.cat(v) for n,v in pools.items()}
            value.update({n+'_labels':torch.cat(v) for n,v in labels.items()})
            atomic_tensor(folder/'pools.pt',value)
            save_json(folder/'report.json',dict(pool_sha256=sha(folder/'pools.pt'),
                pools={n:len(value[n]) for n in pools},source_cache_sha256=sha(ROOT/m['cache_parent_report']),
                actual_parent_training_pixels_and_stop_counts_reproduced=True,
                all23262_training_pixel_scores_verified=True,heldout_model_calls=0,optimizer_updates=0,
                cached_body_fixed_during_subsequent_writer_fit=True,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


def fit(run):
    m=verify(run);require(load(run/'cache/status.json')['phase']=='completed','finish genuine cache first')
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    from scripts.train_semantic_glyph_language import bridge_model,numerics,atomic_tensor
    from scripts.train_semantic_glyph_bridge import bridge_gpu
    from scripts.train_genuine_glyph_readout import pools
    from scripts import distill_semantic_glyph_readout as base
    folder=run/'fit';folder.mkdir(exist_ok=False)
    monitor=base.Monitor(folder,dict(maximum_stage_seconds=m['fit_seconds'],maximum_vram_gib=12.,
        minimum_runtime_ram_gib=32.,maximum_pressure_wait_seconds=1800.));monitor.status('admitting')
    try:
        require(sha(run/'cache/pools.pt')==load(run/'cache/report.json')['pool_sha256'],'training states changed')
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission):
            monitor.admission=admission;numerics();torch.manual_seed(m['seed']);torch.cuda.manual_seed_all(m['seed'])
            model=bridge_model({},device,ROOT/m['initial_checkpoint'])
            model.readout.requires_grad_(True);model.stop.requires_grad_(True)
            initial=torch.load(ROOT/m['initial_checkpoint'],weights_only=True,map_location='cpu')
            p={n:v.to(device) for n,v in torch.load(run/'cache/pools.pt',weights_only=True,map_location='cpu').items()}
            images=(1-2*torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'].float()).to(device)
            copied=pools(ROOT/m['genuine'],load(ROOT/m['genuine']/'preparation.json'))['copy_train']
            copied=tuple(x.to(device) for x in copied)
            parameters=[v for v in model.parameters() if v.requires_grad]
            frozen=[(v,v._version) for v in model.parameters() if not v.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=m['readout_learning_rate'],weight_decay=.01)
            sample_digest=hashlib.sha256();counts=dict(first=0,later=0,terminal=0,copy=0)
            def sample(name,n):
                ix=torch.randint(len(p[name]),(n,),device=device)
                sample_digest.update(name.encode());sample_digest.update(ix.cpu().numpy().tobytes());counts[name]+=n
                return ix
            for step in range(m['fit_steps']):
                monitor.position=dict(step=step,total=m['fit_steps']);monitor.guard();optimizer.zero_grad(set_to_none=True)
                if step%4==3:
                    ix=torch.randint(len(copied[0]),(128,),device=device);counts['copy']+=128
                    sample_digest.update(b'copy');sample_digest.update(ix.cpu().numpy().tobytes())
                    logits,_,_=model.predict(copied[0][ix]);pixel=reconstruction_loss(logits,images[copied[1][ix]])
                    stop=pixel*0;first=later=pixel*0
                else:
                    a,b,t=sample('first',64),sample('later',64),sample('terminal',64)
                    states=torch.cat([p['first'][a],p['later'][b]])
                    logits,_,stops=model.predict(states)
                    first=reconstruction_loss(logits[:64],images[p['first_labels'][a]])
                    later=reconstruction_loss(logits[64:],images[p['later_labels'][b]])
                    pixel=.5*(first+later)
                    stop=.5*(F.softplus(stops).mean()+F.softplus(-model.stop(p['terminal'][t]).flatten()).mean())
                loss=pixel+stop;require(bool(torch.isfinite(loss)),'nonfinite balanced writer loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
                decay=.1+.9*.5*(1+math.cos(math.pi*(step+1)/m['fit_steps']))
                for group in optimizer.param_groups:group['lr']=m['readout_learning_rate']*decay
                require(all(v._version==ver for v,ver in frozen),'frozen encoder/decoder/buffers changed')
                if step%100==0 or (step+1)%100==0:
                    row=dict(step=step+1,loss=float(loss.detach()),pixel=float(pixel.detach()),stop=float(stop.detach()),
                        first=float(first.detach()),later=float(later.detach()),route='copy' if step%4==3 else 'question')
                    with (folder/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    monitor.status('training_balanced_image_writer',**row);print(json.dumps(row),flush=True)
                if (step+1)%1000==0:
                    target=folder/'checkpoint_latest.pt'
                    if target.exists():target.replace(folder/'checkpoint_previous.pt')
                    atomic_tensor(target,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        body_tail_adapters=initial['body_tail_adapters'],tail_start_layer=32,step=step+1,
                        inherited_optimizer_updates=18000,new_optimizer_updates=step+1,
                        language_training_complete=step+1==m['fit_steps'],balanced_writer_refinement=True,
                        preparation_sha256=sha(run/'preparation.json'),optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                        trained_body_parameters=0,usable_language_demonstrated=False))
            expected=dict(first=864000,later=864000,terminal=864000,copy=576000)
            require(counts==expected,'fixed exposure count differs')
            require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'nonfinite final writer')
            save_json(folder/'report.json',dict(actual_optimizer_updates=m['fit_steps'],all_checks_passed=True,
                checkpoint_sha256=sha(folder/'checkpoint_latest.pt'),inherited_checkpoint_sha256=PARENT_SHA,
                inherited_optimizer_updates=18000,new_optimizer_updates=m['fit_steps'],sampled_target_counts=counts,
                sampler_sha256=sample_digest.hexdigest(),trained_body_parameters=0,
                trained_bridge_parameters=sum(v.numel() for v in parameters),
                all_frozen_parameter_versions_unchanged=True,language_accuracy_not_measured=True,
                full_body_loaded=False,new_heldout_references_used_for_training=False,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','cache','fit'],required=True)
    p.add_argument('--parent',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--run',type=Path,default=Path('artifacts/balanced_glyph_writer_v1'))
    p.add_argument('--companion',type=Path,default=Path('artifacts/balanced_glyph_parent_v1'))
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.parent.resolve(),a.run.resolve(),a.companion.resolve())
    else:{'cache':cache,'fit':fit}[a.stage](a.run.resolve())
