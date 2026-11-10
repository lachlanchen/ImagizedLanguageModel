#!/usr/bin/env python3
"""Matched broader-data condition using actual frozen final states, not a mock."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/train_glyph_tail_readout_control.py'
PROTOCOL='references/glyph_tail_data_control_protocol.md'


def load(path):
    return json.loads(Path(path).read_text())


def identity():
    for name in [OWN,PROTOCOL]:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'commit readout control first')
    return {name:sha(ROOT/name) for name in [OWN,PROTOCOL]}


def prepare(parent,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation requires hidden CUDA')
    require(not output.exists(),'preserve previous controls')
    from scripts.train_glyph_tail_adapter import verify
    m=verify(parent); code=identity()
    input_review=load(parent/'input_review.json')
    require(input_review['preparation_sha256']==sha(parent/'preparation.json')
            and input_review['all32_fresh_input_images_actually_viewed'],'parent inputs need actual review')
    require(load(parent/'cache/status.json')['phase']=='completed','complete the common prefix cache first')
    cached=load(parent/'cache/report.json')
    require(len(cached['files'])==len(m['training_jobs']),'incomplete common cache')
    output.mkdir(parents=True); (output/'cache').mkdir()
    for name in ['evaluation_references.json','input_review.json']:
        shutil.copy2(parent/name,output/name)
    save_json(output/'cache/report.json',cached)
    save_json(output/'cache/status.json',dict(phase='completed',pid=None,shared_from=str(parent.relative_to(ROOT)),
        original_report_sha256=sha(parent/'cache/report.json'),new_reader_forward_calls=0))
    bindings=dict(m['bindings_sha256'])
    for path in [parent/'preparation.json',parent/'cache/report.json']:
        bindings[str(path.relative_to(ROOT))]=sha(path)
    save_json(output/'preparation.json',dict(m,code_sha256={**m['code_sha256'],**code},bindings_sha256=bindings,
        comparison_parent=str(parent.relative_to(ROOT)),readout_only_control=True,
        context_adapters_trained=False,parent_preparation_sha256=sha(parent/'preparation.json'),
        shared_cache_report_sha256=sha(parent/'cache/report.json'),sampler_indices_will_be_retained=True,
        exact_primary_sampler_trace_equality_claimed=False))
    review=load(output/'input_review.json'); review['preparation_sha256']=sha(output/'preparation.json')
    review['identical_images_previously_viewed']=True
    review['parent_input_review_sha256']=sha(parent/'input_review.json')
    save_json(output/'input_review.json',review)
    save_json(output/'comparison_registration.json',dict(companion_run=str(parent.relative_to(ROOT)),
        protocol_sha256=sha(ROOT/PROTOCOL),all_outputs_reviewed_before_reference_scoring=True))


def fit(output):
    m=load(output/'preparation.json')
    for name,digest in {**m['code_sha256'],**m['bindings_sha256']}.items():
        require(sha(ROOT/name)==digest,'bound control source/input changed: '+name)
    require(m['readout_only_control'],'readout-control preparation required')
    require(identity()=={k:m['code_sha256'][k] for k in [OWN,PROTOCOL]},'control implementation changed')
    parent=ROOT/m['comparison_parent']
    require(load(parent/'fit/status.json')['phase']=='completed'
            and load(parent/'fit/report.json')['actual_optimizer_updates']==m['fit_steps'],
            'keep this control serial after the fixed main fit')
    require(sha(parent/'cache/report.json')==m['shared_cache_report_sha256'],'common cache changed')
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.glyph_tail_adapter import load_tail,adapter_state
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss,tensor_hash
    from scripts.train_semantic_glyph_language import bridge_model,numerics
    from scripts.train_semantic_glyph_bridge import bridge_gpu,atomic_tensor
    from scripts.train_genuine_glyph_readout import pools
    from scripts import distill_semantic_glyph_readout as base
    folder=output/'fit'; folder.mkdir(exist_ok=False)
    monitor=base.Monitor(folder,dict(maximum_stage_seconds=m['fit_seconds'],maximum_vram_gib=8.,
        minimum_runtime_ram_gib=32.,maximum_pressure_wait_seconds=300.))
    monitor.status('admitting')
    try:
        rows=load(output/'cache/report.json')['files']
        for row in rows: require(sha(ROOT/row['file'])==row['sha256'],'common frozen state changed')
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission):
            numerics(); torch.manual_seed(m['seed']); torch.cuda.manual_seed_all(m['seed'])
            # Match the main fit's initializer calls, then release the unused
            # source tail. Its setup allocation remains in the peak receipt.
            tail,start=load_tail(device); tail.train()
            model=bridge_model({},device,ROOT/m['bridge'])
            model.readout.requires_grad_(True); model.stop.requires_grad_(True)
            data=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')
            images=(1-2*data['ink'].float()).to(device)
            copied=pools(ROOT/m['genuine'],load(ROOT/m['genuine']/'preparation.json'))['copy_train']
            copied=(copied[0].to(device),copied[1].to(device))
            initial_adapters=adapter_state(tail)
            require(all(not bool(v.any()) for n,v in initial_adapters.items() if n.endswith('.up')),
                    'control source adapters must be zero')
            del tail
            parameters=[p for p in model.parameters() if p.requires_grad]
            frozen=[(p,p._version) for p in model.parameters() if not p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=m['readout_learning_rate'],weight_decay=.01)
            initial_rng=dict(cpu=tensor_hash(torch.get_rng_state()),cuda=tensor_hash(torch.cuda.get_rng_state()))
            sampled=[]; copy_indices=[]
            for step in range(m['fit_steps']):
                monitor.position=dict(step=step,total=m['fit_steps']); monitor.guard()
                optimizer.zero_grad(set_to_none=True)
                if step%4==3:
                    index=torch.randint(len(copied[0]),(128,),device=device)
                    copy_indices.append(index.cpu())
                    logits,_,_=model.predict(copied[0][index])
                    pixel=reconstruction_loss(logits,images[copied[1][index]])
                    stop=pixel*0; sampled.append(-1)
                else:
                    index=int(torch.randint(len(rows),())); sampled.append(index)
                    value=torch.load(ROOT/rows[index]['file'],weights_only=True,map_location='cpu',mmap=True)
                    states=value['original_final_predictors'][0].to(device)
                    labels=value['labels'].to(device)
                    logits,_,stops=model.predict(states)
                    pixel=reconstruction_loss(logits[:-1],images[labels[:-1]])
                    stop=.5*(F.softplus(-stops[-1])+F.softplus(stops[:-1]).mean())
                loss=pixel+stop; require(bool(torch.isfinite(loss)),'nonfinite readout-control loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step()
                fraction=(step+1)/m['fit_steps']; decay=.1+.9*.5*(1+math.cos(math.pi*fraction))
                for group in optimizer.param_groups: group['lr']=m['readout_learning_rate']*decay
                require(all(p._version==version for p,version in frozen),'frozen bridge parameter changed')
                if (step+1)%100==0:
                    item=dict(step=step+1,loss=float(loss.detach()),pixel=float(pixel.detach()),stop=float(stop.detach()))
                    with (folder/'history.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
                    monitor.status('training_readout_only',**item); print(json.dumps(item),flush=True)
                if (step+1)%1000==0:
                    path=folder/'checkpoint_latest.pt'
                    if path.exists(): path.replace(folder/'checkpoint_previous.pt')
                    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        body_tail_adapters=initial_adapters,tail_start_layer=start,
                        optimizer=optimizer.state_dict(),step=step+1,language_training_complete=True,
                        preparation_sha256=sha(output/'preparation.json'),readout_only_control=True,
                        trained_body_parameters=0,usable_language_demonstrated=False,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            path=folder/'sampled_indices.pt'
            atomic_tensor(path,dict(question_or_copy=torch.tensor(sampled),copy_indices=torch.stack(copy_indices)))
            main=torch.load(parent/'fit/checkpoint_latest.pt',weights_only=True,map_location='cpu')
            rng_match=dict(cpu=torch.equal(torch.get_rng_state(),main['torch_rng']),
                           cuda=torch.equal(torch.cuda.get_rng_state().cpu(),main['cuda_rng'].cpu()))
            save_json(folder/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),
                checkpoint_sha256=sha(folder/'checkpoint_latest.pt'),actual_optimizer_updates=m['fit_steps'],
                trained_body_parameters=0,trained_bridge_parameters=sum(p.numel() for p in parameters),
                all_frozen_parameter_versions_unchanged=True,all_checks_passed=True,
                context_adapter_ups_remain_zero=True,source_tail_initialized_only_for_matched_RNG=True,
                setup_tail_allocation_included_in_peak=True,initial_rng_sha256=initial_rng,
                final_rng_matches_main=rng_match,observed_full_sampler_trace_equality_claimed=False,
                sampled_indices_sha256=sha(path),language_accuracy_not_measured=True,
                **monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error)); raise
    finally:
        save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','fit'],required=True)
    p.add_argument('--parent',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--output',type=Path,default=Path('artifacts/glyph_tail_readout_control_v1'))
    a=p.parse_args()
    if a.stage=='prepare': prepare(a.parent.resolve(),a.output.resolve())
    else: fit(a.output.resolve())
