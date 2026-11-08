#!/usr/bin/env python3
"""Check checkpoint/RNG replay on an actual released native projection.

This CPU check exercises optimizer persistence, not language learning or a
complete-model training forward. CUDA RNG replay requires the GPU experiment.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.image_edit import visual_semantic as vs
from ilm.image_edit.native_flow_run import (CONFIG,DATA,schedule,copy_adapter_values,
    atomic_checkpoint,checkpoint_state,validate_checkpoint)


def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for this CPU-only check')
    import torch
    from torch import nn
    from safetensors import safe_open
    from ilm.image_edit.native_flow_step import NativeLowRankLinear
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    source_file=vs.SNAPSHOT/'transformer/diffusion_pytorch_model.safetensors'
    release=json.loads((vs.ACQUISITION/'report.json').read_text())
    source=next(r for r in release['files'] if r['file']=='transformer/diffusion_pytorch_model.safetensors')
    if sha(source_file)!=source['sha256']:raise ValueError('actual released generator weights changed')
    with safe_open(str(source_file),framework='pt',device='cpu') as handle:
        weight=handle.get_tensor('context_embedder.weight');bias=handle.get_tensor('context_embedder.bias')

    def construct():
        base=nn.Linear(weight.shape[1],weight.shape[0],bias=True,dtype=weight.dtype)
        with torch.no_grad():base.weight.copy_(weight);base.bias.copy_(bias)
        module=NativeLowRankLinear(base,CONFIG['rank'])
        selected={'projection.down':module.down,'projection.up':module.up}
        optimizer=torch.optim.AdamW(list(selected.values()),lr=CONFIG['learning_rate'],
            betas=tuple(CONFIG['betas']),weight_decay=CONFIG['weight_decay'],foreach=False)
        return module,selected,optimizer

    rows=[r for r in map(json.loads,(DATA/'image_pairs.jsonl').read_text().splitlines()) if r['split']=='train']
    plan=schedule(rows)
    if [r['update'] for r in plan]!=list(range(1,1537)):raise AssertionError('nonconsecutive schedule')
    metadata=dict(check='actual_released_projection_CPU_resume',source_sha256=source['sha256'],
        config=CONFIG,helper_sha256=sha(ROOT/'ilm/image_edit/native_flow_run.py'))
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20269501);module,selected,opt=construct()
    time_rng=torch.Generator().manual_seed(20269502);noise_rng=torch.Generator().manual_seed(20269503)

    def update(module,opt):
        opt.zero_grad(set_to_none=True)
        x=torch.randn((4,weight.shape[1]),generator=noise_rng,dtype=weight.dtype)
        target=torch.randn((4,weight.shape[0]),dtype=torch.float32)
        scale=torch.rand((),generator=time_rng)
        loss=((module(x).float()-target).square().mean())*(1+scale)
        loss.backward();torch.nn.utils.clip_grad_norm_(list(p for p in module.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())

    history=[];loss1=update(module,opt);history.append({**plan[0],'loss':loss1})

    def state(step):
        return checkpoint_state(adapters={k:v.detach().clone() for k,v in selected.items()},
            optimizer=opt,step=step,history=history,metadata=metadata,time_rng=time_rng,
            noise_rng=noise_rng,cuda_rng=torch.empty(0,dtype=torch.uint8))

    checkpoint=output/'checkpoint.pt';first=state(1)
    validate_checkpoint(first,metadata,plan);atomic_checkpoint(checkpoint,first,'cpu_projection_1')
    first_sha=sha(checkpoint)
    loss2=update(module,opt);history.append({**plan[1],'loss':loss2})
    expected={k:v.detach().clone() for k,v in selected.items()}
    expected_opt=copy.deepcopy(opt.state_dict());expected_rng=(torch.get_rng_state(),time_rng.get_state(),noise_rng.get_state())
    second=state(2);validate_checkpoint(second,metadata,plan)
    atomic_checkpoint(checkpoint,second,'cpu_projection_2')
    if sha(output/'checkpoint.previous.pt')!=first_sha:raise AssertionError('predecessor was not preserved byte-exactly')

    restored=torch.load(output/'checkpoint.previous.pt',map_location='cpu',weights_only=True)
    if validate_checkpoint(restored,metadata,plan)!=1:raise AssertionError('wrong replay start step')
    resumed,resumed_params,resumed_opt=construct()
    copy_adapter_values(resumed_params,restored['adapters']);resumed_opt.load_state_dict(restored['optimizer'])
    torch.set_rng_state(restored['torch_cpu_rng']);time_rng.set_state(restored['time_rng']);noise_rng.set_state(restored['noise_rng'])
    replay_loss=update(resumed,resumed_opt)
    if replay_loss!=loss2 or any(not torch.equal(expected[k],p) for k,p in resumed_params.items()):
        raise AssertionError('actual projection update did not replay exactly')
    for index,states in expected_opt['state'].items():
        for key,value in states.items():
            if not torch.equal(value,resumed_opt.state_dict()['state'][index][key]):raise AssertionError('AdamW state did not replay')
    if not all(torch.equal(a,b) for a,b in zip(expected_rng,(torch.get_rng_state(),time_rng.get_state(),noise_rng.get_state()))):
        raise AssertionError('CPU/time/noise RNG replay changed')
    if not torch.equal(weight,resumed.base.weight) or not torch.equal(bias,resumed.base.bias):raise AssertionError('frozen base changed')
    altered=copy.deepcopy(restored);altered['history'][0]['id']='wrong example'
    try:validate_checkpoint(altered,metadata,plan)
    except ValueError:pass
    else:raise AssertionError('changed schedule prefix was accepted')
    if torch.cuda.is_initialized():raise AssertionError('CPU persistence check initialized CUDA')
    report=dict(check='actual_released_projection_CPU_resume',implementation_sha256=sha(Path(__file__)),metadata=metadata,
        source_projection='context_embedder',source_dtype=str(weight.dtype),projection_updates_executed=3,
        second_update_loss=loss2,replayed_update_loss=replay_loss,adapter_tensors_exact=True,AdamW_tensors_exact=True,
        CPU_time_and_noise_RNG_exact=True,frozen_source_tensors_unchanged=True,previous_checkpoint_byte_exact=True,
        invalid_schedule_rejected=True,current_checkpoint_sha256=sha(checkpoint),previous_checkpoint_sha256=first_sha,
        checkpoint_has_actual_optimizer_state=True,full_training_schedule_updates=len(plan),cuda_initialized=False,
        CUDA_RNG_replay_tested=False,full_model_forward_exercised=False,language_learning_established=False,
        general_ilm_goal_complete=False)
    save_json(output/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
