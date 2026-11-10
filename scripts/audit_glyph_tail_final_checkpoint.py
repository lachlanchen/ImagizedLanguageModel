#!/usr/bin/env python3
"""Check real final optimizer counters and frozen tensors without neural execution."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/audit_glyph_tail_final_checkpoint.py'


def load(path):
    return json.loads(Path(path).read_text())


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit requires hidden CUDA')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'commit checkpoint auditor first')
    require(not output.exists(),'preserve the previous checkpoint audit')
    require(load(run/'fit/status.json')['phase']=='completed','audit a completed final fit only')
    import torch
    torch.set_num_threads(2)
    m=load(run/'preparation.json'); report=load(run/'fit/report.json')
    checkpoint=run/'fit/checkpoint_latest.pt'; base=ROOT/m['bridge']
    require(sha(checkpoint)==report['checkpoint_sha256'],'final checkpoint changed')
    require(sha(base)==m['bindings_sha256'][m['bridge']],'starting bridge changed')
    value=torch.load(checkpoint,weights_only=True,map_location='cpu')
    original=torch.load(base,weights_only=True,map_location='cpu')['model']
    require(value['step']==m['fit_steps']==18000 and value['language_training_complete']
            and value['preparation_sha256']==sha(run/'preparation.json'),'incomplete or wrong fit')
    model=value['model']; body=value['body_tail_adapters']; opt=value['optimizer']
    require(set(model)==set(original),'bridge tensor keys changed')
    require(all(torch.isfinite(t).all() for t in [*model.values(),*body.values()]),'nonfinite final model')
    head=[n for n in model if n.startswith(('readout.','stop.'))]
    frozen=[n for n in model if n not in head]
    require(all(torch.equal(model[n],original[n]) for n in frozen),'frozen E/D/normalization differs from start')
    is_control=bool(m.get('readout_only_control',False))
    require(len(body)==24 and all(n.endswith(('.up','.down')) for n in body),'tail parameter identity differs')
    up_nonzero=sum(int(torch.count_nonzero(t)) for n,t in body.items() if n.endswith('.up'))
    require((up_nonzero==0)==is_control,'body intervention differs from registered condition')
    expected_groups=1 if is_control else 2
    require(len(opt['param_groups'])==expected_groups,'optimizer group count differs')
    mapping=[]
    if not is_control:
        mapping.append(('body',list(body),opt['param_groups'][0],m['adapter_learning_rate']))
    mapping.append(('bridge',head,opt['param_groups'][-1],m['readout_learning_rate']))
    counters=[]
    for group_name,names,group,rate in mapping:
        require(len(names)==len(group['params']),'optimizer parameter ordering differs')
        require(group['lr']==rate*.1,'final cosine schedule position differs')
        for name,index in zip(names,group['params']):
            state=opt['state'][index]
            expected=18000 if name.startswith('readout.') else 13500
            actual=int(state['step'].item())
            require(actual==expected,'real optimizer update counter differs: '+name)
            require(all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)),
                    'nonfinite optimizer state')
            counters.append(dict(group=group_name,name=name,actual_updates=actual,expected_updates=expected))
    changed={prefix:sum(int(torch.count_nonzero(model[n]!=original[n])) for n in head if n.startswith(prefix))
             for prefix in ['readout.','stop.']}
    require(all(n>0 for n in changed.values()),'writer or stop head did not change')
    trained_body=0 if is_control else sum(t.numel() for t in body.values())
    trained_head=sum(model[n].numel() for n in head)
    require(trained_body==report['trained_body_parameters'] and trained_head==report['trained_bridge_parameters'],
            'trained-parameter report differs from actual tensors')
    require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    save_json(output,dict(source_sha256=sha(ROOT/OWN),checkpoint_sha256=sha(checkpoint),
        starting_bridge_sha256=sha(base),preparation_sha256=sha(run/'preparation.json'),
        global_retained_optimizer_updates=18000,question_updates=13500,copy_rehearsal_updates=4500,
        optimizer_counters=counters,frozen_bridge_tensor_count=len(frozen),
        frozen_encoder_decoder_and_normalization_bit_identical=True,changed_coordinates=changed,
        nonzero_contextual_up_coordinates=up_nonzero,trained_body_parameters=trained_body,
        trained_bridge_parameters=trained_head,readout_only_control=is_control,
        all_checkpoint_and_optimizer_tensors_finite=True,neural_forward_calls=0,CUDA_initialized=False,
        language_accuracy_proven=False,full_cross_attempt_compute_cost_not_inferred=True))
    print(json.dumps(dict(global_updates=18000,question_updates=13500,copy_updates=4500,
        frozen_bridge_bit_identical=True,trained_body_parameters=trained_body,trained_bridge_parameters=trained_head)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
