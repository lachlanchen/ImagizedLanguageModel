#!/usr/bin/env python3
"""Independently inspect actual writer tensors and AdamW counters, with no forward call."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/audit_balanced_glyph_writer.py'


def main(run,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU checkpoint audit requires hidden CUDA')
    require(not output.exists(),'preserve previous checkpoint audits')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'commit final tensor auditor first')
    from scripts.train_balanced_glyph_writer import verify,PARENT_SHA
    m=verify(run);report=json.loads((run/'fit/report.json').read_text())
    require(json.loads((run/'fit/status.json').read_text())['phase']=='completed','finish the writer fit')
    path=run/'fit/checkpoint_latest.pt';require(sha(path)==report['checkpoint_sha256'],'final writer changed')
    require(sha(ROOT/m['initial_checkpoint'])==PARENT_SHA,'inherited parent changed')
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
    torch.set_num_threads(2)
    value=torch.load(path,weights_only=True,map_location='cpu')
    parent=torch.load(ROOT/m['initial_checkpoint'],weights_only=True,map_location='cpu')
    require(value['step']==value['new_optimizer_updates']==18000 and value['inherited_optimizer_updates']==18000
            and value['language_training_complete'],'completed stage counters required')
    require(value['preparation_sha256']==sha(run/'preparation.json'),'checkpoint prepared-data identity differs')
    with torch.device('meta'):model=SemanticGlyphBridge()
    trained={n:p for n,p in model.named_parameters() if n.startswith(('readout.','stop.'))}
    require(set(value['model'])==set(parent['model'])==set(model.state_dict()),'writer architecture differs')
    frozen=[];changed=[]
    for name,tensor in value['model'].items():
        require(torch.isfinite(tensor).all() and tensor.shape==model.state_dict()[name].shape,'invalid actual model tensor')
        if name in trained:
            if not torch.equal(tensor,parent['model'][name]):changed.append(name)
        else:
            require(torch.equal(tensor,parent['model'][name]),'frozen image module or normalization changed')
            frozen.append(name)
    require(set(value['body_tail_adapters'])==set(parent['body_tail_adapters'])
            and all(torch.equal(v,parent['body_tail_adapters'][n]) for n,v in value['body_tail_adapters'].items()),
            'inherited contextual adapters changed')
    optimizer=value['optimizer'];require(len(optimizer['param_groups'])==1,'unexpected optimizer groups')
    ids=optimizer['param_groups'][0]['params'];require(len(ids)==len(trained) and set(ids)==set(optimizer['state']),
        'optimizer must own exactly the writer and stop tensors')
    counters={}
    for (name,param),index in zip(trained.items(),ids):
        state=optimizer['state'][index];counter=int(state['step']);expected=18000 if name.startswith('readout.') else 13500
        require(counter==expected,'actual AdamW update count differs: '+name)
        for key in ['exp_avg','exp_avg_sq']:
            require(state[key].shape==param.shape and torch.isfinite(state[key]).all(),'invalid optimizer moment')
        counters[name]=counter
    require(len(changed)==len(trained),'some intended trained tensor did not change')
    require(report['sampled_target_counts']==dict(first=864000,later=864000,terminal=864000,copy=576000),
            'fixed sampled-target receipt differs')
    require(not torch.cuda.is_initialized(),'CPU auditor initialized CUDA')
    save_json(output,dict(auditor_sha256=sha(ROOT/OWN),checkpoint_sha256=sha(path),
        inherited_checkpoint_sha256=PARENT_SHA,preparation_sha256=sha(run/'preparation.json'),
        actual_AdamW_counters=counters,changed_writer_tensors=changed,
        frozen_image_and_normalization_tensors_bit_exact=frozen,inherited_body_adapters_bit_exact=True,
        trained_parameters=sum(p.numel() for p in trained.values()),all_model_and_optimizer_tensors_finite=True,
        new_writer_updates=18000,new_question_updates=13500,new_copy_updates=4500,inherited_updates=18000,
        sampler_trace_verified=False,sampled_target_counts=report['sampled_target_counts'],
        neural_forward_calls=0,optimizer_updates=0,CUDA_initialized=False,language_accuracy_proven=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/balanced_glyph_writer_v1'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run.resolve(),a.output.resolve())
