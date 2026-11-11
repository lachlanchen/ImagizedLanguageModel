#!/usr/bin/env python3
"""CPU optimizer, full-target and resource audit of the real36-layer preflight."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.preflight_visual_continuation import RUN,DATA,validate,load,CHECKPOINT,CONTROLS
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/audit_visual_continuation_preflight.py'
PUBLIC=ROOT/'references/evidence/visual_continuation_preflight_20261112'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU audit only')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),'Commit audit first')
    require(not PUBLIC.exists(),'Preserve completed audit')
    torch.set_num_threads(2);r=validate(RUN);folder=RUN/'execution';report=load(folder/'report.json')
    require(report['registration_sha256']==sha(RUN/'registration.json') and report['all_checks_passed'],'Complete registered execution required')
    require(sha(folder/'checkpoint.pt')==report['checkpoint_sha256'],'Actual checkpoint differs')
    c=torch.load(folder/'checkpoint.pt',weights_only=True,map_location='cpu');w=c['trainable_weights'];opt=c['optimizer']
    require(c['preflight_only'] and c['optimizer_updates']==4 and not report['carry_preflight_updates_into_full_fit'],
            'Four discarded preflight updates required')
    ids=opt['param_groups'][0]['params'];names=list(w)
    require(len(ids)==len(names)==len(opt['state'])==85,'Missing optimizer tensors')
    moment_norms={}
    for name,index in zip(names,ids):
        state=opt['state'][index];require(int(state['step'])==4,'Nominal step count differs from Adam state')
        require(state['exp_avg'].shape==state['exp_avg_sq'].shape==w[name].shape,'Optimizer shape differs')
        require(w[name].dtype==state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32,'Wrong optimizer precision')
        require(torch.isfinite(w[name]).all() and torch.isfinite(state['exp_avg']).all() and
                torch.isfinite(state['exp_avg_sq']).all() and state['exp_avg_sq'].max()>0,'Invalid weight/moment')
        moment_norms[name]=float(state['exp_avg'].norm())
    require(sum(v.numel() for v in w.values())==report['trainable_parameters']==20212225,'Trainable count differs')
    baseline=torch.load(CHECKPOINT,weights_only=True,map_location='cpu',mmap=True)['model']
    control=torch.load(CONTROLS,weights_only=True,map_location='cpu');direct_changes={}
    for name,value in w.items():
        if name.startswith('bridge.'):initial=baseline[name.removeprefix('bridge.')]
        elif name=='dynamics.query':initial=control['after'][-1].float()
        elif name.endswith('.up'):initial=torch.zeros_like(value)
        else:continue
        count=int((value!=initial).sum());require(count==report['changed_coordinates'][name] and count>0,'Independent change count differs')
        direct_changes[name]=count
    require(len(direct_changes)==49,'Expected12writer/stop+query+36up checks')
    require(len(report['records'])==4 and [x['step'] for x in report['records']]==[1,2,3,4],'Incomplete updates')
    for step in report['records']:
        require(set(step['gradients'])==set(names) and all(np.isfinite(x) for x in step['gradients'].values()),'Gradient coverage differs')
        require(len(step['forward_and_recomputed_layer_calls'])==36 and set(step['forward_and_recomputed_layer_calls'])=={2},
                'All36 actual layers must run forward and recomputation')
        for i in range(36):
            require(step['gradients'][f'dynamics.adapters.{i}.up']>0,'Missing up gradient')
            if step['step']>1:require(step['gradients'][f'dynamics.adapters.{i}.down']>0,'Missing down gradient')
    targets=torch.load(RUN/'targets.pt',weights_only=True,map_location='cpu')
    corpus=VisualSequenceCorpus(DATA,split='train',expected_report_sha256=sha(DATA/'report.json'));cases=[]
    for job in r['cases']:
        source=corpus[job['source']['index']]['ink'];cut=job['prompt_glyphs'];gold=targets[job['name']]
        require(torch.equal(source[cut:],gold) and len(gold)==job['target_glyphs'] and tensor_hash(gold)==job['target_sha256'],
                'Full authentic target differs')
        prompt=np.asarray(Image.open(ROOT/job['prompt_image']).convert('L'));expected=np.full_like(prompt,255)
        for i,tile in enumerate(source[:cut]):expected[(i//32)*32:(i//32+1)*32,(i%32)*32:(i%32+1)*32]=255-255*tile[0].numpy()
        require(np.array_equal(prompt,expected),'Prompt pixels differ from original corpus')
        cases.append(dict(name=job['name'],source_id=job['source']['id'],prompt_glyphs=cut,complete_target_glyphs=len(gold)))
    require(sum(x['complete_target_glyphs'] for x in cases)==772,'Target exposure count differs')
    require(report['peak_allocated_vram_gib']<=22 and report['minimum_available_gib']>=24 and
            report['elapsed_seconds']<=1800 and report['resource_wait_seconds']<=300,'Registered resources exceeded')
    require(report['zero_adapter_native_split_max_abs']<=.002 and report['checks']['serial_parallel_max_abs']<=.002,
            'Numerical tolerance exceeded')
    require(all(v for k,v in report['checks'].items() if k!='serial_parallel_max_abs'),'Failed causal/prompt check')
    require(sha(folder/'unqualified_inference.pt')==report['unqualified_inference_sha256'],'Runtime exercise differs')
    output=torch.load(folder/'unqualified_inference.pt',weights_only=True,map_location='cpu');ink=((1-output['images'])/2).to(torch.uint8)
    require(torch.equal(output['images'],1-2*ink.float()) and len(output['feedback'])==len(ink), 'Invalid emitted pixels')
    for tile,feedback in zip(output['images'],output['feedback']):require(tensor_hash(tile[None])==feedback['image_sha256'],'Feedback not from emitted pixels')
    PUBLIC.mkdir(parents=True);canvas=Image.new('L',(max(384,len(ink)*32),80),255);draw=ImageDraw.Draw(canvas)
    draw.text((4,4),'Four-update runtime exercise; not a language result.',fill=0)
    if not len(ink):draw.text((4,36),'Empty output: learned stop fired before writing.',fill=0)
    for i,tile in enumerate(ink):canvas.paste(Image.fromarray(255-255*tile[0].numpy()),(32*i,36))
    canvas.save(PUBLIC/'unqualified_runtime.png')
    save_json(PUBLIC/'execution_report.json',report)
    value=dict(schema='visual-continuation-preflight-independent-audit-v1',source_sha256=sha(ROOT/OWN),
        registration_sha256=sha(RUN/'registration.json'),execution_report_sha256=sha(folder/'report.json'),
        checkpoint_sha256=report['checkpoint_sha256'],actual_Adam_steps_per_parameter=4,optimizer_parameters=85,
        trainable_parameters=20212225,all36_forward_and_recomputed=True,actual_gradient_norms_authenticated=True,
        direct_initial_final_change_counts=direct_changes,residual_down_initial_tensors_not_archived=True,
        residual_down_updates_evidenced_by_actual_gradients_and_Adam_moments=True,moment_norms=moment_norms,
        full_source_prefill_and_causal_checks_are_assertions_from_registered_GPU_execution=True,
        full_body_frozen_tensors_not_duplicated_for_a_second_comparison=True,cases=cases,complete_target_positions=772,
        resources={k:report[k] for k in ['elapsed_seconds','loading_seconds','resource_wait_seconds','peak_allocated_vram_gib','minimum_available_gib']},
        runtime_exercise_glyphs=len(ink),runtime_exercise_stopped=output['stopped'],runtime_pixels_sha256=sha(PUBLIC/'unqualified_runtime.png'),
        language_quality_evaluated=False,preflight_weights_not_qualified_or_used_for_full_fit=True,
        full_fit_registered=False,goal_achieved=False)
    save_json(PUBLIC/'audit.json',value)
    print(json.dumps(dict(audit_passed=True,trainable_parameters=20212225,full_target_positions=772,resources=value['resources']),indent=2))


if __name__=='__main__':main()
