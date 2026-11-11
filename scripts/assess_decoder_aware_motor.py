#!/usr/bin/env python3
"""Test the final decoder-aware EMA on the unchanged128 cached glyph positions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
from scripts.audit_glyph_tail_first_writing import pixels_metrics
from scripts.review_glyph_context_motor_capacity import aggregate
from scripts.summarize_glyph_tail_first_writing import recount,same
from scripts.train_decoder_aware_motor import validate,CAPACITY,CHECKPOINT,Monitor,load,bridge_gpu,numerics,atomic_tensor
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/assess_decoder_aware_motor.py'
RUN=ROOT/'artifacts/glyph_decoder_aware_motor_v1'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Physical GPU0 cached-state assessment only')
    registration=validate(RUN)
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'Commit assessment source before sampling')
    fit=load(RUN/'fit/report.json');checkpoint=RUN/'fit/checkpoint_latest.pt'
    require(fit['training_complete'] and fit['actual_optimizer_updates']==12000
        and sha(checkpoint)==fit['checkpoint_sha256']
        and load(RUN/'fit/status.json')['phase']=='complete','Completed final fit required')
    previous=load(CAPACITY/'execution/report.json');positions=load(CAPACITY/'registration.json')['positions']
    require(len(positions)==128,'Keep all128 fixed diagnostic positions')
    output=RUN/'assessment';output.mkdir(exist_ok=False)
    monitor=Monitor(output);monitor.seconds=600;monitor.wait_allowance=120;monitor.status('admitting')
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission),torch.no_grad():
            numerics();torch.set_num_threads(2);monitor.guard()
            prepared=torch.load(CAPACITY/'prepared.pt',map_location='cpu',weights_only=True)
            old=torch.load(CAPACITY/'execution/raw.pt',map_location='cpu',weights_only=True)
            arrays={k:old[k].clone() for k in ['gold_ink','parent_ink','supplied_ink']}
            arrays['control_ink']=old['candidate_ink'].clone();arrays['candidate_ink']=torch.zeros_like(old['candidate_ink'])
            parent=SemanticGlyphBridge().to(device).eval().requires_grad_(False)
            parent.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)['model'])
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            require(payload['step']==12000 and payload['training_complete'],'Final EMA checkpoint required')
            flow=ContextMotorFlow().to(device).eval().requires_grad_(False);flow.load_state_dict(payload['ema'])
            require(torch.equal(payload['motor_mean'],prepared['motor_mean'])
                and torch.equal(payload['motor_scale'],prepared['motor_scale']),'Physical control normalization changed')
            del payload
            versions=[(p,p._version) for m in [parent,flow] for p in m.parameters()]
            mean,scale=prepared['motor_mean'].to(device),prepared['motor_scale'].to(device)
            records=[];motors=[];noises=[]
            for index,position in enumerate(positions):
                monitor.step=index;monitor.guard();monitor.status('sampling_fixed_cached_state',position=position['id'])
                h=prepared['states'][index:index+1].to(device)
                generator=torch.Generator(device=device).manual_seed(position['seed'])
                normalized,noise=flow.sample(h,generator=generator)
                motor=normalized*scale+mean;logits=parent.decoder(motor.reshape(1,64,4,4)).float()
                arrays['candidate_ink'][index]=(logits.sigmoid()>=.5).to(torch.uint8).cpu()[0]
                motors.append(normalized.cpu().flatten());noises.append(noise.cpu())
                require(torch.equal(noise.cpu(),old['noise'][index]),'Paired initial noise differs')
                stop=float(parent.stop(h).sigmoid().flatten()[0])
                require(stop==previous['records'][index]['stop_probability'],'Frozen stopping control differs')
                scores={name:pixels_metrics(arrays[key][index].numpy(),arrays['gold_ink'][index].numpy())
                    for name,key in [('candidate','candidate_ink'),('control','control_ink'),('parent','parent_ink'),('codec','supplied_ink')]}
                target=(prepared['target_motor'][index]-prepared['motor_mean'])/prepared['motor_scale']
                records.append(dict(**position,scores=scores,stop_probability=stop,
                    candidate_normalized_motor_MSE=float((motors[-1]-target).square().mean())))
            monitor.guard();require(all(p._version==v for p,v in versions),'Frozen assessment parameter changed')
            atomic_tensor(output/'raw.pt',dict(**arrays,normalized_motor=torch.stack(motors),noise=torch.stack(noises)))
            np.savez_compressed(output/'pixels.npz',**{k:v.numpy() for k,v in arrays.items()})
            # Independent vectorized contingency counts, not saved scalar scores.
            for route,key in [('candidate','candidate_ink'),('control','control_ink'),('parent','parent_ink'),('codec','supplied_ink')]:
                counts=recount(arrays[key].numpy(),arrays['gold_ink'].numpy())
                for index,row in enumerate(counts):same(row,records[index]['scores'][route])
            groups={}
            for group in ['train_first','train_later','fixed_cells','proportional']:
                rows=[r for r in records if r['group']==group];require(len(rows)==32,'Incomplete physical group')
                groups[group]={route:aggregate([r['scores'][route] for r in rows]) for route in ['candidate','control','parent','codec']}
            gates={g:groups[g]['candidate']['median_foreground_f1']>=.99
                and groups[g]['candidate']['at_least_095']>=30 for g in ['train_first','train_later']}
            save_json(output/'report.json',dict(schema='decoder-aware-motor-capacity-v1',source_sha256=sha(ROOT/OWN),
                registration_sha256=sha(RUN/'registration.json'),fit_report_sha256=sha(RUN/'fit/report.json'),
                checkpoint_sha256=sha(checkpoint),original_positions_sha256=sha(CAPACITY/'registration.json'),
                original_control_report_sha256=sha(CAPACITY/'execution/report.json'),records=records,groups=groups,
                training_writing_gates=gates,full216_language_generation_eligible=all(gates.values()),
                raw_sha256=sha(output/'raw.pt'),pixels_sha256=sha(output/'pixels.npz'),
                all128_positions_scores_independently_recounted=True,paired_noise_and_stopping_bit_exact=True,
                new_optimizer_updates=0,new_contextual_body_calls=0,one_glyph_regardless_of_stop=True,
                whole_language_scores_unmeasured_for_new_model=True,goal_achieved=False,admission=admission,**monitor.receipt()))
        # Labels are outside neural execution. Every row is retained for visual review.
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);boards=[]
        columns=[('Target','gold_ink'),('Control','control_ink'),('New','candidate_ink'),('Codec','supplied_ink')]
        for first in range(0,128,16):
            surface=Image.new('L',(1040,1916),245);draw=ImageDraw.Draw(surface);tiles=[]
            for col,(label,_) in enumerate(columns):draw.text((575+116*col,10),label,font=font,fill=0)
            for index in range(first,first+16):
                row=records[index];y=50+(index-first)*116
                draw.text((12,y+9),row['id'],font=font,fill=0)
                draw.text((12,y+36),f"stop p={row['stop_probability']:.5g}",font=font,fill=0)
                for col,(label,key) in enumerate(columns):
                    x=575+116*col;tile=Image.fromarray(255-255*arrays[key][index,0].numpy())
                    surface.paste(tile.resize((96,96),Image.Resampling.NEAREST),(x,y))
                    tiles.append(dict(position=index,array=key,x=x,y=y,width=96,height=96))
                    if col:
                        route={'Control':'control','New':'candidate','Codec':'codec'}[label]
                        draw.text((x,y+97),f"{row['scores'][route]['foreground_f1']:.4f}",font=font,fill=0)
            path=output/f'board_{first//16+1:02}.png';surface.save(path)
            boards.append(dict(file=path.name,sha256=sha(path),tiles=tiles))
        save_json(output/'display_manifest.json',dict(boards=boards,positions=128,displayed_tiles=512,
            report_sha256=sha(output/'report.json'),actual_visual_review_complete=False))
        monitor.status('complete')
        print(json.dumps(dict(groups=groups,gates=gates,full216_language_generation_eligible=all(gates.values()))))
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:save_json(output/'status.json',dict(load(output/'status.json'),pid=None))


if __name__=='__main__':main()
