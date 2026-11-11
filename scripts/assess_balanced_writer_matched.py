#!/usr/bin/env python3
"""Frozen balanced-bridge diagnosis on all128 previously fixed actual states."""
import json
import os

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
from scripts.audit_glyph_tail_first_writing import pixels_metrics
from scripts.review_glyph_context_motor_capacity import aggregate
from scripts.summarize_glyph_tail_first_writing import recount,same
from scripts.compare_balanced_writer_matched import validate,RUN,CAPACITY,CHECKPOINT,CHECKPOINT_SHA,load
from scripts.train_glyph_context_motor import Monitor
from scripts.train_semantic_glyph_bridge import bridge_gpu,atomic_tensor
from scripts.train_semantic_glyph_language import numerics
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/assess_balanced_writer_matched.py'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 only')
    validate(RUN);positions=load(CAPACITY/'registration.json')['positions']
    previous=load(CAPACITY/'execution/report.json');require(len(positions)==128,'All fixed positions required')
    folder=RUN/'assessment';folder.mkdir(exist_ok=False)
    monitor=Monitor(folder);monitor.seconds=600;monitor.wait_allowance=120;monitor.status('admitting')
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission),torch.no_grad():
            numerics();torch.set_num_threads(2);monitor.guard()
            prepared=torch.load(CAPACITY/'prepared.pt',map_location='cpu',weights_only=True)
            old=torch.load(CAPACITY/'execution/raw.pt',map_location='cpu',weights_only=True)
            arrays={k:old[k].clone() for k in ['gold_ink','parent_ink','supplied_ink']}
            arrays['flow_ink']=old['candidate_ink'].clone();arrays['candidate_ink']=torch.zeros_like(old['candidate_ink'])
            bridge=SemanticGlyphBridge().to(device).eval().requires_grad_(False)
            bridge.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)['model'])
            frozen=[(p,p._version) for p in bridge.parameters()];records=[];motors=[]
            mean,scale=prepared['motor_mean'].to(device),prepared['motor_scale'].to(device)
            for i,position in enumerate(positions):
                monitor.step=i;monitor.guard();monitor.status('diagnosing_balanced_state',position=position['id'])
                h=prepared['states'][i:i+1].to(device)
                logits,semantic,stop=bridge.predict(h)
                arrays['candidate_ink'][i]=(logits.sigmoid()>=.5).to(torch.uint8).cpu()[0]
                normalized=(bridge.decoder_input(semantic).float()-mean)/scale;motors.append(normalized.cpu()[0])
                target=(prepared['target_motor'][i]-prepared['motor_mean'])/prepared['motor_scale']
                routes=dict(candidate='candidate_ink',parent='parent_ink',flow='flow_ink',codec='supplied_ink')
                scores={name:pixels_metrics(arrays[key][i].numpy(),arrays['gold_ink'][i].numpy()) for name,key in routes.items()}
                records.append(dict(**position,scores=scores,stop_probability=float(stop.sigmoid()[0]),
                    unchanged_parent_stop_probability=previous['records'][i]['stop_probability'],
                    normalized_motor_MSE=float((motors[-1]-target).square().mean())))
            require(all(p._version==v for p,v in frozen),'Frozen bridge changed')
            atomic_tensor(folder/'raw.pt',dict(**arrays,normalized_motor=torch.stack(motors)))
            np.savez_compressed(folder/'pixels.npz',**{k:v.numpy() for k,v in arrays.items()})
            for route,key in routes.items():
                for i,row in enumerate(recount(arrays[key].numpy(),arrays['gold_ink'].numpy())):same(row,records[i]['scores'][route])
            groups={}
            for group in ['train_first','train_later','fixed_cells','proportional']:
                rows=[r for r in records if r['group']==group];require(len(rows)==32,'Missing diagnostic group')
                groups[group]={route:aggregate([r['scores'][route] for r in rows]) for route in routes}
            gates={g:groups[g]['candidate']['median_foreground_f1']>=.99 and
                groups[g]['candidate']['at_least_095']>=30 for g in ['train_first','train_later']}
            save_json(folder/'report.json',dict(schema='balanced-writer-matched-diagnosis-v1',source_sha256=sha(ROOT/OWN),
                registration_sha256=sha(RUN/'registration.json'),checkpoint_sha256=CHECKPOINT_SHA,
                records=records,groups=groups,descriptive_training_writing_gates=gates,
                full_language_trial_previously_registered_regardless_of_this_diagnosis=True,
                raw_sha256=sha(folder/'raw.pt'),pixels_sha256=sha(folder/'pixels.npz'),admission=admission,
                independent_integer_pixel_recount_passed=True,all_frozen_parameter_versions_unchanged=True,
                one_glyph_regardless_of_stop=True,new_optimizer_updates=0,new_contextual_body_calls=0,
                language_quality_unmeasured=True,goal_achieved=False,**monitor.receipt()))
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);boards=[]
        columns=[('Target','gold_ink'),('Parent','parent_ink'),('Balanced','candidate_ink'),('Codec','supplied_ink')]
        for first in range(0,128,16):
            canvas=Image.new('L',(1040,1916),245);draw=ImageDraw.Draw(canvas);tiles=[]
            for col,(name,_) in enumerate(columns):draw.text((575+116*col,10),name,font=font,fill=0)
            for i in range(first,first+16):
                row=records[i];y=50+(i-first)*116
                draw.text((12,y+9),row['id'],font=font,fill=0)
                draw.text((12,y+36),f"stop p={row['stop_probability']:.5g}",font=font,fill=0)
                for col,(name,key) in enumerate(columns):
                    x=575+116*col;tile=Image.fromarray(255-255*arrays[key][i,0].numpy())
                    canvas.paste(tile.resize((96,96),Image.Resampling.NEAREST),(x,y))
                    tiles.append(dict(position=i,array=key,x=x,y=y,width=96,height=96))
                    if col:
                        route={'Parent':'parent','Balanced':'candidate','Codec':'codec'}[name]
                        draw.text((x,y+97),f"{row['scores'][route]['foreground_f1']:.4f}",font=font,fill=0)
            path=folder/f'board_{first//16+1:02}.png';canvas.save(path);boards.append(dict(file=path.name,sha256=sha(path),tiles=tiles))
        save_json(folder/'display_manifest.json',dict(boards=boards,positions=128,displayed_tiles=512,
            report_sha256=sha(folder/'report.json'),actual_visual_review_complete=False))
        monitor.status('complete');print(json.dumps(dict(groups=groups,gates=gates,whole_language_unmeasured=True)))
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':main()
