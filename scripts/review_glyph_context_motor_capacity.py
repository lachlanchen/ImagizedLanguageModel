#!/usr/bin/env python3
"""Independently recount all128 cached-state glyphs and make complete raw boards."""
import argparse
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.diagnose_glyph_context_motor import validate
from scripts.summarize_glyph_tail_first_writing import recount, same
from scripts.train_glyph_context_motor import load
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/review_glyph_context_motor_capacity.py'
ROUTES=dict(candidate='candidate_ink',parent='parent_ink',codec='supplied_ink')


def aggregate(rows):
    scores=np.asarray([r['foreground_f1'] for r in rows])
    return dict(cases=len(rows),exact=sum(r['pixel_exact'] for r in rows),
        mean_foreground_f1=float(scores.mean()) if len(rows) else None,
        median_foreground_f1=float(np.median(scores)) if len(rows) else None,
        at_least_095=int((scores>=.95).sum()))


def main(run,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only independent pixel recount')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'Commit raw review implementation first')
    torch.set_num_threads(2); registration=validate(run); execution=run/'execution'
    report=load(execution/'report.json')
    require(load(execution/'status.json')['phase']=='complete'
        and report['registration_sha256']==sha(run/'registration.json')
        and report['pixels_sha256']==sha(execution/'pixels.npz')
        and report['raw_sha256']==sha(execution/'raw.pt'),'Completed diagnostic identity changed')
    require(not output.exists(),'Preserve earlier raw reviews')
    data=torch.load(run/'prepared.pt',map_location='cpu',weights_only=True)
    raw=torch.load(execution/'raw.pt',map_location='cpu',weights_only=True)
    require(torch.equal(raw['target_motor'],data['target_motor']),'Supplied motor targets changed')
    archive=np.load(execution/'pixels.npz',allow_pickle=False)
    require(set(archive.files)=={'gold_ink',*ROUTES.values()},'Unexpected pixel archive')
    arrays={name:archive[name] for name in archive.files}
    for name,a in arrays.items():
        require(a.shape==(128,1,32,32) and a.dtype==np.uint8 and np.isin(a,[0,1]).all(),
                'Missing or nonbinary raw tiles: '+name)
        require(np.array_equal(a,raw[name].numpy()),'Raw tensor/archive pixel mismatch')
    require(np.array_equal(arrays['gold_ink'],data['gold_ink'].numpy()),'Original gold pixels changed')
    counters={route:recount(arrays[name],arrays['gold_ink'],
        [r.get('actual_first_present',True) for r in registration['positions']] if route=='candidate' else None)
        for route,name in ROUTES.items()}
    records=[]
    for index,(position,record) in enumerate(zip(registration['positions'],report['records'])):
        require(all(record[k]==v for k,v in position.items()),'Diagnostic position relabelled')
        for route in ROUTES:same(counters[route][index],record['scores'][route])
        target=(data['target_motor'][index]-data['motor_mean'])/data['motor_scale']
        for route in ['candidate','parent']:
            value=raw[route+'_normalized_motor'][index]
            mse=float((value-target).square().mean())
            require(math.isclose(mse,record[route+'_normalized_motor_MSE'],rel_tol=1e-7,abs_tol=1e-9),
                    'Normalized motor score changed')
        if not position['group'].startswith('train_'):
            for name in ['candidate_ink','parent_ink','supplied_ink']:
                require(torch.equal(raw[name][index],data[name][index]),'Retained development image changed')
        records.append(dict(**position,scores={r:counters[r][index] for r in ROUTES},
            stop_probability=record['stop_probability'],
            candidate_normalized_motor_MSE=record['candidate_normalized_motor_MSE'],
            parent_normalized_motor_MSE=record['parent_normalized_motor_MSE']))
    require(len(report['records'])==len(records)==128,'Keep all128 registered positions')
    groups={};subgroups={}
    for group in ['train_first','train_later','fixed_cells','proportional']:
        selected=[r for r in records if r['group']==group];require(len(selected)==32,'Incomplete group')
        groups[group]={route:aggregate([r['scores'][route] for r in selected]) for route in ROUTES}
        if group.startswith('train_'):continue
        subgroups[group]={str(agree):{route:aggregate([r['scores'][route] for r in selected
            if r['native_canonical_first_agree']==agree]) for route in ROUTES} for agree in [False,True]}
    gates={g:(groups[g]['candidate']['median_foreground_f1']>=.99
        and groups[g]['candidate']['at_least_095']>=30) for g in ['train_first','train_later']}
    output.mkdir(parents=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    boards=[];columns=[('Target','gold_ink'),('Parent','parent_ink'),('Motor','candidate_ink'),('Codec','supplied_ink')]
    for start in range(0,128,16):
        picture=Image.new('L',(1040,60+16*116),245);draw=ImageDraw.Draw(picture)
        for col,(label,_) in enumerate(columns):draw.text((575+col*116,10),label,font=font,fill=0)
        tiles=[]
        for index in range(start,start+16):
            row=records[index];y=50+(index-start)*116
            draw.text((12,y+9),row['id'],font=font,fill=0)
            draw.text((12,y+36),f"stop p={row['stop_probability']:.5g}",font=font,fill=0)
            if 'native_canonical_first_agree' in row:
                draw.text((12,y+63),f"native prefix agrees: {row['native_canonical_first_agree']}",font=font,fill=0)
            for col,(label,key) in enumerate(columns):
                tile=Image.fromarray(255-255*arrays[key][index,0]).resize((96,96),Image.Resampling.NEAREST)
                x=575+col*116;picture.paste(tile,(x,y))
                tiles.append(dict(position=index,array=key,x=x,y=y,width=96,height=96))
                if col:draw.text((x,y+97),f"{row['scores'][label.lower() if label!='Motor' else 'candidate']['foreground_f1']:.4f}",font=font,fill=0)
        path=output/f'board_{start//16+1:02}.png';picture.save(path)
        boards.append(dict(file=path.name,sha256=sha(path),tiles=tiles))
    summary=dict(schema='motor-capacity-learning-v1',source_sha256=sha(ROOT/OWN),
        registration_sha256=sha(run/'registration.json'),execution_report_sha256=sha(execution/'report.json'),
        raw_pixels_sha256=sha(execution/'pixels.npz'),groups=groups,native_prefix_subgroups=subgroups,
        training_writing_gates=gates,all128_pixel_counts_independently_verified=True,
        original_development_images_unchanged=True,records=records,boards=boards,
        supplied_state_one_glyph_diagnosis_not_language=True,canonical_pixel_mismatch_not_character_error=True,
        new_optimizer_updates=0,new_contextual_body_calls=0,
        resources={k:report[k] for k in ['elapsed_seconds','resource_wait_seconds','peak_allocated_vram_gib','minimum_available_gib']},
        visual_review_complete=False,whole_language_scores_unchanged=True,goal_achieved=False)
    save_json(output/'summary.json',summary)
    require(not torch.cuda.is_initialized(),'CPU review initialized CUDA')
    print({'groups':groups,'training_gates':gates,'boards_to_view':8})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_context_motor_capacity_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/glyph_context_motor_capacity_learning_v1')
    args=parser.parse_args();main(args.run.resolve(),args.output.resolve())
