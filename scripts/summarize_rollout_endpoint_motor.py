#!/usr/bin/env python3
"""Recount every saved pixel, motor error and board tile before publishing results."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import require
from scripts.train_rollout_endpoint_motor import RUN,validate,load


ROUTES=dict(candidate='candidate_ink',control='control_ink',parent='parent_ink',
            original_flow='original_flow_ink',codec='supplied_ink')


def main(reviewed=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only independent recount')
    torch.set_num_threads(2);r=validate(RUN);folder=RUN/'assessment'
    report=load(folder/'report.json');display=load(folder/'display_manifest.json')
    require(load(folder/'status.json')['phase']=='complete','Complete the actual assessment first')
    require(sha(folder/'report.json')==display['report_sha256'],'Display report differs')
    require(sha(folder/'raw.pt')==report['raw_sha256'] and sha(folder/'pixels.npz')==report['pixels_sha256'],
            'Raw assessment archive changed')
    require(report['registration_sha256']==sha(RUN/'registration.json'),'Unregistered assessment')
    raw=torch.load(folder/'raw.pt',map_location='cpu',weights_only=True)
    rows=report['records'];require(len(rows)==128,'All fixed positions required')
    prepared=torch.load(ROOT/'artifacts/glyph_context_motor_capacity_v1/prepared.pt',map_location='cpu',weights_only=True)
    scores={}
    with np.load(folder/'pixels.npz',allow_pickle=False) as archive:
        for route,key in ROUTES.items():
            require(np.array_equal(archive[key],raw[key].numpy()),'Tensor/NPZ pixel archive mismatch')
            values=[]
            for i,row in enumerate(rows):
                p,g=archive[key][i].astype(bool),archive['gold_ink'][i].astype(bool)
                tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
                f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
                source=row['scores'][route]
                require((tp,fp,fn)==(source['tp'],source['fp'],source['fn']) and
                    f==source['foreground_f1'] and np.array_equal(p,g)==source['pixel_exact'],
                    'Independent integer pixel recount differs')
                values.append(dict(f=f,exact=bool(np.array_equal(p,g))))
            scores[route]=values
        for i,row in enumerate(rows):
            target=(prepared['target_motor'][i]-prepared['motor_mean'])/prepared['motor_scale']
            for arm in ['control','rollout']:
                error=float((raw['normalized_motor'][arm][i]-target).square().mean())
                require(error==row['normalized_motor_MSE'][arm],'Independent motor recount differs')
        groups={}
        for name in ['train_first','train_later','fixed_cells','proportional']:
            idx=[i for i,row in enumerate(rows) if row['group']==name];require(len(idx)==32,'Incomplete fixed group')
            groups[name]={}
            for route,values in scores.items():
                fs=[values[i]['f'] for i in idx]
                observed=dict(cases=32,exact=sum(values[i]['exact'] for i in idx),
                    mean_foreground_f1=float(np.mean(fs)),median_foreground_f1=float(np.median(fs)),
                    at_least_095=sum(f>=.95 for f in fs))
                require(all(math.isclose(v,report['groups'][name][route][k],rel_tol=0,abs_tol=1e-12)
                    for k,v in observed.items()),'Aggregate recount differs')
                groups[name][route]=observed
        tiles=0
        for board in display['boards']:
            path=folder/board['file'];require(sha(path)==board['sha256'],'Board bytes changed')
            canvas=np.asarray(Image.open(path).convert('L'))
            for tile in board['tiles']:
                original=archive[tile['array']][tile['position'],0]
                expected=np.repeat(np.repeat(255-255*original,3,axis=0),3,axis=1)
                actual=canvas[tile['y']:tile['y']+96,tile['x']:tile['x']+96]
                require(np.array_equal(expected,actual),'Displayed tile differs from original pixel array')
                tiles+=1
        require(tiles==512 and len(display['boards'])==8,'Keep all eight boards and512 displayed originals')
    gates={route:{g:groups[g][route]['median_foreground_f1']>=.99 and
        groups[g][route]['at_least_095']>=30 for g in ['train_first','train_later']}
        for route in ['candidate','control']}
    require(gates==report['training_writing_gates'],'Independent gate decision differs')
    value=dict(report,all128_records_and_motor_errors_recounted_from_saved_files=True,
        all512_displayed_tiles_bit_exact=True,independent_group_recount=groups,
        verification_source_sha256=sha(Path(__file__)),CPU_CUDA_initialized=False,
        visual_review_complete=reviewed,independent_human_adjudication=False,
        canonical_pixel_agreement_not_language_accuracy=True,full_language_generation_run=False)
    if reviewed:
        # Keep raw process/session arguments in the private assessment receipt.
        admission=dict(value['admission']);inventory=admission.pop('project_processes',[])
        admission['project_process_inventory_count']=len(inventory)
        admission['raw_process_arguments_retained_privately']=True
        value['admission']=admission
        value['visual_review']=dict(boards_viewed=8,original_displayed_tiles=512,
            reviewer='Model-based labelled review with prior cohort familiarity',
            whole_language_quality_unmeasured=True)
        public=ROOT/'references/evidence/rollout_endpoint_motor_20261111';public.mkdir(exist_ok=False)
        for name in ['pixels.npz',*[b['file'] for b in display['boards']]]:shutil.copy2(folder/name,public/name)
        value['public_files_sha256']={p.name:sha(p) for p in public.iterdir() if p.is_file()}
        save_json(ROOT/'references/evidence/rollout_endpoint_motor_learning_20261111.json',value)
    save_json(RUN/'assessment_verification.json',value)
    print(json.dumps(dict(gates=gates,groups=groups,displayed_tiles_verified=512,
        visual_review_complete=reviewed,language_quality_unmeasured=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reviewed',action='store_true')
    args=parser.parse_args();main(args.reviewed)
