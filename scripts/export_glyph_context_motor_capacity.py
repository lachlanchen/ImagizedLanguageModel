#!/usr/bin/env python3
"""Verify512 displayed raster tiles and export the complete conditional diagnosis."""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.diagnose_glyph_context_motor import validate,CACHE
from scripts.score_pixel_language_first_blocks import require
from scripts.train_glyph_context_motor import load

OWN='scripts/export_glyph_context_motor_capacity.py'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only final diagnostic export')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'Commit export implementation first')
    run=ROOT/'artifacts/glyph_context_motor_capacity_v1';review=ROOT/'artifacts/glyph_context_motor_capacity_learning_v1'
    registration=validate(run);result=load(review/'summary.json');visual=load(review/'visual_review.json')
    require(visual['summary_sha256']==sha(review/'summary.json') and visual['all8_boards_actually_viewed']
        and visual['all128_positions_and512_displayed_tiles_viewed']
        and visual['board_files']==[b['file'] for b in result['boards']],'Complete original tile review first')
    archive=np.load(run/'execution/pixels.npz',allow_pickle=False);count=0
    for board in result['boards']:
        require(sha(review/board['file'])==board['sha256'],'Viewed board changed')
        with Image.open(review/board['file']) as image:
            for tile in board['tiles']:
                x,y,w,h=(tile[k] for k in ['x','y','width','height'])
                actual=np.asarray(image.crop((x,y,x+w,y+h)))
                expected=np.repeat(np.repeat(255-255*archive[tile['array']][tile['position'],0],3,0),3,1)
                require(np.array_equal(actual,expected),'Displayed pixels differ from saved raw output')
                count+=1
    require(count==512,'Missing displayed position or route')
    torch.set_num_threads(2)
    pools=torch.load(CACHE/'pools.pt',map_location='cpu',weights_only=True)
    first=torch.bincount(pools['first_labels'],minlength=4087)
    later=torch.bincount(pools['later_labels'],minlength=4087)
    coverage=[]
    for row in result['records']:
        if row['group'] not in ['fixed_cells','proportional']:continue
        index=row['primitive_index']
        coverage.append(dict(case=row['id'],character=row['canonical_first_character'],
            first_training_positions=int(first[index]),later_training_positions=int(later[index]),
            target_absent_from_conditional_training=not bool(first[index]+later[index])))
    require(len(coverage)==64,'All64 development target coverage counts required')
    public=ROOT/'references/evidence/glyph_context_motor_capacity_20261111'
    require(not public.exists(),'Preserve previous complete export')
    public.mkdir()
    for board in result['boards']:shutil.copy2(review/board['file'],public/board['file'])
    shutil.copy2(run/'execution/pixels.npz',public/'pixels.npz')
    evidence=dict(result,visual_review_complete=True,visual_review=visual,all512_displayed_tiles_bit_exact=True,
        export_source_sha256=sha(ROOT/OWN),original_summary_sha256=sha(review/'summary.json'),
        conditional_target_coverage=coverage,conditional_training_unique_forms=dict(
            first=int((first>0).sum()),later=int((later>0).sum()),combined=int((first+later>0).sum())),
        conditional_target_coverage_is_not_codec_inventory_coverage=True,
        first_pool_diagnostic_count=32,later_pool_diagnostic_count=32,
        first_pool_F1_gate_passed=result['training_writing_gates']['train_first'],
        later_pool_F1_gate_passed=result['training_writing_gates']['train_later'],
        no_new_fit_authorized_by_diagnostic_alone=True,CPU_CUDA_initialized=torch.cuda.is_initialized())
    save_json(ROOT/'references/evidence/glyph_context_motor_capacity_learning_20261111.json',evidence)
    save_json(review/'verification.json',dict(all128_records_independently_recounted=True,
        all512_displayed_tiles_bit_exact=True,visual_review_complete=True,
        public_evidence_sha256=sha(ROOT/'references/evidence/glyph_context_motor_capacity_learning_20261111.json'),
        first_writing_gate=result['training_writing_gates']['train_first'],
        later_writing_gate=result['training_writing_gates']['train_later'],
        conditional_training_unique_forms=evidence['conditional_training_unique_forms'],
        whole_language_scores_unchanged=True,goal_achieved=False))
    require(not torch.cuda.is_initialized(),'CPU export initialized CUDA')
    print(dict(displayed_tiles=count,training_gates=result['training_writing_gates'],
        public_evidence_sha256=sha(ROOT/'references/evidence/glyph_context_motor_capacity_learning_20261111.json')))


if __name__=='__main__':main()
