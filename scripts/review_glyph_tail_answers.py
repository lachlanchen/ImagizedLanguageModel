#!/usr/bin/env python3
"""Authenticate and inspect every adapted-model stream before reference scoring."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require


def load(path):
    return json.loads(Path(path).read_text())


def review(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU review requires hidden CUDA')
    import torch
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    require(load(run/'assessment/status.json')['phase']=='completed','finish every complete stream first')
    report=load(run/'assessment/report.json'); require(len(report['records'])==72,'incomplete fixed cohort')
    folder=run/'raw_review'; folder.mkdir(exist_ok=False)
    for row in report['records']:
        path=run/'assessment'/row['file']; require(sha(path)==row['sha256'],'raw tensor changed')
        v=torch.load(path,weights_only=True,map_location='cpu'); pixels=v['images']
        require(tuple(pixels.shape)==(row['glyphs'],1,32,32)
                and bool(((pixels==-1)|(pixels==1)).all()),'invalid whole-glyph pixels')
        require(len(v['states'])==len(v['stop_probabilities'])==len(pixels)+1
                and len(v['feedback'])==len(pixels),'causal decision count differs')
        require(all(p<.5 for p in v['stop_probabilities'][:-1])
                and (v['stop_probabilities'][-1]>=.5)==row['stopped']
                and row['stopped']!=row['capped'],'recorded stop/cap differs')
        require(not row['capped'] or len(pixels)==64,'wrong autonomous cap')
        actual=np.full((32,max(1,len(pixels))*32),255,np.uint8)
        for i,tile in enumerate(pixels):
            require(tensor_hash(tile[None])==v['feedback'][i]['image_sha256'],'actual feedback image changed')
            actual[:,i*32:(i+1)*32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
        png=run/'assessment'/row['png']; require(sha(png)==row['png_sha256'],'display PNG changed')
        require(np.array_equal(actual,np.asarray(Image.open(png).convert('L'))),'PNG does not show actual pixels')
    boards=[]
    for split,route in [('train','normal'),('development','normal'),('development','blank'),('fresh','normal')]:
        group=[r for r in report['records'] if r['key'].split('/')[::2]==[split,route]]
        for start in range(0,len(group),4):
            rows=group[start:start+4]
            width=max(Image.open(run/'assessment'/r['png']).width for r in rows)
            canvas=Image.new('L',(max(640,width+12),len(rows)*70),245); draw=ImageDraw.Draw(canvas)
            for j,row in enumerate(rows):
                draw.text((6,j*70+3),row['key']+f" n={row['glyphs']} stop={row['stopped']} cap={row['capped']}",fill=0)
                canvas.paste(Image.open(run/'assessment'/row['png']).convert('L'),(6,j*70+25))
            name=f'{split}_{route}_{start//4:02}.png'; canvas.save(folder/name)
            boards.append(dict(file=name,sha256=sha(folder/name)))
    require(len(boards)==18,'unexpected board count')
    save_json(folder/'report.json',dict(assessment_report_sha256=sha(run/'assessment/report.json'),
        records=report['records'],boards=boards,evaluator_references_read=False,
        all_output_PNG_and_feedback_image_hashes_verified=True,
        feedback_embeddings_cold_recomputed=False,output_pixels_repaired=False))
    print(json.dumps(dict(streams=72,boards=18,references_read=False)))


def score(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU scorer requires hidden CUDA')
    folder=run/'raw_review'; report=load(folder/'report.json'); manual=load(folder/'manual_review.json')
    require(manual['raw_review_report_sha256']==sha(folder/'report.json')
            and manual['all72_actual_streams_viewed'] and manual['all18_boards_actually_viewed'],
            'record actual complete output-first review before scoring')
    for row in report['boards']: require(sha(folder/row['file'])==row['sha256'],'viewed board changed')
    require(set(manual['output_first_readings'])=={r['key'] for r in report['records']},'retain readings/uncertainty for every stream')
    m=load(run/'preparation.json')
    require(sha(run/'evaluation_references.json')==m['evaluation_references_sha256'],'fixed new references changed')
    expected=load(run/'evaluation_references.json')
    source={split:{r['id']:r for r in map(json.loads,(ROOT/f'data/visual_qa/cmrc2018_short/{split}.jsonl').read_text().splitlines())}
            for split in ['train','development']}
    controls=load(ROOT/'artifacts/semantic_glyph_language_v1/preparation.json')['jobs'][:8]
    old=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')['development'][:16]
    for i,row in enumerate(controls):
        expected[f'train/{i:04}/normal']=[r['text'] for r in source['train'][Path(row['source']).stem]['references']]
    for i,row in enumerate(old):
        for route in ['normal','blank']:
            expected[f'development/{i:04}/{route}']=[r['text'] for r in source['development'][row['id']]['references']]
    import torch
    torch.set_num_threads(2)
    primitive=ROOT/m['primitive']
    data=torch.load(primitive/'training.pt',weights_only=True,map_location='cpu')
    charmap={c:i for i,row in enumerate(load(primitive/'inventory.json')['records']) for c in row['chars']}
    measurements=[]
    for row in report['records']:
        path=run/'assessment'/row['file']; require(sha(path)==row['sha256'],'scored raw tensor changed')
        v=torch.load(path,weights_only=True,map_location='cpu')
        actual=(v['images']<0).to(torch.uint8)[:,0].numpy()
        readings=manual['output_first_readings'][row['key']]
        require(isinstance(readings['uncertain'],bool)
                and (readings['text'] is None or isinstance(readings['text'],str)),'invalid manual reading')
        scores=[]; unsupported=[]
        for text in expected[row['key']]:
            missing=sorted(set(text)-charmap.keys())
            if missing:
                unsupported.append(dict(text=text,missing=missing)); continue
            gold=np.stack([data['ink'][charmap[c],0].numpy() for c in text])
            n=max(len(actual),len(gold)); a=np.zeros((n,32,32),np.uint8); g=a.copy()
            a[:len(actual)]=actual; g[:len(gold)]=gold
            denominator=int(a.sum()+g.sum()); f1=2*int((a&g).sum())/denominator if denominator else 1.
            exact=bool(row['stopped'] and len(actual)==len(gold) and np.array_equal(a,g))
            scores.append(dict(reference=text,foreground_f1=f1,canonical_complete_exact=exact))
        measurements.append(dict(**row,references=expected[row['key']],canonical_reference_scores=scores,
            unsupported_reference_strings=unsupported,
            canonical_complete_exact=any(r['canonical_complete_exact'] for r in scores),
            best_foreground_f1=max((r['foreground_f1'] for r in scores),default=None),
            output_first_reading=readings,manual_clear_complete_exact=bool(row['stopped']
                and not readings['uncertain'] and readings['text'] in expected[row['key']])))
    groups={}
    for split,route in [('train','normal'),('development','normal'),('development','blank'),('fresh','normal')]:
        rows=[r for r in measurements if r['key'].split('/')[::2]==[split,route]]
        supported=[r['best_foreground_f1'] for r in rows if r['best_foreground_f1'] is not None]
        groups[split+'/'+route]=dict(cases=len(rows),canonical_complete_exact=sum(r['canonical_complete_exact'] for r in rows),
            manual_clear_complete_exact=sum(r['manual_clear_complete_exact'] for r in rows),
            mean_foreground_f1_supported=float(np.mean(supported)) if supported else None,
            cases_with_supported_reference=len(supported),stopped=sum(r['stopped'] for r in rows),
            capped=sum(r['capped'] for r in rows))
    fresh={int(r['key'].split('/')[1]):r for r in measurements if r['key'].startswith('fresh/')}
    pairs=sum(all(fresh[i]['manual_clear_complete_exact'] for i in [2*p,2*p+1]) for p in range(16))
    pilot=groups['fresh/normal']['manual_clear_complete_exact']>=24 and pairs>=10
    save_json(folder/'scores.json',dict(records=measurements,groups=groups,fresh_both_questions_correct_contexts=pairs,
        pilot_complete_answer_requirement_passed=pilot,
        fixed_component_gate_passed=load(run/'qualify/report.json')['readout_gate_passed'],
        general_ILM_goal_achieved=False,independent_human_grading=False,
        inherited_pretraining_exposure_unknown=True,checkpoint_selection_performed=False,
        runtime_outputs_replaced=False,approximate_nearest_glyph_decoding_used=False,
        manual_review_sha256=sha(folder/'manual_review.json')))
    print(json.dumps(dict(groups=groups,both_questions_correct=pairs,pilot=pilot),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['review','score'],required=True)
    parser.add_argument('--run',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    args=parser.parse_args(); {'review':review,'score':score}[args.stage](args.run.resolve())
