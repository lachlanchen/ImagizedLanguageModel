#!/usr/bin/env python3
"""Opaque complete-output review, original QA scoring and matched long-book scoring."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import os
import shutil
import tempfile

from PIL import Image,ImageDraw

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.continuation_training import RUN,load
from scripts.prepare_visual_continuation_training import validate
from scripts.train_visual_continuation import verify_generation
from scripts.review_compact_visual_span import score,normalized
from scripts.score_pixel_language_first_blocks import require

EVALS={'fixed_cells':ROOT/'artifacts/compact_visual_span_evaluation_v1',
       'proportional':ROOT/'artifacts/glyph_tail_input_presentation_v1/evaluation'}


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU review only');r=validate(run);review=run/'review'
    require(not review.exists(),'Preserve prior review');review.mkdir();pages=review/'pages';pages.mkdir();mapping={}
    for arm_folder in ['evaluation','parent_long_books']:
        report_path=run/arm_folder/'report.json';report=load(report_path);require(report['complete'],'Complete every registered output first')
        require(report['total_streams']==(708 if arm_folder=='evaluation' else 24),'Missing fixed streams')
        for arm in report['arms']:
            for rec in arm['records']:
                row=rec['input'];step=arm['step'];cap=rec['maximum_glyphs']
                parent=run/arm_folder/('long640' if arm_folder=='parent_long_books' else f'step_{step:06}/'+('core128' if cap==128 else 'long640'))
                verify_generation(parent/row['id'],row,sha(run/'preparation.json'),rec['checkpoint_sha256'],cap)
                opaque=hashlib.sha256((rec['checkpoint_sha256']+'/'+arm['arm']+'/'+row['id']).encode()).hexdigest()[:20]
                require(opaque not in mapping,'Opaque identity collision')
                os.link(parent/row['id']/'answer.png',pages/(opaque+'.png'))
                if '__book_' in row['id'] and row['condition']=='normal':os.link(ROOT/row['image'],pages/(opaque+'_prefix.png'))
                mapping[opaque]=dict(case=row['original_case'],layout=row['layout'],condition=row['condition'],step=step,
                    arm=arm['arm'],checkpoint_sha256=rec['checkpoint_sha256'],run_report_sha256=sha(report_path),
                    image_sha256=rec['image_sha256'],stopped=rec['stopped'],emitted_glyphs=rec['emitted_glyphs'],elapsed_seconds=rec['elapsed_seconds'])
    require(len(mapping)==732,'All660core+72long outputs required')
    save_json(review/'mapping.json',dict(registration_sha256=sha(run/'preparation.json'),rows=mapping))
    save_json(review/'readings_template.json',dict(evaluator=None,readings={k:dict(text=None,uncertain=None,coherent_continuation=None) for k in sorted(mapping)}))
    ids=sorted(mapping);boards=[]
    for start in range(0,len(ids),4):
        chosen=ids[start:start+4];images=[Image.open(pages/(k+'.png')).convert('L') for k in chosen]
        height=sum(max(40,im.height)+38 for im in images);canvas=Image.new('L',(1056,height),245);draw=ImageDraw.Draw(canvas);y=0
        for key,image in zip(chosen,images):draw.text((8,y+5),key,fill=0);canvas.paste(image,(8,y+28));y+=max(40,image.height)+38
        name=f'board_{start//4+1:03}.png';canvas.save(review/name);boards.append(dict(file=name,sha256=sha(review/name),ids=chosen))
    save_json(review/'boards.json',dict(boards=boards,pages=732,all_outputs_reviewed=False,prior_cohort_familiarity=True))


def score_review(run,reading_path):
    r=validate(run);review=run/'review';mapping=load(review/'mapping.json')['rows'];receipt=load(reading_path);readings=receipt['readings']
    require(set(mapping)==set(readings) and receipt.get('all_boards_actually_viewed') is True,'Review every complete output')
    require(receipt.get('mapping_sha256')==sha(review/'mapping.json'),'Freeze against actual output mapping')
    for key,value in readings.items():
        require(isinstance(value['text'],str) and type(value['uncertain']) is bool,'Complete reading or uncertainty flag required')
        require(sha(review/'pages'/(key+'.png'))==mapping[key]['image_sha256'],'Reviewed pixels changed')
    require(not (review/'frozen_readings.json').exists(),'Preserve frozen readings');shutil.copy2(reading_path,review/'frozen_readings.json')
    core={}
    for layout,path in EVALS.items():
        rows={k:v for k,v in mapping.items() if v['layout']==layout and v['arm']=='candidate_core128'}
        with tempfile.TemporaryDirectory(prefix='ilm-continuation-score-') as temp:
            tmp=Path(temp);(tmp/'pages').symlink_to((review/'pages').resolve())
            save_json(tmp/'mapping.json',dict(preparation_sha256=sha(path/'preparation.json'),rows=rows))
            save_json(tmp/'readings.json',dict(evaluator=receipt['evaluator'],readings={k:readings[k] for k in rows}))
            with redirect_stdout(io.StringIO()):score(path,tmp,tmp/'readings.json')
            value=load(tmp/'scores.json');save_json(review/(layout+'_core_scores.json'),value);core[layout]=value
    long=defaultdict(list)
    for key,row in mapping.items():
        if row['arm']=='candidate_core128':continue
        reading=readings[key];normal=row['condition']=='normal'
        if normal:require(type(reading['coherent_continuation']) is bool,'Judge continuation against visible book prefix')
        clear=not reading['uncertain'] and bool(normalized(reading['text'])) and row['emitted_glyphs']>0
        long[row['arm'],row['step'],row['condition']].append(dict(row,reading=reading,readable=clear,
            coherent_complete=clear and row['stopped'] and reading['coherent_continuation'] is True if normal else None))
    long_summary=[]
    for (arm,step,condition),rows in sorted(long.items()):
        require(len(rows)==12,'Keep all twelve book cases')
        long_summary.append(dict(arm=arm,step=step,condition=condition,cases=12,readable=sum(x['readable'] for x in rows),
            stopped=sum(x['stopped'] for x in rows),coherent_complete=sum(x['coherent_complete'] for x in rows) if condition=='normal' else None))
    final=r['total_steps'];p=next(x for x in core['proportional']['results'] if x['step']==final and x['condition']=='normal')
    erased=[x for x in core['proportional']['results'] if x['step']==final and x['condition']!='normal']
    retention={}
    for layout,required in r['exact_retention'].items():
        actual={x['case']:x for x in core[layout]['records'] if x['step']==final and x['condition']=='normal'}
        retention[layout]=dict(required=required,retained=[k for k in required if actual[k]['complete_exact']])
    books=next(x for x in long_summary if x['arm']=='candidate_long640' and x['step']==final and x['condition']=='normal')
    pilot=(p['complete_exact']>=12 and p['paired_both_correct']>=4 and p['readable']>=24 and books['coherent_complete']>=4 and
        all(x['complete_exact']<=2 for x in erased) and p['complete_exact']-max(x['complete_exact'] for x in erased)>=8 and
        all(x['required']==x['retained'] for x in retention.values()))
    save_json(review/'summary.json',dict(registration_sha256=sha(run/'preparation.json'),frozen_readings_sha256=sha(review/'frozen_readings.json'),
        core_results={k:v['results'] for k,v in core.items()},long_book_results=long_summary,retention=retention,pilot_passed=pilot,
        untouched_confirmation_required=True,independent_human_adjudication=False,prior_cohort_familiarity=True,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score']);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--readings',type=Path)
    a=p.parse_args();prepare(a.run) if a.action=='prepare' else score_review(a.run,a.readings)
