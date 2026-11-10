#!/usr/bin/env python3
"""Fixed-state motor diagnosis; no language generation or lexical repair."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.train_semantic_glyph_bridge import bridge_gpu,atomic_tensor
from scripts.train_semantic_glyph_language import load,numerics
from scripts import distill_semantic_glyph_readout as base
from scripts.train_robust_glyph_decoder import check
from scripts.score_pixel_language_first_blocks import require


def main(output,decoder):
    own='scripts/check_glyph_motor_states.py'
    require(subprocess.check_output(['git','show','HEAD:'+own],cwd=ROOT)==(ROOT/own).read_bytes(),'commit motor diagnostic first')
    meta=check(decoder);candidate=decoder/'fit/checkpoint_latest.pt'
    require(sha(candidate)==load(decoder/'fit/report.json')['checkpoint_sha256'],'candidate changed')
    require(not output.exists(),'preserve all earlier motor assessments')
    source=ROOT/meta['parent_run'];report=load(source/'assessment/report.json')
    primitive=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    inventory=load(ROOT/meta['primitive']/'inventory.json')['records'];chars={c:i for i,r in enumerate(inventory) for c in r['chars']}
    corpus=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    train={r['id']:r for r in map(json.loads,(ROOT/'data/visual_qa/cmrc2018_short/train.jsonl').read_text().splitlines())}
    dev={r['id']:r for r in map(json.loads,(ROOT/'data/visual_qa/cmrc2018_short/development.jsonl').read_text().splitlines())}
    states=[];references=[];original=[]
    for r in report['records']:
        p=source/'assessment'/r['file'];require(sha(p)==r['sha256'],'saved language stream changed')
        v=torch.load(p,map_location='cpu',weights_only=True);states.append(v['states'][0]);original.append(v['images'][:1]<0)
        if r['split']=='train':item=train[Path(corpus['train'][r['index']]['source']).stem]
        else:item=dev[corpus['development'][r['index']]['id']]
        references.append(sorted({chars[t['text'][0]] for t in item['references'] if t['text'] and t['text'][0] in chars}))
    output.mkdir();monitor=base.Monitor(output,meta);monitor.status('admitting')
    try:
        emitted={};records=[]
        with bridge_gpu(meta) as (device,admission),torch.no_grad():
            numerics();monitor.guard();save_json(output/'admission.json',admission)
            h=torch.stack(states).to(device)
            for version,path in [('baseline',ROOT/meta['baseline']),('robust',candidate)]:
                model=base.bridge(meta,device,path);z0=model.readout(h).float()
                for steps in [1,4]:
                    z=z0.clone()
                    for i in range(steps):
                        monitor.position=dict(model=version,projection_step=i,total_steps=steps);monitor.guard()
                        actual=model.decode_logits(z)>=0
                        if i+1<steps:
                            encoded=model.encode(1-2*actual.float())
                            # Fixed anchor retention; no convergence assumption.
                            z=.5*F.normalize(z0,dim=-1)*math.sqrt(4096)+.5*F.normalize(encoded,dim=-1)*math.sqrt(4096)
                    key=version+'_passes'+str(steps);emitted[key]=actual.cpu()
                    if key=='baseline_passes1':
                        require(all(not len(original[i]) or torch.equal(actual[i:i+1].cpu(),original[i]) for i in range(len(states))),
                                'original first image did not replay')
                    for i,r in enumerate(report['records']):
                        scores=[]
                        for label in references[i]:
                            gold=primitive['ink'][label].bool();a=actual[i].cpu();d=int(gold.sum()+a.sum())
                            scores.append(dict(form=inventory[label]['chars'],f1=2*int((a&gold).sum())/d if d else 1.,pixel_exact=torch.equal(a,gold)))
                        records.append(dict(method=key,split=r['split'],route=r['route'],index=r['index'],
                            scores=scores,best_f1=max((s['f1'] for s in scores),default=None),
                            exact_first_form=any(s['pixel_exact'] for s in scores)))
                del model
        atomic_tensor(output/'all_outputs.pt',dict(ink=emitted))
        methods=list(emitted);boards=[]
        for start in range(0,len(states),10):
            canvas=Image.new('L',(720,10*104),245);draw=ImageDraw.Draw(canvas)
            for j in range(start,min(start+10,len(states))):
                r=report['records'][j];y=(j-start)*104
                draw.text((4,y),f"{r['split']} {r['index']} {r['route']}",fill=0)
                for col,key in enumerate(methods):
                    x=col*176;draw.text((x+4,y+13),key,fill=0)
                    a=(255*(1-emitted[key][j,0].numpy().astype(np.uint8))).astype(np.uint8)
                    canvas.paste(Image.fromarray(a).resize((64,64),Image.Resampling.NEAREST),(x+4,y+27))
            name=f'first_states_{start:02}.png';canvas.save(output/name);boards.append(name)
        summary={}
        for key in methods:
            rows=[r for r in records if r['method']==key and r['split']=='development' and r['route']=='normal']
            summary[key]=dict(normal_cases=len(rows),exact_first_form=sum(r['exact_first_form'] for r in rows),
                first_form_f1_at_least095=sum(r['best_f1'] is not None and r['best_f1']>=.95 for r in rows),
                mean_first_form_f1=sum(r['best_f1'] for r in rows if r['best_f1'] is not None)/len(rows))
        save_json(output/'report.json',dict(records=records,summary=summary,boards=boards,
            source_sha256=sha(ROOT/own),decoder_checkpoint_sha256=sha(candidate),all_outputs_sha256=sha(output/'all_outputs.pt'),
            original_decoder_gate_passed=load(decoder/'qualify/report.json')['primitive_gate_passed'],
            diagnostic_only=True,full_language_assessed=False,nearest_anchor_used=False,teacher_loaded=False,
            source_body_loaded=False,optimizer_updates=0,actually_viewed=False,**monitor.receipt()))
        monitor.status('completed');print(json.dumps(summary),flush=True)
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(output/'status.json',dict(load(output/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decoder-run',type=Path,required=True);a=p.parse_args();main(a.output.resolve(),a.decoder_run.resolve())
