#!/usr/bin/env python3
"""Evaluator-only geometry check; no nearest-anchor repair enters an answer."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
from scripts.distill_semantic_glyph_readout import Monitor
from scripts.train_semantic_glyph_bridge import bridge_gpu
from scripts.train_semantic_glyph_language import load,numerics
from scripts.score_pixel_language_first_blocks import require

RUN=ROOT/'artifacts/semantic_glyph_language_v1'
PRIMITIVE=ROOT/'artifacts/semantic_glyph_bridge_v1'


def main(output):
    own='scripts/diagnose_semantic_glyph_geometry.py'
    require(subprocess.check_output(['git','show','HEAD:'+own],cwd=ROOT)==(ROOT/own).read_bytes(),'commit geometry diagnostic first')
    require(not output.exists(),'preserve earlier geometry observations')
    checkpoint=RUN/'fit/checkpoint_latest.pt'
    require(sha(checkpoint)==load(RUN/'fit/report.json')['checkpoint_sha256'],'preserved baseline changed')
    data=torch.load(PRIMITIVE/'training.pt',weights_only=True,map_location='cpu')
    inventory=load(PRIMITIVE/'inventory.json')['records'];charmap={c:i for i,r in enumerate(inventory) for c in r['chars']}
    corpus=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    train={r['id']:r for r in map(json.loads,(ROOT/'data/visual_qa/cmrc2018_short/train.jsonl').read_text().splitlines())}
    dev={r['id']:r for r in map(json.loads,(ROOT/'data/visual_qa/cmrc2018_short/development.jsonl').read_text().splitlines())}
    output.mkdir();meta=dict(minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,maximum_vram_gib=5.,
        maximum_stage_seconds=600.,maximum_pressure_wait_seconds=300.)
    monitor=Monitor(output,meta);monitor.status('admitting')
    try:
        records=[]
        with bridge_gpu(meta) as (device,admission),torch.no_grad():
            numerics();monitor.guard();save_json(output/'admission.json',admission)
            model=SemanticGlyphBridge().to(device)
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['model']);model.eval()
            anchors=((data['semantic']-data['mean'])/data['scale']).to(device);unit=F.normalize(anchors,dim=-1)
            ink=data['ink'].to(device).bool()
            assessment=load(RUN/'assessment/report.json')
            for row in assessment['records']:
                monitor.position=dict(split=row['split'],route=row['route'],index=row['index']);monitor.guard()
                p=RUN/'assessment'/row['file'];require(sha(p)==row['sha256'],'assessed state changed')
                value=torch.load(p,map_location='cpu',weights_only=True)
                h=value['states'][0:1].to(device);logits,z,_=model.predict(h)
                actual=logits>=0
                if len(value['images']):require(torch.equal(actual.cpu(),value['images'][:1]<0),'fixed first output changed')
                similarity=(F.normalize(z,dim=-1)@unit.T)[0];ranking=similarity.argsort(descending=True)
                if row['split']=='train':item=train[Path(corpus['train'][row['index']]['source']).stem]
                else:item=dev[corpus['development'][row['index']]['id']]
                refs=[v['text'] for v in item['references']]
                ids=sorted({charmap[t[0]] for t in refs if t and t[0] in charmap})
                scores=[]
                for i in ids:
                    gold=ink[i:i+1];denom=int(actual.sum()+gold.sum())
                    scores.append(dict(form=inventory[i]['chars'],rank=int((ranking==i).nonzero()[0])+1,
                        cosine=float(similarity[i]),standardized_rms_error=float((z[0]-anchors[i]).square().mean().sqrt()),
                        pixel_f1=2*int((actual&gold).sum())/denom if denom else 1.,pixel_exact=bool(torch.equal(actual,gold))))
                records.append(dict(split=row['split'],route=row['route'],index=row['index'],reference_first_forms=scores,
                    top_forms=[inventory[int(i)]['chars'] for i in ranking[:3]],
                    best_reference_rank=min((r['rank'] for r in scores),default=None),
                    saved_first_pixels_reproduced=bool(len(value['images'])),
                    source_state_sha256=row['sha256']))
            stress=[];torch.manual_seed(202611104);torch.cuda.manual_seed_all(202611104)
            for route in ['gaussian_0.03','gaussian_0.3','gaussian_0.6','mixture_0.3','mixture_0.45']:
                f1s=[];exact=0
                for start in range(0,len(anchors),64):
                    monitor.position=dict(stress=route,start=start,total=len(anchors));monitor.guard()
                    z=anchors[start:start+64].clone()
                    strength=float(route.rsplit('_',1)[1])
                    if route.startswith('gaussian'):z+=strength*torch.randn_like(z)
                    else:
                        # A deterministic different form; no selection by results.
                        other=(torch.arange(start,start+len(z),device=device)+1049)%len(anchors)
                        z=(1-strength)*z+strength*anchors[other]
                    actual=model.decode_logits(z)>=0;gold=ink[start:start+len(z)]
                    denom=actual.flatten(1).sum(-1)+gold.flatten(1).sum(-1)
                    f1=torch.where(denom==0,torch.ones_like(denom).float(),2*(actual&gold).flatten(1).sum(-1)/denom)
                    f1s.extend(f1.cpu().tolist());exact+=int((actual==gold).flatten(1).all(-1).sum())
                stress.append(dict(route=route,forms=len(anchors),mean_f1=sum(f1s)/len(f1s),pixel_exact=exact,
                    fraction_f1_at_least_095=sum(f>=.95 for f in f1s)/len(f1s)))
        normal=[r for r in records if r['split']=='development' and r['route']=='normal']
        summary=dict(normal_cases=len(normal),normal_top_reference_form=sum(r['best_reference_rank']==1 for r in normal),
            normal_reference_in_top3=sum(r['best_reference_rank'] is not None and r['best_reference_rank']<=3 for r in normal),
            normal_first_image_exact=sum(any(s['pixel_exact'] for s in r['reference_first_forms']) for r in normal))
        save_json(output/'report.json',dict(records=records,summary=summary,decoder_stress=stress,
            source_sha256=sha(ROOT/own),checkpoint_sha256=sha(checkpoint),
            evaluator_only_anchor_ranking=True,runtime_answers_replaced=False,training_updates=0,
            teacher_head_loaded=False,body_loaded=False,semantic_accuracy_measured=False,**monitor.receipt()))
        monitor.status('completed');print(json.dumps(dict(summary=summary,decoder_stress=stress)),flush=True)
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(output/'status.json',dict(load(output/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
