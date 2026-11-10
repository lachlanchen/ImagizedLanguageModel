#!/usr/bin/env python3
"""Cold replay all held-out readout qualifications and retain every glyph image."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.distill_semantic_glyph_readout import check,bridge,load,Monitor
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.train_semantic_glyph_language import numerics
from scripts.score_pixel_language_first_blocks import require


def run(output):
    import torch
    prepared=load(output/'preparation.json')
    if prepared.get('variant')=='contrastive_loss_only':
        from scripts.train_contrastive_glyph_readout import check as check_variant
        meta=check_variant(output)
    else:meta=check(output)
    own='scripts/replay_semantic_readout_qualification.py'
    require(subprocess.check_output(['git','show','HEAD:'+own],cwd=ROOT)==(ROOT/own).read_bytes(),'commit replay source first')
    q=output/'qualify';require(load(q/'status.json')['phase']=='completed','complete original qualification first')
    report=load(q/'report.json');require(sha(q/'records.json')==report['records_sha256'],'qualification records changed')
    baseline=ROOT/meta['baseline'];candidate=output/'fit/checkpoint_latest.pt'
    require(sha(candidate)==load(output/'fit/report.json')['checkpoint_sha256'],'candidate changed')
    raw=load(q/'records.json');folder=output/'visual_replay';folder.mkdir(exist_ok=False)
    monitor=Monitor(folder,meta);monitor.status('admitting')
    try:
        torch.set_num_threads(4)
        data=torch.load(ROOT/meta['primitive']/'training.pt',map_location='cpu',weights_only=True)
        synth=torch.load(output/'calibrate/calibration.pt',weights_only=True,map_location='cpu')
        genuine=torch.load(output/'genuine.pt',weights_only=True,map_location='cpu')
        labels=torch.load(output/'calibrate/genuine_labels.pt',weights_only=True,map_location='cpu')['labels']
        a=synth['split']==1;b=(genuine['split']==1)&(labels>=0)
        pools={'synthetic':(synth['states'][a],synth['labels'][a]),'genuine':(genuine['states'][b],labels[b])}
        emissions={};files={};checked=0
        with bridge_gpu(meta) as (device,admission),torch.no_grad():
            numerics();monitor.guard();save_json(folder/'admission.json',admission)
            for version,path in [('baseline',baseline),('distilled',candidate)]:
                model=bridge(meta,device,path)
                for route,(states,targets) in pools.items():
                    actual=[];expected=[r for r in raw if r['model']==version and r['route']==route]
                    for start in range(0,len(states),64):
                        monitor.position=dict(model=version,route=route,start=start,total=len(states));monitor.guard()
                        logits,_,_=model.predict(states[start:start+64].to(device))
                        bits=logits>=0;gold=data['ink'][targets[start:start+64]].to(device).bool()
                        intersection=(bits&gold).flatten(1).sum(-1);denominator=bits.flatten(1).sum(-1)+gold.flatten(1).sum(-1)
                        f1=torch.where(denominator==0,torch.ones_like(denominator).float(),2*intersection/denominator)
                        exact=(bits==gold).flatten(1).all(-1);pixels=bits.cpu()
                        for i in range(len(pixels)):
                            old=expected[start+i]
                            require(old['index']==start+i and old['label']==int(targets[start+i]) and
                                    old['pixel_exact']==bool(exact[i]) and old['f1']==float(f1[i]),
                                    'cold replay differs from original qualification')
                            checked+=1
                        actual.append(pixels)
                    emissions[version+'/'+route]=torch.cat(actual)
                    name=version+'_'+route+'.pt'
                    atomic_tensor(folder/name,dict(ink=emissions[version+'/'+route],labels=targets,
                        checkpoint_sha256=sha(path),qualification_records_sha256=sha(q/'records.json')))
                    files[name]=sha(folder/name)
                del model
        # Every output is retained above. Display 64 evenly spaced triplets,
        # fixed by pool order, without selecting successful examples.
        boards=[]
        for route,(_,labels) in pools.items():
            count=min(64,len(labels));indices=np.linspace(0,len(labels)-1,count,dtype=int).tolist()
            canvas=Image.new('L',(8*112,8*49),245);draw=ImageDraw.Draw(canvas)
            for i,index in enumerate(indices):
                x=(i%8)*112;y=(i//8)*49
                draw.text((x+2,y),str(index),fill=0)
                rows=[data['ink'][labels[index]],emissions['baseline/'+route][index],emissions['distilled/'+route][index]]
                for j,value in enumerate(rows):
                    image=Image.fromarray((255*(1-value[0].numpy().astype(np.uint8))).astype(np.uint8))
                    canvas.paste(image,(x+j*34+2,y+13))
            name=route+'_gold_baseline_distilled.png'
            canvas.resize((canvas.width*2,canvas.height*2),Image.Resampling.NEAREST).save(folder/name)
            boards.append(dict(file=name,sha256=sha(folder/name),indices=indices,columns=['gold','baseline','distilled']))
        save_json(folder/'report.json',dict(source_sha256=sha(ROOT/own),all_outputs_retained=True,
            replayed_measurements=checked,all_pixel_metrics_bit_exact=True,files_sha256=files,boards=boards,
            qualification_report_sha256=sha(q/'report.json'),teacher_head_loaded=False,body_loaded=False,
            semantic_accuracy_measured=False,actually_viewed=False,**monitor.receipt()))
        monitor.status('completed')
        print(json.dumps(dict(replayed_measurements=checked,all_pixel_metrics_bit_exact=True,boards=boards)),flush=True)
    except BaseException as error:
        monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
