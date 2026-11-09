#!/usr/bin/env python3
"""Score a finished training-image milestone while the one GPU job continues.

No neural model is loaded. The terminal full audit separately re-encodes all
history images after the training process exits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(run,stage,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for this light CPU inspection')
    import torch
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    if out.exists():raise FileExistsError('completed milestone inspections are immutable')
    meta=read(run/'preparation.json');cases=read(run/'cases.json');producer=read(run/'execution'/stage/'report.json')
    if len(producer['cases'])!=32:raise ValueError('complete the32-case producer assessment first')
    for name,digest in meta['code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('active model source changed: '+name)
    if sha(run/'cases.json')!=meta['cases_sha256']:raise ValueError('prepared source cases changed')
    out.mkdir(parents=True);rows=[];feedback_count=0
    for i in range(32):
        case=cases[i];folder=run/'execution'/stage/f'{i:02}';receipt=read(folder/'receipt.json')
        identity=receipt['identity']
        if (identity['index']!=i or identity['update']!=producer['update'] or identity['route']!='normal'
                or identity['preparation_sha256']!=sha(run/'preparation.json')):
            raise ValueError('owning checkpoint or input identity differs')
        for name,digest in receipt['hashes'].items():
            if sha(folder/name)!=digest:raise ValueError('saved original output changed')
        result=torch.load(folder/'result.pt',map_location='cpu',weights_only=True)
        images=result['images'];n=len(images);probabilities=result['stop_probabilities']
        if (images.shape!=(n,1,32,64) or images.dtype!=torch.float32 or not bool(((images==-1)|(images==1)).all())
                or n>128 or len(probabilities)!=n+1 or any(not np.isfinite(p) or not 0<=p<=1 for p in probabilities)
                or any(p>=.5 for p in probabilities[:n]) or (probabilities[-1]>=.5)!=result['stopped']
                or result['stopped']==result['capped'] or (result['capped'] and n!=128)):
            raise ValueError('physical output or structural stopping differs')
        actual=np.concatenate([v[0].numpy()<0 for v in images],axis=1) if n else np.zeros((32,0),bool)
        display=(~actual).astype('uint8')*255 if n else np.full((32,64),255,'uint8')
        with Image.open(folder/'answer.png') as im:
            if not np.array_equal(np.asarray(im.convert('L')),display):raise ValueError('PNG differs from actual emitted tensor')
        exact=False;f1=0.;correct_length=False
        for ref in case['references']:
            if sha(run/ref['path'])!=ref['sha256']:raise ValueError('complete reference changed')
            with Image.open(run/ref['path']) as im:target=np.asarray(im.convert('L'))==0
            width=max(actual.shape[1],target.shape[1]);a=np.pad(actual,((0,0),(0,width-actual.shape[1])))
            b=np.pad(target,((0,0),(0,width-target.shape[1])))
            tp,fp,fn=int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum());den=2*tp+fp+fn
            f1=max(f1,2*tp/den if den else 1.)
            exact|=result['stopped'] and not result['capped'] and actual.shape==target.shape and np.array_equal(actual,target)
            correct_length|=actual.shape==target.shape
        if bool(exact)!=producer['cases'][i]['exact'] or abs(f1-producer['cases'][i]['foreground_f1'])>1e-12:
            raise ValueError('independent physical score differs from producer')
        feedback=torch.load(folder/'feedback.pt',map_location='cpu',weights_only=True)
        if len(feedback)!=n or len(result['feedback_sha256'])!=n:raise ValueError('history feedback count differs')
        for j,record in enumerate(feedback):
            if not torch.equal(record['images'],images[j:j+1]):raise ValueError('history did not consume actual emitted image')
            if hashlib.sha256(record['codes'].numpy().tobytes()).hexdigest()!=result['feedback_sha256'][j]:
                raise ValueError('recorded history feature hash differs')
            feedback_count+=1
        rows.append(dict(index=i,pair=case['pair'],exact=bool(exact),foreground_f1=f1,blocks=n,
            reference_blocks=case['blocks'],correct_length=correct_length,stopped=result['stopped'],capped=result['capped'],
            result_sha256=sha(folder/'result.pt'),output_sha256=sha(folder/'answer.png'),receipt_sha256=sha(folder/'receipt.json')))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13);plates=[]
    def parts(image):return [image.crop((x,0,min(x+768,image.width),32)) for x in range(0,image.width,768)]
    for begin in range(0,32,8):
        source=[]
        for i in range(begin,begin+8):
            with Image.open(run/'execution'/stage/f'{i:02}'/'answer.png') as im:a=parts(im.convert('RGB'))
            with Image.open(run/cases[i]['references'][0]['path']) as im:b=parts(im.convert('RGB'))
            source.append((i,a,b))
        height=35+sum(46+36*(len(a)+len(b)) for i,a,b in source)
        canvas=Image.new('RGB',(900,height),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,5),f'Update {producer["update"]} | cases {begin:02}–{begin+7:02} | original pixels, no resizing',font=font,fill='black');y=32
        for i,a,b in source:
            draw.text((12,y),f'{i:02} {cases[i]["id"]} | exact {rows[i]["exact"]} | F1 {rows[i]["foreground_f1"]:.3f}',font=font,fill='black');y+=24
            for label,images in (('Output',a),('Reference',b)):
                for j,image in enumerate(images):
                    draw.text((12,y),label if j==0 else f'+{j*12} blocks',font=font,fill='black');canvas.paste(image,(110,y));y+=36
            y+=22
        path=out/f'cases_{begin:02}_{begin+7:02}.png';canvas.save(path);plates.append(dict(path=str(path),sha256=sha(path)))
    result=dict(stage=stage,update=producer['update'],cases=rows,exact=sum(r['exact'] for r in rows),
        foreground_f1=sum(r['foreground_f1'] for r in rows)/32,
        both_exact_pairs=sum(rows[2*i]['exact'] and rows[2*i+1]['exact'] for i in range(16)),pairs=16,
        stopped=sum(r['stopped'] for r in rows),capped=sum(r['capped'] for r in rows),correct_length=sum(r['correct_length'] for r in rows),
        actual_feedback_records_hash_and_pixels_checked=feedback_count,independent_encoder_replay_performed=False,
        all32_PNG_tensor_reference_and_stop_checks_pass=True,producer_report_sha256=sha(run/'execution'/stage/'report.json'),
        preparation_sha256=sha(run/'preparation.json'),inspection_code_sha256=sha(Path(__file__)),plates=plates,
        cached_native_prompt_context=not producer['actual_PNG_prefill'],semantic_accuracy_measured=False,
        held_out_generalization_measured=False,output_pixels_repaired=False,model_loaded=False,general_ilm_goal_complete=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','plates')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=('initial_train','train_update_01000','train_update_02000','train_update_04000','train_update_08000','final_train'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();inspect(a.run.resolve(),a.stage,a.output.resolve())
