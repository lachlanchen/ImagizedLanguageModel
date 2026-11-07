#!/usr/bin/env python3
"""Export all verified source-canvas outputs without changing their pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json, committed_identity
from scripts.compare_source_canvas import prepared, checked_image


def export(run,output):
    producer=committed_identity(('scripts/export_source_canvas.py',))
    identity,prep=prepared(run)
    verified=json.loads((run/'verification.json').read_text())
    if verified['identity']!=identity:raise ValueError('verification belongs to a different run')
    if output.exists():raise ValueError('preserve earlier exported evidence')
    output.mkdir(parents=True);files={};references={};rows=[]
    index={r['id']:r for r in prep['assessment']+prep['training']}

    def copy(source,name):
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if sha(source)!=sha(target):raise ValueError('raw PNG changed on export')
        files[name]=sha(target)
        return name

    for arm in ('summary','source'):
        for r in verified['arms'][arm]['rows']:
            if sha(run/r['output'])!=r['output_sha256'] or sha(run/r['coordinates'])!=r['coordinates_sha256']:
                raise ValueError('verified output changed')
            names=[]
            for target in index[r['id']]['targets']:
                p=ROOT/target['file'];checked_image(p,target['sha256'])
                name='references/'+target['sha256']+'.png'
                if name not in files:copy(p,name)
                names.append(name)
            p=copy(run/r['output'],'raw/'+r['output'])
            rows.append(dict(**r,arm=arm,output=p,reference_images=names))
    if Counter(r['arm'] for r in rows)!=Counter(summary=256,source=256):
        raise ValueError('retain all 512 raw canvases')
    # Fixed inventory order; all output width remains on each review sheet.
    overviews=[]
    for arm in ('summary','source'):
        for task in ('qa','copy'):
            chosen=[r for r in rows if r['arm']==arm and r['cohort']=='development'
                and r['condition']=='normal' and r['task']==task][:8]
            canvas=Image.new('RGB',(2100,112*len(chosen)),'#e9edf0');draw=ImageDraw.Draw(canvas)
            for i,row in enumerate(chosen):
                y=i*112
                draw.text((5,y+3),f"{arm}  {row['id']}  F1 {row['foreground_f1']:.3f}  upper: prediction / lower: reference",fill='black')
                canvas.paste(Image.open(output/row['output']).convert('RGB'),(5,y+22))
                canvas.paste(Image.open(output/row['reference_images'][0]).convert('RGB'),(5,y+60))
            name=f'{arm}_{task}_first08.png';canvas.save(output/name);files[name]=sha(output/name);overviews.append(name)
    payload=dict(producer=producer,training_identity=identity,
        verification_sha256=sha(run/'verification.json'),arms=verified['arms'],
        gates=verified['gates'],screen_pass=verified['screen_pass'],
        supplied_code_exact=verified['supplied_code_exact'],supplied_code_f1=verified['supplied_code_f1'],
        rows=rows,files_sha256=files,overviews=overviews,all512_raw_PNGs_unchanged=True,
        cached_feature_diagnostic=True,native_image_path_qualified=False,
        external_pretraining_retained=True,additional_preflight_optimizer_invocations=10,
        initial_preparation_index_errors=2,initial_preparation_optimizer_updates=0,
        model_weights_published=False,general_ilm_goal_complete=False,
        attribution='CMRC2018 Cui et al. (2019), CC BY-SA 4.0; Qwen3-VL Alibaba, Apache-2.0; prepared copy requests.')
    # Retain the complete verified records; local coordinate tensors stay private.
    atomic_json(output/'result.json',payload)
    lines=['# Source features versus summary-only: all raw writing outputs','',
        'Each arm has 256 complete generated canvases: 224 development and the first 32 training cases.',
        'All 512 raw PNGs are unchanged. The fixed canvas is 32 by 2048 pixels; no ink is cropped or cleaned.',
        'This is a cached-feature comparison. A clean native image-input path is not qualified.',
        'CMRC2018: Cui et al. (2019), CC BY-SA 4.0. Qwen3-VL: Alibaba, Apache-2.0.','',
        '| Arm | Cohort | Task | Condition | ID | Exact | F1 | References | Output |',
        '|---|---|---|---|---|---|---:|---|---|']
    for r in rows:
        refs=' '.join(f'[PNG {i+1}]({name})' for i,name in enumerate(r['reference_images']))
        lines.append(f"| {r['arm']} | {r['cohort']} | {r['task']} | {r['condition']} | {r['id']} | {r['exact']} | {r['foreground_f1']:.6f} | {refs} | [PNG]({r['output']}) |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(arms={k:v['groups'] for k,v in verified['arms'].items()},
        gates=verified['gates'],screen_pass=verified['screen_pass'],raw_canvases=512,png_files=len(files)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/source_canvas_v1')
    p.add_argument('--output',type=Path,default=ROOT/'references/evidence/source_canvas_result')
    a=p.parse_args();export(a.run.resolve(),a.output.resolve())
