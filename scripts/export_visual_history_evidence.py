#!/usr/bin/env python3
"""Export complete auxiliary history readouts and independently audit ridge fits."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from scripts.cache_semantic_images_v72 import ROOT,sha,atomic_json,committed_identity


def read(p):return json.loads(Path(p).read_text())


def verify_bridge(source,bridge):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only audit required')
    torch.set_num_threads(4)
    result=read(bridge/'result.json');prep=read(source/'preparation.json');generation=read(source/'generation.json')
    if (result['source_verification_sha256']!=sha(source/'verification.json')
            or result['source_generation_sha256']!=sha(source/'generation.json')
            or result['predictions_sha256']!=sha(bridge/'predictions.pt')):
        raise ValueError('source probe or retained bridge predictions changed')
    metadata={r['index']:r for r in prep['rows']};features={};teachers={}
    for record in generation['outputs']:
        if sha(source/record['file'])!=record['sha256']:raise ValueError('original source call changed')
        m=metadata[record['index']];v=torch.load(source/record['file'],map_location='cpu',weights_only=True)
        entry=dict(state=v['hidden_states'][0].numpy().copy(),group=(m['group'],m['pair']),
            key=(m['group'],m['case'],m['position']),index=m['index'])
        if m['route']=='text':teachers[entry['key']]=entry
        else:features[m['index']]=entry
    normal=[v for v in features.values() if metadata[v['index']]['route']=='visual']
    saved=torch.load(bridge/'predictions.pt',map_location='cpu',weights_only=True)
    offsets={key:i for i,key in enumerate(saved['indices'])};counts=Counter();max_coefficient_error=0.;max_forward_error=0.
    max_coefficient_relative_error=0.;max_forward_relative_error=0.

    def rms(x):return x/np.maximum((x*x).mean(-1,keepdims=True),1e-12)**.5

    for fold in result['folds']:
        holdout=tuple(fold['holdout']);train=[v for v in normal if v['group']!=holdout]
        test=[v for v in features.values() if v['group']==holdout]
        if fold['training_indices']!=[v['index'] for v in train] or fold['test_indices']!=[v['index'] for v in test]:
            raise ValueError('fold exclusion or complete test membership differs')
        p=bridge/fold['model']
        if sha(p)!=fold['model_sha256']:raise ValueError('analytic fold parameters changed')
        model={k:v.numpy() for k,v in torch.load(p,map_location='cpu',weights_only=True).items()}
        if not all(np.isfinite(v).all() for v in model.values()):raise ValueError('nonfinite learned factor')
        x=rms(np.stack([v['state'] for v in train]).astype(np.float64));y=rms(np.stack([teachers[v['key']]['state'] for v in train]).astype(np.float64))
        mx=x.mean(0);delta=(y-x).mean(0);xc=x-mx
        right=np.linalg.solve(xc@xc.T/4096+.1*np.eye(len(x)),(y-x)-delta)
        expected={'mean_x':mx,'mean_delta':delta,'left':xc.T/4096,'right':right}
        error=max(float(np.max(np.abs(model[k]-v))) for k,v in expected.items())
        relative=max(float(np.max(np.abs(model[k]-v)/(1+np.abs(v)))) for k,v in expected.items())
        if relative>2e-6:raise ValueError('stored coefficients differ from independent NumPy fit')
        max_coefficient_error=max(max_coefficient_error,error)
        max_coefficient_relative_error=max(max_coefficient_relative_error,relative)
        z=rms(np.stack([v['state'] for v in test]).astype(np.float32))
        values={'identity':z,'mean':z+model['mean_delta'],
            'ridge':z+model['mean_delta']+(z-model['mean_x'])@model['left']@model['right']}
        for kind,value in values.items():
            actual=saved['predictions'][kind][[offsets[v['index']] for v in test]].numpy()
            error=float(np.max(np.abs(value-actual)));max_forward_error=max(max_forward_error,error)
            relative=error/max(1.,float(np.max(np.abs(value))))
            max_forward_relative_error=max(max_forward_relative_error,relative)
            if relative>2e-4:raise ValueError('held-out saved vectors differ from independent affine forward')
        for v in test:counts[v['index']]+=1
    if set(counts)!=set(features) or set(counts.values())!={1}:raise ValueError('each visual/erased call must be withheld exactly once')
    # Independent scalar reconstruction; no answer string is used in fitting.
    metrics={}
    for kind in ('identity','mean','ridge'):
        metrics[kind]={}
        for group in ('copy','training'):
            rows=[r for r in result['rows'] if r['kind']==kind and r['group']==group and r['route']=='visual']
            nonempty=[r for r in rows if r['expected']]
            for r in rows:
                piece=r['raw_piece'].strip()
                correct=bool(r['expected'] and piece and '\ufffd' not in piece and piece[0]==r['expected'][0])
                if correct!=r['first_piece_codepoint_correct'] or (r['selected_id']==r['teacher_first_id'])!=r['teacher_piece_agreement']:
                    raise ValueError('first-piece scalar report changed')
            cases=sorted({r['case'] for r in rows});pairs=0
            for case in cases:
                terminal=next(r for r in rows if r['case']==case and r['terminal'])
                almost=next(r for r in rows if r['case']==case and r['position']==r['characters']-1)
                pairs+=bool(terminal['eog'] and not almost['eog'])
            metrics[kind][group]=dict(calls=len(rows),teacher_piece_agreement=sum(r['teacher_piece_agreement'] for r in rows)/len(rows),
                nonempty_suffixes=len(nonempty),canonical_first_piece_accuracy=sum(r['first_piece_codepoint_correct'] for r in nonempty)/len(nonempty),
                terminal_one_short_both_correct=pairs,stopping_pairs=len(cases),normalized_state_mse=sum(r['normalized_state_mse'] for r in rows)/len(rows))
    if metrics!=result['metrics']:raise ValueError('independent group aggregates differ')
    receipt=dict(result_sha256=sha(bridge/'result.json'),source_verification_sha256=sha(source/'verification.json'),
        predictions_sha256=sha(bridge/'predictions.pt'),analytic_fits_reconstructed=22,heldout_calls_verified=len(counts),
        maximum_coefficient_error=max_coefficient_error,maximum_forward_error=max_forward_error,
        maximum_coefficient_relative_error=max_coefficient_relative_error,maximum_forward_relative_error=max_forward_relative_error,
        all_fold_exclusions_verified=True,metrics_reconstructed=True,source_body_calls=0,
        native_projection_replayed=False,new_image_outputs=0,general_ilm_goal_complete=False)
    atomic_json(bridge/'verification.json',receipt)
    print(json.dumps(receipt,indent=2))


def export(source,bridge,out):
    producer=committed_identity(('scripts/export_visual_history_evidence.py',))
    probe=read(source/'verification.json');generation=read(source/'generation.json');prep=read(source/'preparation.json')
    aligned=read(bridge/'result.json');checked=read(bridge/'verification.json')
    if checked['result_sha256']!=sha(bridge/'result.json') or aligned['source_verification_sha256']!=sha(source/'verification.json'):
        raise ValueError('current independent bridge verification required')
    if probe['generation_sha256']!=sha(source/'generation.json'):raise ValueError('verified source readouts changed')
    if out.exists():raise FileExistsError('preserve previously exported evidence')
    out.mkdir(parents=True);files={};rows=[];metadata={r['index']:r for r in prep['rows']}
    for row in probe['rows']:
        m=metadata[row['index']];name=None
        if row['route']!='text':
            p=source/row['input']
            if sha(p)!=row['input_sha256']:raise ValueError('physical input changed')
            name='inputs/'+row['input_sha256']+'.png';target=out/name
            if name not in files:
                target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
                if sha(target)!=sha(p):raise ValueError('input PNG bytes changed')
                files[name]=sha(target)
        p=source/row['raw_file']
        if sha(p)!=row['raw_sha256']:raise ValueError('raw native call changed')
        v=torch.load(p,map_location='cpu',weights_only=True)
        rows.append({**row,'input_image':name,'selected_ids':v['selected_ids'],'selected_logits':v['selected_logits'],
            'metadata':v['metadata'],'elapsed_seconds':v['elapsed_seconds'],'raw_response':v['text']})
    source_cases=read(ROOT/'artifacts/reader_content_v77_20261106/preparation.json')['cases']
    cases=[{k:r[k] for k in ('group','position','pair','script','source_prompt','reference')} for r in source_cases]
    public_probe={k:v for k,v in probe.items() if k!='rows'}
    public_bridge={k:v for k,v in aligned.items() if k!='producer'}
    result=dict(producer=producer,probe=public_probe,probe_rows=rows,cases=cases,bridge=public_bridge,
        bridge_verification=checked,source_generation_sha256=sha(source/'generation.json'),source_verification_sha256=sha(source/'verification.json'),
        source_execution={k:generation[k] for k in ('setup_seconds','current_inference_seconds','current_total_seconds','peak_allocated_gib')},
        files_sha256=files,all216_visual_inputs_byte_unchanged=True,all388_raw_responses_retained=True,
        model_weights_published=False,source_text_projection_instrument_only=True,source_pretraining_retained=True,
        image_answer_generated=False,general_ilm_goal_complete=False,
        attribution='CMRC2018 Cui et al. (2019), CC BY-SA 4.0; Qwen3-VL Alibaba, Apache-2.0; prepared copy requests.')
    if len(rows)!=388 or sum(r['input_image'] is not None for r in rows)!=216:raise ValueError('complete probe inventory required')
    atomic_json(out/'result.json',result)
    lines=['# Native visual history: complete auxiliary readouts','',
        'All 388 source responses below are an offline native text readout, not generated ILM answer images.',
        'Question and reference-prefix pixels are inputs. All 216 visual/erased input calls are retained unchanged; duplicate PNGs share a hash path.',
        'Raw strings, native decisions, source identities and all 648 cross-fitted first-piece readouts are in result.json.',
        'This test supplies correct reference history. No autonomous visual-language model is qualified.','',
        'CMRC2018: Cui et al. (2019), CC BY-SA 4.0. Qwen3-VL: Alibaba, Apache-2.0.','',
        '| Group | Case | Prefix position | Route | Expected missing suffix | Raw source response | Exact after outer whitespace trim | Input |',
        '|---|---:|---:|---|---|---|---|---|']
    def safe(s):return json.dumps(s,ensure_ascii=False).replace('|','&#124;').replace('`','&#96;')
    for r in rows:
        inp=f"[PNG]({r['input_image']})" if r['input_image'] else 'native text'
        lines.append(f"| {r['group']} | {r['case']} | {r['position']} | {r['route']} | {safe(r['expected'])} | {safe(r['raw_response'])} | {r['exact']} | {inp} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(probe=probe['groups'],probe_gate=probe['writer_trial_admitted'],bridge=aligned['metrics'],
        bridge_gate=aligned['writer_trial_admitted'],unique_input_pngs=len(files)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('verify-bridge','export'),required=True)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/visual_history_reading_v1')
    p.add_argument('--bridge',type=Path,default=ROOT/'artifacts/visual_history_bridge_v1')
    p.add_argument('--output',type=Path,default=ROOT/'references/evidence/visual_history_reading_result');a=p.parse_args()
    if a.stage=='verify-bridge':verify_bridge(a.source.resolve(),a.bridge.resolve())
    else:export(a.source.resolve(),a.bridge.resolve(),a.output.resolve())
