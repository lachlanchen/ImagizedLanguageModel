#!/usr/bin/env python3
"""Leave whole case pairs out when learning a visual-to-text state residual."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ilm.visual_lm.visual_history_bridge import VisualHistoryBridge,unit_rms
from scripts.cache_semantic_images_v72 import ROOT,sha,atomic_json,committed_identity
from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer

SOURCES=('scripts/fit_visual_history_bridge.py','ilm/visual_lm/visual_history_bridge.py',
    'references/visual_history_bridge_protocol.md','scripts/check_native_visual_transfer_v71.py',
    'scripts/cache_semantic_images_v72.py')


def read(p):return json.loads(Path(p).read_text())


def run(source,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only learner required')
    torch.set_num_threads(4)
    if out.exists():raise FileExistsError('preserve completed or partial fit; choose a fresh output')
    verified=read(source/'verification.json');generation=read(source/'generation.json');prep=read(source/'preparation.json')
    if (verified['generation_sha256']!=sha(source/'generation.json') or verified['preparation_sha256']!=sha(source/'preparation.json')
            or len(generation['outputs'])!=len(prep['rows']) or not verified['page_pixels_reconstructed']):
        raise ValueError('complete independently verified source probe required')
    producer=committed_identity(SOURCES);out.mkdir(parents=True);(out/'folds').mkdir()
    records={r['index']:r for r in generation['outputs']};features={};texts={};groups={};metadata={r['index']:r for r in prep['rows']}
    started=time.monotonic()
    for row in prep['rows']:
        record=records[row['index']];p=source/record['file']
        if sha(p)!=record['sha256']:raise ValueError('verified native call changed')
        v=torch.load(p,map_location='cpu',weights_only=True)
        if not torch.equal(v['memory'][0,-1].float(),v['hidden_states'][0]):
            raise ValueError('first projection must use the stored prompt state')
        key=(row['group'],row['case'],row['position']);group=(row['group'],row['pair'])
        entry=dict(index=row['index'],key=key,group=group,state=v['hidden_states'][0].clone(),native_first=v['selected_ids'][0])
        if row['route']=='text':texts[key]=entry
        else:features[row['index']]=entry
        groups[group]=1
    if len(groups)!=22 or len(texts)!=172 or len(features)!=216:raise ValueError('complete paired history inventory required')
    normal=[v for v in features.values() if metadata[v['index']]['route']=='visual']
    predicted={kind:{} for kind in ('identity','mean','ridge')};folds=[]
    for number,holdout in enumerate(sorted(groups)):
        train=[v for v in normal if v['group']!=holdout];test=[v for v in features.values() if v['group']==holdout]
        x=torch.stack([v['state'] for v in train]);y=torch.stack([texts[v['key']]['state'] for v in train])
        model=VisualHistoryBridge.fit(x,y)
        p=out/'folds'/f'{number:02d}.pt';torch.save(model.state_dict(),p)
        restored=VisualHistoryBridge(**torch.load(p,map_location='cpu',weights_only=True))
        states=torch.stack([v['state'] for v in test])
        with torch.no_grad():
            factorized=restored(states);explicit=unit_rms(states)+restored.mean_delta+(unit_rms(states)-restored.mean_x)@(restored.left@restored.right)
            maximum=float((factorized-explicit).abs().max())
            if maximum>2e-5:raise ValueError('factorized/explicit affine evaluations differ')
            for kind in predicted:
                values=restored(states,kind=kind)
                for row,value in zip(test,values):
                    if row['index'] in predicted[kind]:raise ValueError('duplicate held-out prediction')
                    predicted[kind][row['index']]=value.clone()
        folds.append(dict(holdout=list(holdout),training_indices=[v['index'] for v in train],test_indices=[v['index'] for v in test],
            model=str(p.relative_to(out)),model_sha256=sha(p),numeric_parameters=sum(p.numel() for p in model.parameters()),
            explicit_affine_maximum_error=maximum))
    keys=sorted(features)
    if any(set(v)!=set(keys) for v in predicted.values()):raise ValueError('one held-out prediction per image call required')
    tensors={kind:torch.stack([v[k] for k in keys]) for kind,v in predicted.items()}
    torch.save(dict(indices=keys,predictions=tensors),out/'predictions.pt')
    fit_seconds=time.monotonic()-started
    # All fold predictions are fixed before any answer string is scored.
    archive,tokenizer,_=archive_and_tokenizer();eog=set(int(x) for x in archive.field('tokenizer.ggml.eos_token_ids'))
    weight=archive.omitted['output.weight']
    import numpy as np
    head=torch.from_numpy(np.array(archive.gguf.dequantize(weight.data,weight.tensor_type),copy=True)).to(torch.bfloat16).float()
    labels={k:texts[v['key']]['native_first'] for k,v in features.items()}
    rows=[]
    for kind,states in tensors.items():
        ids=[]
        with torch.no_grad():
            for start in range(0,len(states),16):ids.extend(F.linear(states[start:start+16],head).argmax(-1).tolist())
        for index,piece,state in zip(keys,ids,states):
            m=metadata[index];teacher=texts[features[index]['key']]
            raw=tokenizer.decode([piece],skip_special_tokens=False) if piece not in eog else ''
            visible=raw.strip();expected=m['remaining'].strip()
            teacher_state=unit_rms(teacher['state'][None])[0]
            rows.append(dict(index=index,kind=kind,group=m['group'],case=m['case'],pair=m['pair'],position=m['position'],
                characters=m['characters'],terminal=m['terminal'],middle=m['middle'],route=m['route'],
                selected_id=piece,native_first_id=features[index]['native_first'],teacher_first_id=labels[index],
                teacher_piece_agreement=piece==labels[index],eog=piece in eog,raw_piece=raw,
                expected=expected,first_piece_codepoint_correct=bool(expected and visible and '\ufffd' not in visible and visible[0]==expected[0]),
                first_piece_incomplete_unicode='\ufffd' in raw,
                normalized_state_mse=float((state-teacher_state).square().mean())))
    metrics={}
    for kind in tensors:
        metrics[kind]={}
        for group in ('copy','training'):
            part=[r for r in rows if r['kind']==kind and r['group']==group and r['route']=='visual']
            nonempty=[r for r in part if r['expected']]
            cases=sorted({r['case'] for r in part});both=0
            for case in cases:
                terminal=next(r for r in part if r['case']==case and r['terminal'])
                almost=next(r for r in part if r['case']==case and r['position']==r['characters']-1)
                both+=terminal['eog'] and not almost['eog']
            metrics[kind][group]=dict(calls=len(part),teacher_piece_agreement=sum(r['teacher_piece_agreement'] for r in part)/len(part),
                nonempty_suffixes=len(nonempty),canonical_first_piece_accuracy=sum(r['first_piece_codepoint_correct'] for r in nonempty)/len(nonempty),
                terminal_one_short_both_correct=both,stopping_pairs=len(cases),
                normalized_state_mse=sum(r['normalized_state_mse'] for r in part)/len(part))
    gates={}
    for group in ('copy','training'):
        r=metrics['ridge'][group];i=metrics['identity'][group]
        gates[group+'/teacher_agreement']=r['teacher_piece_agreement']>=.8
        gates[group+'/teacher_gain']=r['teacher_piece_agreement']-i['teacher_piece_agreement']>=.1
        gates[group+'/canonical_piece']=r['canonical_first_piece_accuracy']>=.7
        gates[group+'/stopping_pairs']=r['terminal_one_short_both_correct']/r['stopping_pairs']>=.8
    atomic_json(out/'result.json',dict(producer=producer,source_preparation_sha256=sha(source/'preparation.json'),
        source_generation_sha256=sha(source/'generation.json'),source_verification_sha256=sha(source/'verification.json'),
        folds=folds,rows=rows,metrics=metrics,gates=gates,writer_trial_admitted=all(gates.values()),
        predictions_sha256=sha(out/'predictions.pt'),analytic_fits=22,optimizer_updates=0,new_source_body_calls=0,
        features=216,fit_and_factor_checks_seconds=fit_seconds,total_seconds=time.monotonic()-started,
        all_heldout_groups_excluded=True,reference_strings_used_to_fit=False,
        source_output_projection_instrument_only=True,new_image_outputs=0,independent_ILM_qualified=False,general_ilm_goal_complete=False))
    print(json.dumps(dict(metrics=metrics,gates=gates,writer_trial_admitted=all(gates.values())),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'artifacts/visual_history_reading_v1')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/visual_history_bridge_v1');a=p.parse_args();run(a.source.resolve(),a.output.resolve())
