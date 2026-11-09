#!/usr/bin/env python3
"""Independently audit complete writing, stopping, image feedback and exposure."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def audit(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA during independent CPU audit')
    import torch
    from ilm.visual_lm.codec_block_writer import CodecBlockWriter,load_qualified_writing_codec
    from scripts.train_complete_pixel_language import CODEC
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1]
    meta=load(output/'preparation.json');report=load(output/'execution/report.json');status=load(output/'execution/status.json')
    if status['phase']!='completed' or status['pid'] is not None:raise ValueError('finish the actual model process first')
    for n,h in {**meta['code_sha256'],**meta['bound_sha256']}.items():
        if sha(root/n)!=h:raise ValueError('source or data changed: '+n)
    if sha(output/'data.pt')!=meta['data_sha256'] or sha(output/'cases.json')!=meta['cases_sha256']:
        raise ValueError('owning images or case identities changed')
    if sha(output/'execution/checkpoint_latest.pt')!=report['final_checkpoint_sha256']:raise ValueError('final checkpoint changed')
    data=torch.load(output/'data.pt',map_location='cpu',weights_only=True);cases=load(output/'cases.json')
    history=[json.loads(line) for line in (output/'execution/history.jsonl').read_text().splitlines()]
    if len(history)!=report['updates'] or any(row['update']!=i+1 for i,row in enumerate(history)):
        raise ValueError('actual complete learning history is missing or duplicated')
    draws=[i for row in history for i in row['indices']]
    if len(draws)!=report['retained_draws'] or len(draws)!=4*report['updates'] or set(draws)!=set(range(32)):
        raise ValueError('training exposure or source split differs')
    for i in range(0,len(draws),32):
        part=draws[i:i+32]
        if len(part)==32 and sorted(part)!=list(range(32)):raise ValueError('the training sampler skipped or repeated within an epoch')
    replay=load(output/'execution/update_replay.json')
    calls=[json.loads(line) for line in (output/'execution/optimizer_calls.jsonl').read_text().splitlines()]
    if (not replay['model_optimizer_sampler_loss_gradients_bit_exact'] or len(calls)!=report['actual_optimizer_calls']
            or any(row['call']!=i+1 for i,row in enumerate(calls))
            or (not report['explicit_resume_used'] and len(calls)!=report['updates']+1)):
        raise ValueError('actual optimizer replay accounting differs')
    codec=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device('cpu')))
    feedback_count=0;maximum_error=0.;scores=[]
    for i,case in enumerate(cases):
        for route in ('normal','blank'):
            if route not in case:continue
            if sha(output/case[route])!=case[route+'_sha256']:raise ValueError('input PNG changed')
            key='prompt' if route=='normal' else 'blank_prompt'
            with Image.open(output/case[route]) as im:
                if not np.array_equal(np.asarray(im.convert('L')),(data[i][key][0,0].numpy()>0).astype('uint8')*255):
                    raise ValueError('input PNG differs from its owning physical image')
        for j,ref in enumerate(case['references']):
            if sha(output/ref['path'])!=ref['sha256']:raise ValueError('reference PNG changed')
            with Image.open(output/ref['path']) as im:
                if not np.array_equal(np.asarray(im.convert('L')),data[i]['references'][j].numpy()):
                    raise ValueError('reference differs from complete owning target')
    folders=sorted(p for p in (output/'execution').iterdir() if p.is_dir() and (p.name in ('initial_train','final_train','development') or p.name.startswith('train_update_')))
    for parent in folders:
        requests=([(i,'normal') for i in range(32)] if parent.name!='development'
            else [(i,route) for i in range(32,56) for route in ('normal','blank')])
        for i,route in requests:
            folder=parent/(f'{i:02}_{route}' if parent.name=='development' else f'{i:02}')
            receipt=load(folder/'receipt.json')
            if receipt['identity']['index']!=i or receipt['identity']['route']!=route or receipt['identity']['preparation_sha256']!=sha(output/'preparation.json'):
                raise ValueError('output receipt source identity differs')
            for n,h in receipt['hashes'].items():
                if sha(folder/n)!=h:raise ValueError('original generated output changed')
            result=torch.load(folder/'result.pt',map_location='cpu',weights_only=True);images=result['images'];n=len(images)
            if images.shape!=(n,1,32,64) or images.dtype!=torch.float32 or not bool(((images==-1)|(images==1)).all()):
                raise ValueError('complete unmodified binary output stream required')
            probs=result['stop_probabilities'];stopped=result['stopped'];capped=result['capped']
            if (type(stopped) is not bool or type(capped) is not bool or stopped==capped or n>128 or len(probs)!=n+1
                    or not all(np.isfinite(p) and 0<=p<=1 for p in probs) or any(p>=.5 for p in probs[:n])
                    or (probs[-1]>=.5)!=stopped or (capped and n!=128)):
                raise ValueError('learned stop decisions or cap events differ from actual output')
            ink=np.concatenate([v[0].numpy()<0 for v in images],axis=1) if n else np.zeros((32,0),bool)
            display=(~ink).astype('uint8')*255 if n else np.full((32,64),255,'uint8')
            with Image.open(folder/'answer.png') as im:
                if not np.array_equal(np.asarray(im.convert('L')),display):raise ValueError('PNG differs from emitted pixels')
            exact=False;f1=0.
            for reference in data[i]['references']:
                target=reference.numpy()==0;width=max(target.shape[1],ink.shape[1])
                a=np.pad(ink,((0,0),(0,width-ink.shape[1])));b=np.pad(target,((0,0),(0,width-target.shape[1])))
                tp=int((a&b).sum());den=int(a.sum()+b.sum());f1=max(f1,2*tp/den if den else 1.)
                exact|=stopped and not capped and ink.shape==target.shape and np.array_equal(ink,target)
            records=torch.load(folder/'feedback.pt',map_location='cpu',weights_only=True)
            if len(records)!=n or len(result['feedback_sha256'])!=n:raise ValueError('actual feedback count differs')
            for j,record in enumerate(records):
                if not torch.equal(record['images'],images[j:j+1]):raise ValueError('feedback did not read the actual emitted image')
                if hashlib.sha256(record['codes'].numpy().tobytes()).hexdigest()!=result['feedback_sha256'][j]:
                    raise ValueError('actual encoded history hash differs')
                code=codec.encode_targets(record['images']);error=float((code-record['codes']).abs().max())
                if error>1e-4:raise ValueError('CPU image re-encoding exceeds fixed1e-4 tolerance')
                maximum_error=max(maximum_error,error);feedback_count+=1
            scores.append(dict(stage=parent.name,index=i,route=route,source=cases[i]['source'],pair=cases[i]['pair'],
                exact=bool(exact),foreground_f1=f1,blocks=n,stopped=stopped,capped=capped,
                path=str((folder/'answer.png').relative_to(output)),update=receipt['identity']['update']))
    def summary(rows):
        return dict(cases=len(rows),exact=sum(r['exact'] for r in rows),
            foreground_f1=float(np.mean([r['foreground_f1'] for r in rows])),stopped=sum(r['stopped'] for r in rows),capped=sum(r['capped'] for r in rows))
    groups={}
    for stage in sorted({r['stage'] for r in scores}):
        for source in sorted({r['source'] for r in scores if r['stage']==stage}):
            for route in sorted({r['route'] for r in scores if r['stage']==stage and r['source']==source}):
                groups[f'{stage}_{source}_{route}']=summary([r for r in scores if r['stage']==stage and r['source']==source and r['route']==route])
    pairs={}
    for stage in ('initial_train','final_train','development'):
        chosen=[r for r in scores if r['stage']==stage and r['source']=='cmrc' and r['route']=='normal']
        identities=sorted({r['pair'] for r in chosen})
        pairs[stage]=dict(pairs=len(identities),both_exact=sum(all(r['exact'] for r in chosen if r['pair']==p) for p in identities))
    by={(r['index'],r['route']):r for r in scores if r['stage']=='development'}
    effects={source:float(np.mean([by[i,'normal']['foreground_f1']-by[i,'blank']['foreground_f1']
        for i in range(32,56) if cases[i]['source']==source])) for source in ('cmrc','dolly')}
    result=dict(report_sha256=sha(output/'execution/report.json'),groups=groups,paired_questions=pairs,
        complete_cases=scores,retained_updates=report['updates'],actual_optimizer_calls=report['actual_optimizer_calls'],
        retained_example_draws=len(draws),complete_epochs=len(draws)//32,all_output_PNGs_match_emitted_tensors=True,
        actual_feedback_records_verified=feedback_count,feedback_CPU_GPU_max_absolute_error=maximum_error,
        question_or_prompt_erasure_F1_effect=effects,development_used_for_gradient_updates=False,development_sealed=False,
        inherited_reader_independently_replayed=False,optimizer_independently_replayed=False,
        semantic_accuracy_measured=False,general_ilm_goal_complete=False)
    if (output/'audit.json').exists():raise FileExistsError('complete audits are immutable')
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='complete_cases'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output.resolve())
