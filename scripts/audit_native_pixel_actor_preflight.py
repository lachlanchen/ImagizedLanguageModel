#!/usr/bin/env python3
"""Independently score complete native outputs and re-encode actual feedback."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for independent audit')
    import torch
    from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec,CodecBlockWriter
    from scripts.preflight_native_pixel_actor import CODEC
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    meta=read(output/'preparation.json');report=read(output/'execution/report.json')
    status=read(output/'execution/status.json')
    if status['phase']!='completed' or status['pid'] is not None:raise ValueError('finish GPU execution first')
    for name,digest in {**meta['code_sha256'],**meta['bound_sha256']}.items():
        if sha(root/name)!=digest:raise ValueError('source binding changed: '+name)
    if sha(output/'execution/final.pt')!=report['final_checkpoint_sha256']:raise ValueError('final checkpoint changed')
    if sha(output/'data.pt')!=meta['data_sha256'] or sha(output/'cases.json')!=meta['cases_sha256']:
        raise ValueError('complete physical data or case identities changed')
    data=torch.load(output/'data.pt',map_location='cpu',weights_only=True)
    codec=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device('cpu')))
    cases=read(output/'cases.json');rows=[];feedback_count=0;maximum_error=0.
    def validate_feedback(images,records,digests):
        nonlocal feedback_count,maximum_error
        if len(images)!=len(records) or len(images)!=len(digests):raise ValueError('actual feedback count differs')
        for i,record in enumerate(records):
            if not torch.equal(images[i:i+1],record['images']):raise ValueError('feedback did not consume emitted image')
            actual=record['codes']
            if hashlib.sha256(actual.numpy().tobytes()).hexdigest()!=digests[i]:raise ValueError('recorded feedback code hash differs')
            replay=codec.encode_targets(record['images'])
            error=float((replay-actual).abs().max());maximum_error=max(maximum_error,error)
            if error>1e-4:raise ValueError('CPU re-encoding exceeds registered tolerance')
            feedback_count+=1
    for i in range(5):
        case=cases[i]
        for key in ('input','reference'):
            if sha(output/case[key])!=case[key+'_sha256']:raise ValueError('owning PNG changed')
        with Image.open(output/case['input']) as im:
            if not np.array_equal(np.asarray(im.convert('L')),(data[i]['prompt'][0,0].numpy()>0).astype('uint8')*255):
                raise ValueError('actual PNG prompt differs from source pixels')
        folder=output/'execution'/f'answer_{i:02}';saved=report['outputs'][i]
        if sha(folder/'result.pt')!=saved['result_sha256'] or sha(folder/'answer.png')!=saved['png_sha256']:
            raise ValueError('original generation changed')
        result=torch.load(folder/'result.pt',map_location='cpu',weights_only=True)
        images=result['images']
        if images.ndim!=4 or images.shape[1:]!=(1,32,64) or not bool(((images==-1)|(images==1)).all()):
            raise ValueError('actual binary complete block stream required')
        if len(images)!=saved['blocks'] or len(images)>16:raise ValueError('registered output capacity differs')
        actual=np.concatenate([v[0].numpy()<0 for v in images],axis=1) if len(images) else np.zeros((32,0),bool)
        expected_png=(~actual).astype('uint8')*255 if len(images) else np.full((32,64),255,'uint8')
        with Image.open(folder/'answer.png') as im:
            if not np.array_equal(np.asarray(im.convert('L')),expected_png):raise ValueError('PNG differs from actual output')
        reference=np.concatenate([v[0].numpy()<0 for v in data[i]['images']],axis=1)
        with Image.open(output/case['reference']) as im:
            if not np.array_equal(np.asarray(im.convert('L')),(~reference).astype('uint8')*255):
                raise ValueError('full reference PNG differs from physical target')
        width=max(actual.shape[1],reference.shape[1]);a=np.pad(actual,((0,0),(0,width-actual.shape[1])))
        b=np.pad(reference,((0,0),(0,width-reference.shape[1])))
        tp,fp,fn=int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum())
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        exact=(result['stopped'] and not result['capped'] and actual.shape==reference.shape and np.array_equal(actual,reference))
        records=torch.load(folder/'feedback.pt',map_location='cpu',weights_only=True)
        validate_feedback(images,records,result['feedback_sha256'])
        rows.append(dict(index=i,id=cases[i]['id'],split=cases[i]['split'],source=cases[i]['source'],
            generated_blocks=len(images),reference_blocks=len(data[i]['images']),stopped=result['stopped'],
            capped=result['capped'],physical_exact=bool(exact),foreground_f1=f1,
            output_png=str(folder.relative_to(output)/'answer.png'),empty_output_display_only=len(images)==0))
    original=torch.load(output/'execution/answer_00/result.pt',map_location='cpu',weights_only=True)
    repeated=torch.load(output/'execution/answer_00_replay/result.pt',map_location='cpu',weights_only=True)
    if not torch.equal(original['images'],repeated['images']) or original['stop_probabilities']!=repeated['stop_probabilities']:
        raise ValueError('independent image/stop replay comparison differs')
    replay_records=torch.load(output/'execution/answer_00_replay/feedback.pt',map_location='cpu',weights_only=True)
    validate_feedback(repeated['images'],replay_records,repeated['feedback_sha256'])
    forced=torch.load(output/'execution/forced_feedback.pt',map_location='cpu',weights_only=True)
    validate_feedback(forced['images'][:1],forced['records'],[forced['first_feedback_sha256']])
    result=dict(report_sha256=sha(output/'execution/report.json'),complete_cases=rows,
        train_exact=sum(r['physical_exact'] for r in rows[:2]),development_exact=sum(r['physical_exact'] for r in rows[2:]),
        full_PNG_outputs_match_actual_tensors=True,independent_feedback_images_verified=feedback_count,
        feedback_CPU_GPU_max_absolute_error=maximum_error,feedback_tolerance=1e-4,
        first_autonomous_replay_independently_verified=True,forced_path_excluded_from_quality=True,
        inherited_reader_forward_independently_replayed=False,optimizer_updates_independently_replayed=False,
        semantic_accuracy_measured=False,language_quality_qualified=False,general_ilm_goal_complete=False)
    target=output/'audit.json'
    if target.exists():raise FileExistsError('completed audits are immutable')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output.resolve())
