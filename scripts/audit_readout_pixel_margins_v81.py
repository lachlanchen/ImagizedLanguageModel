#!/usr/bin/env python3
"""Inspect retained image-code predictions with the fixed decoder on CPU."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, CODEC, sha, committed_identity
from scripts.probe_native_corpus_fit_v80 import cases_and_data, targets

PLAN = "references/readout_pixel_margin_v81_protocol.md"
SOURCES = ("scripts/audit_readout_pixel_margins_v81.py", PLAN,
    "ilm/visual_lm/codec_block_writer.py", "ilm/visual_lm/continuous_glyph_codec.py",
    "scripts/probe_native_corpus_fit_v80.py", "scripts/train_native_chinese_v80.py")
EDGES = [-float('inf'),-8,-4,-2,-1,-.25,0,.25,1,2,4,8,float('inf')]


def load(path):
    return json.loads(Path(path).read_text())


def histogram(values):
    return np.histogram(values,bins=EDGES)[0].astype(int).tolist()


def logits_for(writer, coordinates):
    values = writer.codec.decode(coordinates.reshape(-1,768)).reshape(-1,2,1,32,32)
    return torch.cat((values[:,0],values[:,1]),dim=-1)[:,0]


def pixels(images):
    if images.ndim != 4 or images.shape[1:] != (1,32,64) or not bool(((images==-1)|(images==1)).all()):
        raise ValueError("original signed binary writing blocks required")
    return ((images[:,0]+1)*127.5).to(torch.uint8).numpy()


def original_reference(data,case):
    path = TRAIN/case['source']
    if sha(TRAIN/'report.json') != TRAIN_SHA or sha(path) != data.images.report['files_sha256'][case['source']]:
        raise ValueError("bit-packed owning training target changed")
    value = torch.load(path,map_location='cpu',weights_only=True)
    raw = np.unpackbits(value['bits'].numpy(),count=32*value['width']).reshape(32,value['width'])*255
    if not np.all(raw[:,-128:]==255):raise ValueError("structural termination pixels changed")
    return raw[:,value['prompt_blocks']*64:-128]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/readout_pixel_margins_v81_20261106')
    args = parser.parse_args();out = args.out.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('explicit CPU-only diagnostic required')
    if out.exists():raise FileExistsError('preserve completed or partial margin audit')
    torch.set_num_threads(2)
    source = ROOT/'artifacts/native_chinese_v80_20261106/corpus_fit_v80r2'
    control = ROOT/'artifacts/adaptive_readout_v81r2_20261106/control'
    reports = {'native':load(source/'report.json'),'control':load(control/'report.json')}
    if (load(source/'verification.json')['report_sha256'] != sha(source/'report.json')
            or reports['control']['updates'] != 2000 or reports['control']['draws'] != 8000):
        raise ValueError('completed original native diagnosis and frozen-control arm required')
    cases = load(source/'cases.json');_,data,indices = cases_and_data()
    identity = dict(**committed_identity(SOURCES),codec_sha256=sha(CODEC),
        cases_sha256=sha(source/'cases.json'),source_report_sha256={
            'native':sha(source/'report.json'),'control':sha(control/'report.json')},
        edges=[str(v) if not np.isfinite(v) else v for v in EDGES],frozen_training_arm=True,actual_model_updates=0)
    out.mkdir();write(out/'identity.json',identity)
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device('cpu')))
    rows=[];retained={}
    for arm,root in [('native',source),('control',control/'after')]:
        for i,case in enumerate(cases):
            folder = root/'predictions'/f'{i:02}'
            header = load(folder/'header.json')
            if header['source'] != case['source']:raise ValueError('source question changed')
            for name,digest in header['files_sha256'].items():
                if sha(folder/name) != digest:raise ValueError('retained prediction changed')
            _,codes,reference = targets(data,indices[case['source']],case)
            raw = original_reference(data,case)
            if not np.array_equal(raw,reference.numpy()):raise ValueError('independent target reconstruction differs')
            target = np.stack([raw[:,j:j+64] for j in range(0,raw.shape[1],64)])
            image = pixels(torch.load(folder/'assisted.pt',map_location='cpu',weights_only=True))
            if not np.array_equal(np.concatenate(list(image),axis=1),np.asarray(Image.open(folder/'assisted.png'))):
                raise ValueError('saved correct-history PNG differs from emitted pixels')
            trace = torch.load(folder/'assisted_trace.pt',map_location='cpu',weights_only=True)
            predicted = logits_for(writer,trace['coordinates'])
            supplied = logits_for(writer,codes)
            if not torch.isfinite(predicted).all() or predicted.shape != target.shape:
                raise ValueError('complete finite predicted logits required')
            retained[f'{arm}_{i:02}'] = dict(predicted_logits=predicted,supplied_logits=supplied,
                coordinates=trace['coordinates'],target_codes=codes,
                source_assisted_trace_sha256=sha(folder/'assisted_trace.pt'))
            cpu = predicted.sigmoid().mul(2).sub(1).numpy() >= 0
            white = target == 255
            signed = torch.from_numpy(white.astype(np.float32)).mul(2).sub(1)*predicted
            supplied_signed = torch.from_numpy(white.astype(np.float32)).mul(2).sub(1)*supplied
            losses = F.binary_cross_entropy_with_logits(predicted,torch.from_numpy(white.astype(np.float32)),reduction='none').numpy()
            mse = (trace['coordinates']-codes).square().flatten(1).mean(1)
            for j in range(len(codes)):
                actual = image[j]==255;truth = white[j];errors = actual != truth
                tp = int((~actual&~truth).sum());fp = int((~actual&truth).sum());fn = int((actual&~truth).sum())
                class_losses = [float(losses[j][mask].mean()) for mask in (~truth,truth) if mask.any()]
                disagreement = cpu[j] != actual
                rows.append(dict(arm=arm,case=i,source=case['source'],block=j,first=j==0,
                    pixel_errors=int(errors.sum()),exact=not bool(errors.any()),
                    foreground_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                    ink_errors=fn,paper_errors=fp,empty_ink_target=not bool((~truth).any()),
                    coordinate_mse=float(mse[j]),class_balanced_bce=float(np.mean(class_losses)),
                    wrong_saved_pixel_margin_histogram=histogram(signed[j].numpy()[errors]),
                    supplied_margin_histogram=histogram(supplied_signed[j].numpy()),
                    cpu_saved_gpu_bit_disagreements=int(disagreement.sum()),
                    max_abs_logit_at_cpu_disagreement=float(predicted[j].abs().numpy()[disagreement].max()) if disagreement.any() else 0.))
        write(out/'status.json',dict(stage='cpu_margin_audit',completed_arm=arm,blocks=len(rows)))
    torch.save(retained,out/'decoded_logits.pt')
    summaries={}
    for arm in ('native','control'):
        summaries[arm]={}
        for name,first in [('first',True),('later',False)]:
            values=[r for r in rows if r['arm']==arm and r['first']==first]
            summaries[arm][name]=dict(blocks=len(values),exact=sum(r['exact'] for r in values),
                pixel_errors=sum(r['pixel_errors'] for r in values),
                foreground_f1=float(np.mean([r['foreground_f1'] for r in values])),
                mean_coordinate_mse=float(np.mean([r['coordinate_mse'] for r in values])),
                mean_class_balanced_bce=float(np.mean([r['class_balanced_bce'] for r in values])),
                wrong_saved_pixel_margin_histogram=np.sum([r['wrong_saved_pixel_margin_histogram'] for r in values],axis=0).astype(int).tolist(),
                cpu_saved_gpu_bit_disagreements=sum(r['cpu_saved_gpu_bit_disagreements'] for r in values))
    write(out/'report.json',dict(identity=identity,rows=rows,summaries=summaries,
        logits_sha256=sha(out/'decoded_logits.pt'),cuda_initialized=False,model_or_optimizer_updated=False,
        auxiliary_decoder_measurement=True,whole_context_or_sampler_replayed=False,
        adapted_arm_not_evaluated=True,language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
