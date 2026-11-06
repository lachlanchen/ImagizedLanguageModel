#!/usr/bin/env python3
"""Independently recompute retained margin measurements with NumPy."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, sha, committed_identity

SOURCES = ('scripts/verify_readout_pixel_margins_v81.py',)
BINS = [-np.inf,-8,-4,-2,-1,-.25,0,.25,1,2,4,8,np.inf]


def read(path):return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/readout_pixel_margins_v81_20261106')
    args = parser.parse_args();run = args.run.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('explicit CPU-only verification required')
    torch.set_num_threads(2)
    if (run/'verification.json').exists():raise FileExistsError('preserve completed verification')
    report = read(run/'report.json');identity = report['identity']
    for name,digest in identity['sources'].items():
        if sha(ROOT/name) != digest:raise ValueError('producing source changed')
    if sha(run/'decoded_logits.pt') != report['logits_sha256'] or sha(TRAIN/'report.json') != TRAIN_SHA:
        raise ValueError('retained logits or training image report changed')
    original = ROOT/'artifacts/native_chinese_v80_20261106/corpus_fit_v80r2'
    control = ROOT/'artifacts/adaptive_readout_v81r2_20261106/control'
    cases = read(original/'cases.json');training = read(TRAIN/'report.json')
    if sha(original/'cases.json') != identity['cases_sha256']:raise ValueError('fixed cohort changed')
    for arm,folder in [('native',original),('control',control)]:
        if sha(folder/'report.json') != identity['source_report_sha256'][arm]:raise ValueError('owning report changed')
        source = read(folder/'report.json')
        for name,digest in source['files_sha256'].items():
            if sha(folder/name) != digest:raise ValueError('completed prediction changed')
    retained = torch.load(run/'decoded_logits.pt',map_location='cpu',weights_only=True)
    indexed = {(r['arm'],r['case'],r['block']):r for r in report['rows']}
    checked=[];bit_disagreements=0
    for arm,folder in [('native',original),('control',control/'after')]:
        for i,case in enumerate(cases):
            path = TRAIN/case['source']
            if sha(path) != training['files_sha256'][case['source']]:raise ValueError('owning bit-packed image changed')
            raw = torch.load(path,map_location='cpu',weights_only=True)
            pixels = np.unpackbits(raw['bits'].numpy(),count=32*raw['width']).reshape(32,raw['width'])*255
            if not np.all(pixels[:,-128:]==255):raise ValueError('structural white termination differs')
            target = pixels[:,raw['prompt_blocks']*64:-128]
            target = np.stack([target[:,j:j+64] for j in range(0,target.shape[1],64)])
            block = folder/'predictions'/f'{i:02}';saved = retained[f'{arm}_{i:02}']
            if sha(block/'assisted_trace.pt') != saved['source_assisted_trace_sha256']:
                raise ValueError('owning predicted coordinates changed')
            trace = torch.load(block/'assisted_trace.pt',map_location='cpu',weights_only=True)
            if not torch.equal(trace['coordinates'],saved['coordinates']):raise ValueError('retained coordinates differ')
            actual = torch.load(block/'assisted.pt',map_location='cpu',weights_only=True)[:,0].numpy()>0
            logits = saved['predicted_logits'].double().numpy()
            # Independent FP64 sigmoid followed by the original FP32 physical
            # mapping; no Torch decoder or producer margin/scoring helper.
            probability = (1/(1+np.exp(-logits))).astype(np.float32)
            cpu = (probability*2-1)>=0
            for j in range(len(target)):
                white = target[j]==255;error = actual[j]!=white
                tp = int((~actual[j]&~white).sum());fp = int((~actual[j]&white).sum());fn = int((actual[j]&~white).sum())
                f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
                signed = (white.astype(np.float64)*2-1)*logits[j]
                numbers = signed[error]
                counts = [int(((numbers>=a)&(numbers<b)).sum()) for a,b in zip(BINS[:-1],BINS[1:])]
                losses = np.logaddexp(0.,logits[j])-white*logits[j]
                balanced = float(np.mean([losses[mask].mean() for mask in (~white,white) if mask.any()]))
                disagreement = int((cpu[j]!=actual[j]).sum());bit_disagreements+=disagreement
                row = indexed[(arm,i,j)]
                if (row['pixel_errors'] != int(error.sum()) or row['foreground_f1'] != f1
                        or row['ink_errors'] != fn or row['paper_errors'] != fp
                        or row['wrong_saved_pixel_margin_histogram'] != counts
                        or sum(counts) != int(error.sum()) or row['cpu_saved_gpu_bit_disagreements'] != disagreement
                        or not math.isclose(balanced,row['class_balanced_bce'],rel_tol=2e-6,abs_tol=1e-7)):
                    raise ValueError('independent NumPy physical-margin measurement differs')
                checked.append(dict(arm=arm,case=i,block=j,pixel_errors=int(error.sum()),
                    foreground_f1=f1,wrong_margin_histogram=counts,balanced_bce=balanced))
    if len(checked) != len(report['rows']):raise ValueError('dropped physical prediction block')
    write = dict(**committed_identity(SOURCES),source_report_sha256=sha(run/'report.json'),
        blocks_checked=len(checked),independent_bitpacked_targets=True,independent_numpy_pixel_counts=True,
        independent_numpy_margin_bins_and_bce=True,cpu_saved_gpu_bit_disagreements=bit_disagreements,
        decoder_or_full_neural_forward_replayed=False,cuda_initialized=False,language_quality_claim=False,
        general_ilm_goal_complete=False)
    (run/'verification.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps(write,indent=2))


if __name__=='__main__':main()
