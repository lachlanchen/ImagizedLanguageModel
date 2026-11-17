#!/usr/bin/env python3
"""Descriptive source-frequency strata of saved physical reconstruction errors."""
import json
import os
from pathlib import Path

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
OUT=ROOT/'references/evidence/binary_codec_frequency_20261118.json'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Saved-array CPU analysis only')
    original.require(not OUT.exists(),'Preserve every analysis')
    import fcntl
    import numpy as np
    import torch
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2)
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources();original.require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,'Frequency analysis reserve')
        source=ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt';part=ROOT/'artifacts/binary_codec_preparation_v1/partition.pt'
        public=ROOT/'references/evidence/binary_codec_comparison_result_20261118.json';result=original.load(public)
        reg=original.load(ROOT/'references/evidence/binary_codec_comparison_registration_20261118.json')
        for p in [source,part]:original.require(original.sha(p)==reg['input_sha256'][str(p.relative_to(ROOT))],'Source changed')
        data=torch.load(source,weights_only=True,map_location='cpu');partition=torch.load(part,weights_only=True,map_location='cpu')
        target=np.unpackbits(data['packed_ink'].numpy(),axis=1,bitorder='big').reshape(-1,1024).astype(bool)
        group=partition['group'].numpy();held=partition['held'].numpy();counts=np.bincount(group[~held],minlength=len(partition['group_mask_sha256']))
        strata=[('once',1,1),('two_to_four',2,4),('five_to_sixteen',5,16),('seventeen_to_sixtyfour',17,64),('sixtyfive_plus',65,10**9)]
        inputs={str(p.relative_to(ROOT)):original.sha(p) for p in [source,part,public,Path(__file__)]};reports=[]
        for arm in result['arms']:
            p=ROOT/arm['final']['raw_file'];original.require(original.sha(p)==arm['final']['raw_sha256'],'Saved pixels changed');inputs[str(p.relative_to(ROOT))]=original.sha(p)
            with np.load(p) as saved:ink=saved['ink'].reshape(-1,1024).astype(bool)
            tp=(ink&target).sum(1);fp=(ink&~target).sum(1);fn=(~ink&target).sum(1);den=2*tp+fp+fn
            f1=np.where(den>0,2*tp/np.maximum(1,den),1.);exact=(ink==target).all(1)
            sums=np.bincount(group,weights=f1,minlength=len(counts));exact_sums=np.bincount(group,weights=exact,minlength=len(counts));allcounts=np.bincount(group,minlength=len(counts))
            rows=[]
            for name,low,high in strata:
                selected=(counts>=low)&(counts<=high);forms=int(selected.sum());cells=int(counts[selected].sum())
                if forms:rows.append(dict(stratum=name,minimum_training_occurrences=low,maximum_training_occurrences=None if high==10**9 else high,
                    physical_forms=forms,training_cells=cells,macro_form_f1=float((sums[selected]/allcounts[selected]).mean()),
                    macro_form_exact_rate=float((exact_sums[selected]/allcounts[selected]).mean())))
            reports.append(dict(mode=arm['mode'],training_strata=rows))
        original.write_new(OUT,dict(schema='ilm-binary-codec-frequency-v1',input_sha256=inputs,reports=reports,
            scope='Post-hoc descriptive training-frequency bins on immutable saved source images; no causal identification, quality-rule change or language score',
            neural_calls=0,optimizer_updates=0,gpu_used=False,full_language_fit_authorized=False,goal_achieved=False))
        print(json.dumps(dict(output=str(OUT.relative_to(ROOT)),sha256=original.sha(OUT),reports=reports)),flush=True)


if __name__=='__main__':main()
