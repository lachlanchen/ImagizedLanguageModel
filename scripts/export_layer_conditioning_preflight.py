#!/usr/bin/env python3
"""CPU recount and lossless full-output export of the fixed writer comparison."""
import json
import os
from pathlib import Path

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/layer_conditioned_writer_preflight_v1'
OUT=ROOT/'references/evidence/layer_conditioned_writer_preflight_20261118'
PUBLIC=ROOT/'references/evidence/layer_conditioned_writer_preflight_result_20261118.json'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU evidence export only')
    import numpy as np
    import torch
    from PIL import Image
    from scripts.measure_reader_writing_consequence import pixels,pool
    regpath=ROOT/'references/evidence/layer_conditioned_writer_preflight_registration_20261118.json'
    reg=original.load(regpath);raw=original.load(RUN/'result.json')
    original.require(raw['registration_sha256']==original.sha(regpath),'Registration changed')
    original.require(not OUT.exists() and not PUBLIC.exists(),'Preserve prior evidence')
    exitpath=ROOT/'artifacts/layer_conditioned_writer_preflight_v1.exit.json'
    terminal=original.load(exitpath)
    original.require(terminal['phase']=='terminal' and terminal['exit_code']==0,'Execution not complete')
    for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
    cachepath=RUN/'conditional_training_cache.pt'
    original.require(original.sha(cachepath)==raw['conditional_cache_sha256'],'Actual image/context cache changed')
    cache=torch.load(cachepath,weights_only=True,map_location='cpu')
    inputs={str(p.relative_to(ROOT)):original.sha(p) for p in [regpath,RUN/'result.json',cachepath,exitpath]}
    exports=[];audits=[];ink_by={}
    OUT.mkdir()
    for arm in raw['arms']:
        original.require(arm['exact_reset'] and arm['source_writer_unchanged'] and arm['frozen_actor_unchanged'] and arm['length_head_unchanged'],'Trial or frozen-source check failed')
        for stage in ['initial','final']:
            checked=[]
            for case in arm[stage]['cases']:
                p=ROOT/case['raw_file'];original.require(original.sha(p)==case['raw_sha256'],'Saved output changed')
                inputs[str(p.relative_to(ROOT))]=original.sha(p)
                saved=np.load(p);ink=saved['ink']
                original.require(ink.dtype==np.uint8 and np.isin(ink,[0,1]).all(),'Output is not binary ink')
                record=next(v for v in cache if v['case']==case['case'])
                actual=pixels(torch.from_numpy(ink),record['ink'])
                original.require(actual==case['pixels'],'Complete pixel metrics differ')
                original.require(saved['span_lengths'].sum()==len(ink),'Missing span positions')
                columns=min(32,len(ink));rows=(len(ink)+columns-1)//columns
                canvas=np.full((rows*36,columns*36),255,dtype=np.uint8)
                for i,tile in enumerate(ink[:,0]):canvas[(i//columns)*36:(i//columns)*36+32,(i%columns)*36:(i%columns)*36+32]=255*(1-tile)
                png=OUT/f'{arm["arm"]}_{stage}_{case["case"]["role"]}.png';Image.fromarray(canvas).save(png)
                restored=np.asarray(Image.open(png))
                for i,tile in enumerate(ink[:,0]):original.require(np.array_equal(restored[(i//columns)*36:(i//columns)*36+32,(i%columns)*36:(i%columns)*36+32],255*(1-tile)),'PNG changed actual ink')
                ink_by[arm['arm'],stage,case['case']['role']]=ink
                exports.append(dict(file=str(png.relative_to(ROOT)),sha256=original.sha(png),arm=arm['arm'],stage=stage,
                                    case=case['case']['role'],positions=len(ink),columns=columns,tile_size=32,gutter=4))
                audits.append(dict(arm=arm['arm'],stage=stage,case=case['case']['role'],pixels=actual,every_exported_pixel_exact=True))
                checked.append(dict(pixels=actual))
            original.require(pool(checked)==arm[stage]['pooled_pixels'],'Pooled metrics differ')
        original.require(len(arm['ledger'])==12 and sum(v['positions'] for v in arm['ledger'])==4*552,'Whole-record exposure differs')
    a,b=raw['arms']
    original.require(all(np.array_equal(ink_by[a['arm'],'initial',v['role']],ink_by[b['arm'],'initial',v['role']]) for v in reg['cases']),'Identity initialization pixel mismatch')
    original.require([v['noise_time_sha256'] for v in a['ledger']]==[v['noise_time_sha256'] for v in b['ledger']],'Recorded training noise differs')
    base=a['initial']['pooled_pixels'];control=a['final']['pooled_pixels'];candidate=b['final']['pooled_pixels']
    checks=dict(gain_over_initial=candidate['f1']-base['f1']>=.05,gain_over_control=candidate['f1']-control['f1']>=.02,
                no_case_deterioration=all(x['pixels']['f1']>=y['pixels']['f1'] for x,y in zip(b['final']['cases'],a['final']['cases'])),
                no_additional_whitespace_errors=all(candidate[k]<=min(base[k],control[k]) for k in ['false_blank','false_ink']))
    original.require(checks==raw['support_checks'] and all(checks.values())==raw['support'],'Support decision differs')
    result=dict(raw);result['resources']=dict(raw['resources']);result['resources'].pop('admission',None)
    result.update(input_sha256=inputs,audits=audits,exports=exports,all_2208_generated_tiles_audited=True,
        all_initial_pixels_identical=True,exporter_sha256=original.sha(Path(__file__)),
        scope='Complete training-record physical support and exact trial reset; no autonomous language or generalization result')
    original.write_new(PUBLIC,result)
    print(json.dumps(dict(public=str(PUBLIC.relative_to(ROOT)),sha256=original.sha(PUBLIC),support=result['support'],positions=2208,exports=len(exports))))


if __name__=='__main__':main()
