#!/usr/bin/env python3
"""Recount all generated occupancy images and causal traces without a model."""
import json
import os
from pathlib import Path

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/occupancy_density_preflight_v1'
OUT=ROOT/'references/evidence/occupancy_density_preflight_20261118'
PUBLIC=ROOT/'references/evidence/occupancy_density_preflight_result_20261118.json'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU recount only')
    import numpy as np
    import torch
    from PIL import Image
    from ilm.visual_lm.occupancy_glyph_density import OccupancySequence,decode_occupancy
    from scripts.measure_reader_writing_consequence import pixels,pool
    regpath=ROOT/'references/evidence/occupancy_density_preflight_registration_20261118.json'
    reg=original.load(regpath);raw=original.load(RUN/'result.json')
    original.require(raw['registration_sha256']==original.sha(regpath),'Registration changed')
    original.require(not OUT.exists() and not PUBLIC.exists(),'Preserve every export')
    exitpath=ROOT/'artifacts/occupancy_density_preflight_v1.exit.json';terminal=original.load(exitpath)
    original.require(terminal['phase']=='terminal' and terminal['exit_code']==0,'Neural comparison not complete')
    for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
    cachepath=ROOT/'artifacts/layer_conditioned_writer_preflight_v1/conditional_training_cache.pt'
    cache=torch.load(cachepath,map_location='cpu',weights_only=True)
    inputs={str(p.relative_to(ROOT)):original.sha(p) for p in [regpath,RUN/'result.json',exitpath,cachepath]}
    exports=[];audits=[];initial_roots={};OUT.mkdir()
    for arm in raw['arms']:
        original.require(arm['exact_reset'] and arm['source_cache_unchanged'],'Trial/source reset failed')
        original.require(len(arm['ledger'])==96 and [r['case'] for r in arm['ledger']]==[cache[j%3]['case'] for j in range(96)],'Complete source order differs')
        original.require(sum(r['positions'] for r in arm['ledger'])==32*552,'Whole-record fitting exposure differs')
        for stage in ['initial','final']:
            checked=[];roots=[]
            for case in arm[stage]['cases']:
                imagepath=ROOT/case['raw_file'];tracepath=ROOT/case['trace_file']
                original.require(original.sha(imagepath)==case['raw_sha256'] and original.sha(tracepath)==case['trace_sha256'],'Actual image/trace bytes changed')
                inputs[str(imagepath.relative_to(ROOT))]=original.sha(imagepath);inputs[str(tracepath.relative_to(ROOT))]=original.sha(tracepath)
                ink=np.load(imagepath)['ink'];original.require(ink.dtype==np.uint8 and np.isin(ink,[0,1]).all(),'Physical output is not binary')
                target=next(v['ink'] for v in cache if v['case']==case['case'])
                actual=pixels(torch.from_numpy(ink),target);original.require(actual==case['pixels'],'Image metric recount differs')
                traces=torch.load(tracepath,weights_only=True,map_location='cpu');restored=[];maximum=0.;events=0
                for batch,trace in zip(traces['batches'],traces['actual_trace']):
                    seq=OccupancySequence(**{n:batch[n] for n in ['actions','depths','rows','columns','valid','allow_zero']})
                    original.require(torch.equal(seq.valid,trace['active']) and torch.equal(seq.actions[seq.valid],trace['actions'][seq.valid]),'Trace is not the actual emitted sequence')
                    meta=torch.stack((seq.depths,seq.rows,seq.columns),-1)
                    original.require(torch.equal(meta[seq.valid],trace['meta'][seq.valid]),'Actual generated topology differs')
                    restored.append(decode_occupancy(seq,32,arm['mode']))
                    error=float((batch['cached_logp'][seq.valid]-batch['parallel_logp'][seq.valid]).abs().max())
                    original.require(error<=reg['maximum_replay_logp_error'],'Stored probability replay exceeds bound')
                    maximum=max(maximum,error);events+=int(seq.valid.sum())
                original.require(np.array_equal(torch.cat(restored).numpy(),ink),'Trace does not decode to every emitted pixel')
                original.require(maximum==case['maximum_replay_logp_error'],'Replay maximum report differs')
                columns=min(32,len(ink));rows=(len(ink)+columns-1)//columns;canvas=np.full((rows*36,columns*36),255,dtype=np.uint8)
                for i,tile in enumerate(ink[:,0]):canvas[(i//columns)*36:(i//columns)*36+32,(i%columns)*36:(i%columns)*36+32]=255*(1-tile)
                png=OUT/f'{arm["mode"]}_{stage}_{case["case"]["role"]}.png';Image.fromarray(canvas).save(png);pixels_back=np.asarray(Image.open(png))
                for i,tile in enumerate(ink[:,0]):original.require(np.array_equal(pixels_back[(i//columns)*36:(i//columns)*36+32,(i%columns)*36:(i%columns)*36+32],255*(1-tile)),'PNG changed a generated pixel')
                exports.append(dict(file=str(png.relative_to(ROOT)),sha256=original.sha(png),mode=arm['mode'],stage=stage,case=case['case']['role'],positions=len(ink),columns=columns,tile_size=32,gutter=4))
                audits.append(dict(mode=arm['mode'],stage=stage,case=case['case']['role'],pixels=actual,all_trace_pixels_exact=True,
                                   physical_events=events,mean_generated_events=events/len(ink),maximum_stored_replay_logp_error=maximum))
                checked.append(dict(pixels=actual));roots.append(torch.from_numpy(ink).bool().flatten(1).any(1))
            original.require(pool(checked)==arm[stage]['pooled_pixels'],'Pooled metrics differ')
            if stage=='initial':initial_roots[arm['mode']]=torch.cat(roots)
    original.require(torch.equal(initial_roots['flat'],initial_roots['tree']),'Common initial root decisions differ')
    flat,tree=raw['arms'];t=tree['final']['pooled_pixels'];f=flat['final']['pooled_pixels'];s=reg['support']
    checks=dict(pooled_f1=t['f1']>=s['pooled_f1'],every_case_f1=all(v['pixels']['f1']>=s['every_case_f1'] for v in tree['final']['cases']),
                gain_over_flat=t['f1']-f['f1']>=s['gain_over_flat'],false_blank=t['false_blank']==0,false_ink=t['false_ink']==0,
                conditional_dependence=tree['final']['pooled_rotated_minus_correct_nll']>=s['rotated_minus_correct_nll'])
    original.require(checks==raw['support_checks'] and all(checks.values())==raw['support'],'Support verdict differs')
    result=dict(raw);result['resources']=dict(raw['resources']);result['resources'].pop('admission',None)
    result.update(input_sha256=inputs,audits=audits,exports=exports,all_2208_generated_images_audited=True,
                  exporter_sha256=original.sha(Path(__file__)),stored_probability_replay_scope='Recounted persisted cached/parallel log probabilities; no new neural evaluation',
                  nll_scope='Original immutable neural execution; no independent neural NLL recount')
    original.write_new(PUBLIC,result)
    print(json.dumps(dict(public=str(PUBLIC.relative_to(ROOT)),sha256=original.sha(PUBLIC),support=result['support'],images=2208,exports=len(exports))))


if __name__=='__main__':main()
