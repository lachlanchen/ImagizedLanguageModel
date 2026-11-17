#!/usr/bin/env python3
"""Audit every saved tile and export the complete paired writing evidence."""
import hashlib
import json
import os
from pathlib import Path

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/reader_writing_consequence_v1'
OUT=ROOT/'references/evidence/reader_writing_consequence_20261118'
PUBLIC=ROOT/'references/evidence/reader_writing_consequence_result_20261118.json'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU evidence export only')
    import numpy as np
    import torch
    from PIL import Image
    from scripts.train_full_rank_visual_span import training_source
    from scripts.measure_reader_writing_consequence import pixels,pool
    regpath=ROOT/'references/evidence/reader_writing_consequence_registration_20261118.json'
    reg=original.load(regpath);raw=original.load(RUN/'result.json')
    original.require(raw['registration_sha256']==original.sha(regpath),'Registration changed')
    original.require(not OUT.exists() and not PUBLIC.exists(),'Preserve prior export')
    exitpath=ROOT/'artifacts/reader_writing_consequence_v1.exit.json'
    terminal=original.load(exitpath)
    original.require(terminal['phase']=='terminal' and terminal['exit_code']==0,'Execution not complete')
    for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Source changed: '+n)
    corpus,*_=training_source(ROOT/reg['teacher'])
    data=torch.load(ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt',weights_only=True,map_location='cpu')
    cache=torch.load(ROOT/'artifacts/nonlinear_feedback_alignment_v1/execution/frozen_queries.pt',weights_only=True,map_location='cpu')
    OUT.mkdir()
    inputs={str(regpath.relative_to(ROOT)):original.sha(regpath),
            str((RUN/'result.json').relative_to(ROOT)):original.sha(RUN/'result.json'),
            str(exitpath.relative_to(ROOT)):original.sha(exitpath)}
    exports=[];audits=[]
    for arm in raw['arms']:
        checked=[]
        for case in arm['cases']:
            p=ROOT/case['raw_file'];q=ROOT/case['live_queries_file']
            original.require(original.sha(p)==case['raw_sha256'] and original.sha(q)==case['live_queries_sha256'],'Saved output changed')
            inputs[str(p.relative_to(ROOT))]=original.sha(p);inputs[str(q.relative_to(ROOT))]=original.sha(q)
            saved=np.load(p);ink=saved['ink']
            original.require(ink.dtype==np.uint8 and np.isin(ink,[0,1]).all(),'Invalid physical output')
            row=corpus[case['case']['corpus_index']]
            first=max(row['prompt_spans']-1,0)
            start=int(row['span_glyph_ends'][first])
            gold=row['ink'][start:]
            actual=pixels(torch.from_numpy(ink),gold)
            original.require(actual==case['pixels'],'Complete pixel metric mismatch')
            lengths=saved['span_lengths'].tolist();guesses=saved['length_argmax']
            original.require(sum(lengths)==len(ink) and len(guesses)==len(lengths)+1,'Span coverage changed')
            original.require(int((guesses==np.array([*lengths,0])).sum())==case['length_correct'],'Length report differs')
            prepared=next(v for v in data['cases'] if v['case']==case['case'])['rows']
            live=torch.load(q,weights_only=True,map_location='cpu')
            valid=torch.arange(3)[None]<live['update_counts'][:,None]
            visible=live['queries'][valid].float()
            cached=cache[data['row_image'][prepared],data['row_slot'][prepared]].float()
            original.require(visible.shape==cached.shape,'Query coverage changed')
            query_rms=float((visible-cached).square().mean().sqrt()/visible.square().mean().sqrt().clamp_min(1e-8))
            # A lossless layout: each native binary tile occupies one 32x32 cell.
            # Four white gutter pixels separate cells; none are part of a tile.
            columns=min(32,len(ink));rows=(len(ink)+columns-1)//columns
            canvas=np.full((rows*36,columns*36),255,dtype=np.uint8)
            for i,tile in enumerate(ink[:,0]):canvas[(i//columns)*36:(i//columns)*36+32,(i%columns)*36:(i%columns)*36+32]=255*(1-tile)
            png=OUT/f'{arm["arm"]}_{case["case"]["role"]}.png'
            Image.fromarray(canvas).save(png)
            # Verify every tile against its exact source array after PNG reload.
            restored=np.asarray(Image.open(png))
            for i,tile in enumerate(ink[:,0]):
                original.require(np.array_equal(restored[(i//columns)*36:(i//columns)*36+32,(i%columns)*36:(i%columns)*36+32],255*(1-tile)),'PNG changed a generated pixel')
            exports.append(dict(file=str(png.relative_to(ROOT)),sha256=original.sha(png),
                                arm=arm['arm'],case=case['case']['role'],positions=len(ink),columns=columns,tile_size=32,gutter=4))
            audits.append(dict(arm=arm['arm'],case=case['case']['role'],pixels=actual,
                live_cached_raw_query_relative_rms=query_rms,live_cached_feedback_relative_rms=case['live_cached_feedback_relative_rms'],
                png_matches_every_saved_tile=True,reference_history=True,supplied_lengths=True))
            checked.append(dict(pixels=actual))
        original.require(pool(checked)==arm['pooled_pixels'],'Pooled metrics differ')
    result=dict(raw)
    result['resources']=dict(raw['resources']);result['resources'].pop('admission',None)
    result.update(raw_result_sha256=original.sha(RUN/'result.json'),input_sha256=inputs,
        audit=audits,exports=exports,exporter_sha256=original.sha(Path(__file__)),
        all_1104_generated_tiles_audited=True,all_png_pixels_lossless=True,
        initial_noise_matching_scope='Generation receipt hashes agree; noise arrays were not independently replayed',
        interpretation='No pooled physical improvement; failed reader gate remains. Canonical agreement is not readability or language accuracy.')
    original.write_new(PUBLIC,result)
    print(json.dumps(dict(public=str(PUBLIC.relative_to(ROOT)),sha256=original.sha(PUBLIC),positions=1104,exports=len(exports))))


if __name__=='__main__':main()
