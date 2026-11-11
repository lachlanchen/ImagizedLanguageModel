#!/usr/bin/env python3
"""Use presentation-specific opaque IDs without relabeling the frozen outputs."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.compare_glyph_tail_input_presentation import validate,verify_case,load


def prepare(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only review preparation')
    import torch
    torch.set_num_threads(2)
    request=validate(run);generation=run/'generation';report=load(generation/'report.json')
    if len(report['records'])!=96 or not report['all_inputs_completed']:
        raise ValueError('Finish all registered cases before review')
    if report['request_sha256']!=sha(generation/'request.json'):raise ValueError('Generation request changed')
    observed={r['id']:r for r in report['records']}
    if set(observed)!={r['id'] for r in request['inputs']}:raise ValueError('Output set differs')
    folder=run/'review';folder.mkdir(exist_ok=False);(folder/'pages').mkdir()
    mapping={};readings={}
    for row in request['inputs']:
        actual=verify_case(generation/row['id'],row)
        if actual!=observed[row['id']]:raise ValueError('Raw generation report differs')
        opaque=hashlib.sha256(('proportional-input-review-v1\0'+sha(run/'registration.json')+
            '\0'+request['checkpoint_sha256']+'/'+row['id']).encode()).hexdigest()[:20]
        if opaque in mapping:raise ValueError('Opaque ID collision')
        shutil.copyfile(generation/row['id']/'answer.png',folder/'pages'/(opaque+'.png'))
        mapping[opaque]=dict(case=row['id'],condition=row['condition'],step=18000,
            checkpoint_sha256=request['checkpoint_sha256'],run_report_sha256=sha(generation/'report.json'),
            image_sha256=actual['image_sha256'],stopped=actual['stopped'],
            emitted_glyphs=actual['emitted_glyphs'],elapsed_seconds=actual['elapsed_seconds'])
        readings[opaque]=dict(text=None,uncertain=None,coherent_continuation=None)
    save_json(folder/'mapping.json',dict(preparation_sha256=sha(run/'evaluation/preparation.json'),rows=mapping))
    save_json(folder/'readings_template.json',dict(evaluator=None,readings=dict(sorted(readings.items()))))
    save_json(folder/'preparation_receipt.json',dict(images=96,registration_sha256=sha(run/'registration.json'),
        generation_report_sha256=sha(generation/'report.json'),source_sha256=sha(Path(__file__)),
        namespace='proportional-input-review-v1 plus registration and checkpoint',
        previous_comparison_opaque_IDs_not_reused=True,reference_text_opened=False,
        outputs_repaired=False))
    print('Prepared96 presentation-specific opaque output images; references not opened')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_tail_input_presentation_v1')
    a=p.parse_args();prepare(a.run.resolve())
