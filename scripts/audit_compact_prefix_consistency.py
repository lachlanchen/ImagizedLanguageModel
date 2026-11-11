#!/usr/bin/env python3
"""Isolate the stopped diagnostic's causal-prefix consistency failure."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
from scripts.diagnose_compact_visual_span import DATA, CODEC, LOCAL, digest_tensor, relative_mse, context_cosine, image_metrics
from scripts.infer_compact_visual_span import load_checkpoint, save_grid

PARENT=ROOT/'artifacts/compact_visual_span_diagnosis_v1'
CASE='development_book_25'


def load(path):return json.loads(Path(path).read_text())


def compare(a,b):
    return {'relative_mse':relative_mse(a,b),'relative_rms':relative_mse(a,b)**.5,
            'cosine':context_cosine(a,b),'maximum_absolute_error':float((a.float()-b.float()).abs().max()),
            'bit_equal':torch.equal(a,b),'shape':list(a.shape)}


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():raise ValueError('New CPU registration required')
    original=load(PARENT/'registration.json');case=next(c for c in original['cases'] if c['id']==CASE)
    checkpoint=original['checkpoints']['v1'];output.mkdir(parents=True)
    source_names=['scripts/audit_compact_prefix_consistency.py',*original['sources_sha256']]
    request=dict(parent_registration_sha256=sha(PARENT/'registration.json'),case=case,checkpoint=checkpoint,
                 sources_sha256={s:sha(ROOT/s) for s in source_names},codec_sha256=sha(CODEC),
                 corpus_report_sha256=sha(DATA/'report.json'),source_vectors_sha256=sha(PARENT/'source_vectors.pt'),
                 previous_execution_exit_code=1,previous_execution_elapsed_seconds=load(PARENT/'execution/status.json')['elapsed_seconds'],
                 interventions=['full_vs_prefix_body','mask_and_cache_flags','encoder_batch_extent','codec_batch_extent',
                                'same_shape_future_erasure','BF16_vs_FP32_context_arithmetic'],
                 maximum_body_calls=16,maximum_velocity_calls=32,maximum_seconds=240,
                 precision_note='FP32 body uses the same BF16-rounded weights cast to FP32; other weights unchanged',
                 seed=202611157,optimizer_updates=0,autonomous_language_test=False,model_calls_during_preparation=0)
    save_json(output/'registration.json',request);print(json.dumps(request,indent=2))


@torch.inference_mode()
def run(output):
    req=load(output/'registration.json')
    for name,digest in req['sources_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Probe source changed')
    for p,digest in [(ROOT/req['checkpoint']['path'],req['checkpoint']['sha256']),
                     (CODEC,req['codec_sha256']),(DATA/'report.json',req['corpus_report_sha256']),
                     (PARENT/'source_vectors.pt',req['source_vectors_sha256'])]:
        if sha(p)!=digest:raise ValueError('Probe input changed')
    execution=output/'execution';execution.mkdir(exist_ok=False)
    monitor=GlyphReaderMonitor(execution,req['maximum_seconds']);monitor.pressure_wait_allowance_seconds=60
    monitor.status('admitting')
    try:
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2);monitor.guard()
            model,updates,provenance=load_checkpoint(ROOT/req['checkpoint']['path'],LOCAL,CODEC,device)
            case=req['case'];corpus=VisualSequenceCorpus(DATA,split=case['split'])
            row=corpus[case['record_index']];aligned=corpus.offline_teacher_alignment(case['record_index'])
            if digest_tensor(row['ink'])!=case['ink_sha256']:raise ValueError('Case pixels changed')
            ends=row['span_glyph_ends'].tolist();sizes=[e-s for s,e in zip([0,*ends[:-1]],ends)]
            n=case['prefix_spans'];cut=ends[n-1]
            counts=torch.diff(aligned['span_unit_ends'],prepend=aligned['span_unit_ends'].new_zeros(1)).to(device).long()
            mask=torch.arange(3,device=device)[None]<counts[:,None];prefix_units=int(counts[:n].sum())
            latents=model.glyph_latents(row['ink'].to(device));prefix_latents=model.glyph_latents(row['ink'][:cut].to(device))
            full_spans=latents.split(sizes)
            full_vectors,_=model.encoder(full_spans)
            prefix_shared,_=model.encoder(full_spans[:n])
            prefix_own,_=model.encoder(prefix_latents.split(sizes[:n]))
            full=full_vectors[mask][None];shared=prefix_shared[mask[:n]][None];own=prefix_own[mask[:n]][None]
            exact_prefix=full[:,:prefix_units]
            result={'codec_prefix_batch_extent':compare(prefix_latents,latents[:cut]),
                    'encoder_same_latents_batch_extent':compare(shared,exact_prefix),
                    'encoder_own_prefix_batch_extent':compare(own,exact_prefix)}
            def body(inputs,cache=False,attention=False):
                monitor.guard();kw={'attention_mask':torch.ones(inputs.shape[:2],device=device,dtype=torch.bool)} if attention else {}
                return model._body(inputs,use_cache=cache,**kw).last_hidden_state[:,min(prefix_units,inputs.shape[1])-1]
            a=body(full,attention=True);b=body(exact_prefix,cache=True)
            c=body(exact_prefix,attention=True);d=body(shared,cache=True);e=body(own,cache=True)
            erased=full.clone();erased[:,prefix_units:]=0
            f=body(erased,attention=True)
            native=torch.load(PARENT/'source_vectors.pt',weights_only=True,map_location='cpu')[CASE].to(device)[None]
            na=body(native,attention=True);nb=body(native[:,:prefix_units],cache=True)
            result.update(full_vs_identical_prefix_BF16=compare(b,a),prefix_mask_cache_BF16=compare(c,b),
                          encoder_batch_effect_on_context_BF16=compare(d,b),
                          original_diagnostic_difference_BF16=compare(e,a),
                          same_shape_future_erasure_BF16=compare(f,a),native_full_vs_prefix_BF16=compare(nb,na))
            gold=row['ink'][cut:ends[n]];save_grid(gold,execution/'reference.png')
            images={}
            for name,state in [('full_reference',a),('identical_prefix',b),('reencoded_prefix',e)]:
                gen=torch.Generator(device=device).manual_seed(req['seed'])
                ink=model.write_span(state,len(gold),8,gen).cpu()
                save_grid(ink,execution/(name+'.png'));images[name]=dict(**image_metrics(ink,gold),sha256=sha(execution/(name+'.png')))
            model.context.to(dtype=torch.float32)
            fa=body(full,attention=True);fb=body(exact_prefix,cache=True);fe=body(own,cache=True)
            ff=body(erased,attention=True)
            result.update(full_vs_identical_prefix_FP32=compare(fb,fa),
                          original_diagnostic_difference_FP32=compare(fe,fa),
                          same_shape_future_erasure_FP32=compare(ff,fa))
            save_json(execution/'report.json',dict(registration_sha256=sha(output/'registration.json'),
                      case=CASE,prefix_glyphs=cut,prefix_native_updates=prefix_units,total_glyphs=len(row['ink']),
                      comparisons=result,first_span_images=images,optimizer_updates=0,body_calls=12,velocity_calls=24,
                      float32_matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:monitor.status('failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','run'))
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/compact_prefix_consistency_v1')
    args=p.parse_args();prepare(args.output) if args.phase=='prepare' else run(args.output)
