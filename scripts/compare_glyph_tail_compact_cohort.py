#!/usr/bin/env python3
"""Register, run and review the earlier readable actor on the compact cohort.

The frozen actor sees only original input PNGs. References and training-source
metadata are restricted to CPU preparation/scoring, never generation.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.infer_compact_visual_span import save_grid, read_image
from scripts.infer_glyph_tail_language import load_actor, prepare_prompt
from scripts.prepare_compact_visual_span_evaluation import order

EVALUATION = ROOT / 'artifacts/compact_visual_span_evaluation_v1'
ANCESTOR = ROOT / 'artifacts/glyph_tail_adaptation_v2'
CHECKPOINT = ANCESTOR / 'fit_recovery_02/checkpoint_latest.pt'
CHECKPOINT_SHA = '5c434a9a9b0c6ddc0c3fe4005265e4f8ca5b7d0add9a74a80ed371806fa71085'
TRAIN = ROOT / 'data/visual_qa/cmrc2018_short/train.jsonl'
SOURCES = (
    'scripts/compare_glyph_tail_compact_cohort.py',
    'scripts/infer_glyph_tail_language.py', 'ilm/visual_lm/glyph_tail_adapter.py',
    'ilm/visual_lm/semantic_glyph_loop.py', 'ilm/visual_lm/image_context_interface.py',
    'ilm/visual_lm/grounded_image_actor.py', 'ilm/visual_lm/grounded_image_weights.py',
    'ilm/visual_lm/semantic_image_memory.py', 'scripts/train_semantic_glyph_language.py',
    'scripts/infer_compact_visual_span.py', 'scripts/review_compact_visual_span.py',
    'ilm/visual_lm/glyph_reader_monitor.py', 'ilm/visual_lm/glyph_runtime_resources.py',
    'ilm/visual_lm/gguf_language_trunk.py', 'ilm/visual_lm/gguf_visual_language.py',
)


def load(path):
    return json.loads(Path(path).read_text())


def sources(committed=False):
    value = {name: sha(ROOT / name) for name in SOURCES}
    if committed:
        for name, digest in value.items():
            original = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
            if hashlib.sha256(original).hexdigest() != digest:
                raise ValueError('Commit the comparison before registration: ' + name)
    return value


def context_hash(text):
    return hashlib.sha256(''.join(text.split()).encode()).hexdigest()


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or output.exists():
        raise ValueError('Register on CPU into a new output directory')
    torch.set_num_threads(2)
    code = sources(committed=True)
    prep = load(EVALUATION / 'preparation.json')
    for path, key in [('inputs/manifest.json', 'input_manifest_sha256'), ('references.json', 'references_sha256')]:
        if sha(EVALUATION / path) != prep[key]:
            raise ValueError('Frozen compact evaluation changed')
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Use the completed earlier contextual actor')
    old_path = ANCESTOR / 'preparation.json'
    old = load(old_path)
    source_name = str(TRAIN.relative_to(ROOT))
    if sha(TRAIN) != old['bindings_sha256'][source_name]:
        raise ValueError('Earlier training-source identity changed')
    raw = {row['id']: row for row in map(json.loads, TRAIN.read_text().splitlines())}
    trained = {row['id'] for row in old['training_jobs']}
    if len(trained) != 3176 or not trained <= raw.keys():
        raise ValueError('Earlier admitted questions are incomplete')
    all_passages = {context_hash(row[field]) for row in raw.values() for field in ['context', 'original_context']}
    train_passages = {context_hash(raw[key][field]) for key in trained for field in ['context', 'original_context']}
    refs = load(EVALUATION / 'references.json')['reference_rows']
    overlap = []
    for row in refs:
        if row['kind'] == 'qa':
            overlap.append(dict(case=row['id'], source_id=row['source_id'], source_group=row['group'],
                                in_recorded_training=row['source_id'] in trained,
                                in_full_training_source_pool=row['source_id'] in raw,
                                passage_in_recorded_training=row['group'] in train_passages,
                                passage_in_full_source_pool=row['group'] in all_passages))
    if len(overlap) != 32:
        raise ValueError('Preserve all32 QA questions')
    manifest = load(EVALUATION / 'inputs/manifest.json')['inputs']
    inputs = []
    for row in manifest:
        path = EVALUATION / 'inputs' / row['image']
        if Path(row['image']).name != row['image'] or sha(path) != row['image_sha256']:
            raise ValueError('Input PNG changed')
        original = read_image(path, row['glyphs'])
        canvas, observed = prepare_prompt(path)
        with Image.open(path) as source:
            if canvas.size != source.size or canvas.tobytes() != source.convert('L').tobytes():
                raise ValueError('Host preparation changed the registered binary input pixels')
        inputs.append(dict(id=row['id'], condition=row['condition'], image=row['image'],
                           glyphs=row['glyphs'], sha256=row['image_sha256']))
    if len(inputs) != 120 or sum(r['condition'] == 'normal' for r in inputs) != 44:
        raise ValueError('Keep the complete endpoint cohort')
    # Exact original image overlap with all bound admitted training prompts.
    known_pngs = {old['bindings_sha256'][job['prompt']] for job in old['training_jobs']}
    audit = dict(earlier_preparation_sha256=sha(old_path), training_source_sha256=sha(TRAIN),
                 recorded_training_questions=len(trained), complete_source_pool_questions=len(raw),
                 qa_cases=overlap,
                 recorded_question_overlap=sum(r['in_recorded_training'] for r in overlap),
                 recorded_passage_overlap=sum(r['passage_in_recorded_training'] for r in overlap),
                 all_source_question_overlap=sum(r['in_full_training_source_pool'] for r in overlap),
                 all_source_passage_overlap=sum(r['passage_in_full_source_pool'] for r in overlap),
                 exact_training_png_overlap=sum(r['sha256'] in known_pngs for r in inputs),
                 inherited_pretraining_exposure_unknown=True,
                 earlier_research_and_evaluation_familiarity=True,
                 book_exposure_in_all_earlier_training_ancestors_not_certified=True)
    output.mkdir(parents=True)
    save_json(output / 'overlap_audit.json', audit)
    request = dict(schema='glyph-tail-compact-cohort-v1', checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                   checkpoint_sha256=CHECKPOINT_SHA, source_sha256=code,
                   evaluation_preparation_sha256=sha(EVALUATION / 'preparation.json'),
                   input_manifest_sha256=sha(EVALUATION / 'inputs/manifest.json'),
                   references_sha256=prep['references_sha256'], overlap_audit_sha256=sha(output / 'overlap_audit.json'),
                   inputs=inputs, maximum_glyphs=128, deterministic_writer=True, flow_steps=None, seed=None,
                   optimizer_updates=18000, new_optimizer_updates=0,
                   retained_native_parameters=7522463984,
                   input_host_preparation='Existing image-only EXIF/grayscale/white-composite and padding; original registered inputs require no resize/padding',
                   native_observer_preparation='Unchanged native_budget_image bilinear physical observer, then existing pretrained visual interface',
                   raw_output_preparation='Lossless signed binary glyphs to uint8 ink; grid layout only; no OCR, font, reference or pixel repair',
                   maximum_seconds=3600, pressure_wait_allowance_seconds=600,
                   minimum_entry_host_ram_gib=48, minimum_runtime_host_ram_gib=24,
                   maximum_allocated_vram_gib=22, minimum_free_gpu_gib=20,
                   fixed_cold_replay_case='book_01_normal',
                   primary_metrics=['complete_exact_qa_32','both_correct_pairs_16','whole_response_readability',
                                    'book_coherent_complete_12','stopping_caps_empty','normal_minus_erased'],
                   interpretation='Descriptive matched-input control; bodies, training histories, observers and image writers differ. No causal attribution to parameter count or matched-quality efficiency.',
                   references_allowed_during_generation=False, model_calls_during_preparation=0)
    save_json(output / 'registration.json', request)
    print(json.dumps({k:audit[k] for k in ['recorded_question_overlap','recorded_passage_overlap',
                      'all_source_question_overlap','all_source_passage_overlap','exact_training_png_overlap']}, indent=2))


def validate(output):
    r = load(output / 'registration.json')
    if r['source_sha256'] != sources(committed=True) or sha(ROOT / r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Registered model or source changed')
    manifest = EVALUATION / 'inputs/manifest.json'
    if sha(manifest) != r['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    for row in r['inputs']:
        if sha(manifest.parent / row['image']) != row['sha256']:
            raise ValueError('Input image changed')
    return r


def verify_case(directory, row):
    actual = load(directory / 'report.json')
    if actual['input_sha256'] != row['sha256'] or actual['input_glyphs'] != row['glyphs']:
        raise ValueError('Retained inference input changed')
    for name, key in [('legacy_output.pt','legacy_output_sha256'), ('ink.npz','raw_sha256'), ('answer.png','image_sha256')]:
        if sha(directory / name) != actual[key]:
            raise ValueError('Retained output changed: ' + name)
    value = torch.load(directory / 'legacy_output.pt', weights_only=True, map_location='cpu')
    with np.load(directory / 'ink.npz', allow_pickle=False) as data:
        ink = torch.from_numpy(data['ink'].copy())
    signed = value['images']
    if signed.shape != ink.shape or not torch.equal(signed, 1 - 2 * ink.float()):
        raise ValueError('Binary serialization changed emitted glyphs')
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    if len(value['feedback']) != len(ink) or len(actual['emitted_pixel_feedback']) != len(ink):
        raise ValueError('Feedback coverage differs')
    for i, (original, event) in enumerate(zip(value['feedback'],actual['emitted_pixel_feedback'])):
        raw_digest = hashlib.sha256(ink[i:i+1].numpy().tobytes()).hexdigest()
        if (original['image_sha256'] != tensor_hash(signed[i:i+1]) or event['ink_sha256'] != raw_digest
                or event['continuous_updates'] != 1 or event['glyphs'] != 1
                or event['native_embedding_sha256'] != original['embedding_sha256']):
            raise ValueError('Actual signed/binary pixels or native feedback hash differs')
    if len(ink) != actual['emitted_glyphs'] or actual['stopped'] != value['stopped'] or actual['capped'] != value['capped']:
        raise ValueError('Original stopping or output count differs')
    if len(ink):
        if not torch.equal(read_image(directory / 'answer.png',len(ink)),ink):
            raise ValueError('Display differs from emitted ink')
    else:
        with Image.open(directory / 'answer.png') as image:
            if image.size!=(32,32) or image.convert('L').getextrema()!=(255,255):
                raise ValueError('An empty response must display as a white placeholder')
    return actual


def run(output):
    registration = validate(output)
    folder = output / 'generation'
    folder.mkdir(exist_ok=True)
    request = {**registration, 'registration_sha256':sha(output / 'registration.json')}
    if (folder / 'request.json').exists():
        if load(folder / 'request.json') != request:
            raise ValueError('Resume request changed')
    else:
        save_json(folder / 'request.json',request)
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_semantic_glyph_language import numerics, atomic_tensor
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    monitor = GlyphReaderMonitor(folder,registration['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds=registration['pressure_wait_allowance_seconds']
    monitor.status('admitting')
    try:
        with full_reader_gpu() as (device,admission), torch.inference_mode():
            monitor.admission=admission;numerics();torch.set_num_threads(2);monitor.guard()
            began=time.perf_counter()
            actor,provenance=load_actor(ROOT / registration['checkpoint'],device)
            loading=time.perf_counter()-began
            if any(p.requires_grad for p in actor.parameters()):raise ValueError('Frozen actor required')
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda _m,_a:monitor.guard())
            records=[]
            for i,row in enumerate(registration['inputs']):
                monitor.position={'completed':i,'total':len(registration['inputs']),'case':row['id']}
                monitor.guard();monitor.status('generating')
                directory=folder/row['id']
                if directory.exists():
                    records.append(verify_case(directory,row));continue
                path=EVALUATION/'inputs'/row['image']
                canvas,observed=prepare_prompt(path)
                prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
                # Retain the complete original native result, including actual
                # pixels, states and feedback hashes. The actor owns generation.
                torch.cuda.synchronize();began=time.perf_counter()
                value=actor.generate(prompt,maximum_glyphs=registration['maximum_glyphs'])
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                signed=value['images'];ink=((1-signed)/2).to(torch.uint8)
                if not torch.equal(signed,1-2*ink.float()):raise ValueError('Lossy output conversion')
                partial=folder/f'.{row["id"]}.{time.time_ns()}.partial';partial.mkdir()
                atomic_tensor(partial/'legacy_output.pt',value)
                np.savez_compressed(partial/'ink.npz',ink=ink.numpy())
                save_grid(ink,partial/'answer.png')
                trace=[]
                for index,event in enumerate(value['feedback']):
                    if event['image_sha256']!=tensor_hash(signed[index:index+1]):raise ValueError('Native feedback did not use emitted pixels')
                    trace.append(dict(glyphs=1,continuous_updates=1,
                        ink_sha256=hashlib.sha256(ink[index:index+1].numpy().tobytes()).hexdigest(),
                        signed_image_sha256=event['image_sha256'],native_embedding_sha256=event['embedding_sha256']))
                result=dict(id=row['id'],input_sha256=row['sha256'],input_glyphs=row['glyphs'],
                    emitted_glyphs=len(ink),stopped=value['stopped'],capped=value['capped'],elapsed_seconds=elapsed,
                    emitted_pixel_feedback=trace,legacy_output_sha256=sha(partial/'legacy_output.pt'),
                    raw_sha256=sha(partial/'ink.npz'),image_sha256=sha(partial/'answer.png'),
                    original_observer_geometry=value['observer_geometry'],host_observation=observed,
                    actual_pixel_feedback_verified=True,reference_inputs=False,output_repaired=False)
                save_json(partial/'report.json',result);partial.rename(directory)
                records.append(verify_case(directory,row))
                print(json.dumps({k:result[k] for k in ['id','emitted_glyphs','stopped','elapsed_seconds']}),flush=True)
                del value,ink,signed;gc.collect()
            hook.remove()
            save_json(folder/'report.json',dict(request_sha256=sha(folder/'request.json'),optimizer_updates=18000,
                new_optimizer_updates=0,records=records,provenance=provenance,checkpoint_load_seconds=loading,
                total_generation_seconds=sum(r['elapsed_seconds'] for r in records),reference_inputs=False,
                output_repaired=False,all_inputs_completed=len(records)==120,**monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:
        status=load(folder/'status.json');status['pid']=None;save_json(folder/'status.json',status)


def review(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU review preparation only')
    request=validate(output);run=output/'generation'
    report=load(run/'report.json')
    if report['request_sha256']!=sha(run/'request.json') or len(report['records'])!=120:
        raise ValueError('Complete the full registered set before review')
    by_id={r['id']:r for r in report['records']}
    if by_id.keys()!={r['id'] for r in request['inputs']}:raise ValueError('Output set differs')
    folder=output/'review';folder.mkdir(exist_ok=False);(folder/'pages').mkdir()
    mapping={};readings={}
    for row in request['inputs']:
        actual=verify_case(run/row['id'],row)
        opaque=order(request['checkpoint_sha256']+'/'+row['id'])[:20]
        shutil.copyfile(run/row['id']/'answer.png',folder/'pages'/f'{opaque}.png')
        if row['id'].startswith('book_') and row['condition']=='normal':
            shutil.copyfile(EVALUATION/'inputs'/row['image'],folder/'pages'/f'{opaque}_prefix.png')
        readings[opaque]=dict(text=None,uncertain=None,coherent_continuation=None)
        mapping[opaque]=dict(case=row['id'],condition=row['condition'],step=18000,
            checkpoint_sha256=request['checkpoint_sha256'],run_report_sha256=sha(run/'report.json'),
            image_sha256=actual['image_sha256'],stopped=actual['stopped'],emitted_glyphs=actual['emitted_glyphs'],
            elapsed_seconds=actual['elapsed_seconds'])
    save_json(folder/'mapping.json',dict(preparation_sha256=sha(EVALUATION/'preparation.json'),rows=mapping))
    save_json(folder/'readings_template.json',dict(evaluator=None,readings=dict(sorted(readings.items()))))
    print(json.dumps({'opaque_images':len(readings),'references_opened_for_review':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','run','prepare-review'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/glyph_tail_compact_cohort_v1')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'prepare-review':review}[args.phase](args.output)
