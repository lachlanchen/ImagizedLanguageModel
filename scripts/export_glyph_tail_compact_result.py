#!/usr/bin/env python3
"""Export matched-cohort measurements without private runtime inventories."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT, save_json, sha


def load(path):return json.loads(path.read_text())


def export():
    run=ROOT/'artifacts/glyph_tail_compact_cohort_v1'
    analysis=ROOT/'artifacts/glyph_tail_compact_learning_v1'
    cold=ROOT/'artifacts/glyph_tail_compact_cold_v1'
    value=load(analysis/'summary.json')
    destination=ROOT/'references/evidence/glyph_tail_compact_learning_20261111'
    destination.mkdir(exist_ok=True)
    value['original_analysis_sha256']=sha(analysis/'summary.json')
    resources=value['generation_resources']
    value['generation_resources']={k:v for k,v in resources.items() if k!='admission'}
    value['attempt_resources']=[dict(
        {k:v for k,v in row.items() if k!='terminal_status'},
        terminal_phase=row['terminal_status']['phase'])
        for row in value['attempt_resources']]
    # Process lists, active-session commands, boot IDs and absolute traces are
    # private operational evidence. Preserve only the measured public costs.
    readings=ROOT/'artifacts/glyph_tail_compact_blind_review_v1/readings.json'
    public_readings=destination/'readings.json';shutil.copyfile(readings,public_readings)
    if sha(public_readings)!=value['readings_sha256']:raise ValueError('Frozen readings changed')
    public_cold=ROOT/'references/evidence/glyph_tail_compact_cold_20261111.json'
    cv=load(cold/'verification.json');inference=load(cold/'inference/report.json')
    cv['measured_costs']={k:inference[k] for k in ['actor_loading_wall_seconds',
        'generation_wall_seconds','elapsed_seconds','resource_wait_seconds','peak_allocated_vram_gib']}
    cv['original_verification_sha256']=sha(cold/'verification.json')
    save_json(public_cold,cv)
    value['cold_public_receipt']=public_cold.name;value['cold_public_receipt_sha256']=sha(public_cold)
    scores=load(run/'review/scores.json');inputs={r['id']:r for r in load(run/'registration.json')['inputs']}
    examples=[]
    fixed={'qa_01_a_normal','qa_01_b_normal','book_01_normal'}
    clear={r['case'] for r in scores['records'] if r['readable']}
    for key in sorted(fixed|clear):
        row=next(r for r in scores['records'] if r['case']==key)
        folder=destination/'raw_examples'/key;folder.mkdir(parents=True,exist_ok=True)
        source=run/'generation'/key/'answer.png';shutil.copyfile(source,folder/'answer.png')
        input_png=ROOT/'artifacts/compact_visual_span_evaluation_v1/inputs'/inputs[key]['image']
        shutil.copyfile(input_png,folder/'input.png')
        if sha(folder/'answer.png')!=row['image_sha256']:raise ValueError('Original answer changed')
        examples.append(dict(case=key,file=f'raw_examples/{key}/answer.png',image_sha256=row['image_sha256'],
            input_sha256=sha(input_png),emitted_glyphs=row['emitted_glyphs'],stopped=row['stopped'],
            reading=row['reading'],complete_exact=row['complete_exact'],
            display_rule='First fixed QA pair/book' if key in fixed else 'All complete readable outputs, identified after opaque review'))
    value['raw_examples']=examples;value['raw_examples_directory']=destination.name
    value['example_selection']='Three prospective fixed cases plus every complete readable output. Readable-output display selection is retrospective and does not change scoring.'
    # A CPU-only description of the actual input-distribution shift, not a
    # causal claim or an output-selected intervention.
    old=load(ROOT/'artifacts/glyph_tail_adaptation_v2/preparation.json')
    heights=[];answer_lengths=[]
    for row in old['training_jobs']:
        path=ROOT/row['prompt']
        if sha(path)!=old['bindings_sha256'][row['prompt']]:raise ValueError('Earlier input changed')
        with Image.open(path) as image:
            if image.width!=1024:raise ValueError('Unexpected earlier page width')
            heights.append(image.height)
        answer_lengths.append(len(row['labels']))
    current=[];current_answers=[]
    refs={r['id']:r for r in load(ROOT/'artifacts/compact_visual_span_evaluation_v1/references.json')['reference_rows']}
    for row in inputs.values():
        if row['condition']=='normal' and row['id'].startswith('qa_'):
            path=ROOT/'artifacts/compact_visual_span_evaluation_v1/inputs'/row['image']
            if sha(path)!=row['sha256']:raise ValueError('Current input changed')
            with Image.open(path) as image:current.append(image.height)
            current_answers.append(len(refs[row['id'].removesuffix('_normal')]['answer']))
    stats=lambda a:dict(n=len(a),minimum=min(a),quartiles=np.quantile(a,[.25,.5,.75]).tolist(),maximum=max(a))
    value['input_distribution_audit']=dict(
        previous_preparation_sha256=sha(ROOT/'artifacts/glyph_tail_adaptation_v2/preparation.json'),
        all3176_training_PNG_hashes_verified=True,width_pixels=1024,
        earlier_training_input_heights=stats(heights),current_normal_QA_input_heights=stats(current),
        earlier_training_answer_glyphs=stats(answer_lengths),current_reference_answer_glyphs=stats(current_answers),
        earlier_layout='Folded proportional text raster, safe blank-column wraps',
        current_layout='Fixed 32x32 tile cells, 32 columns, including separate Latin/digit cells',
        interpretation='Observed presentation/length differences, not a demonstrated cause. No crop around a reference answer or new model call.')
    value['compact_control_evidence_sha256']={name:sha(ROOT/'references/evidence'/name) for name in [
        'compact_visual_span_learning_20261111.json','compact_visual_span_full_rank_learning_20261111.json',
        'compact_visual_span_qwen_uncertainty_20261111.json']}
    value['export_source_sha256']=sha(Path(__file__))
    save_json(ROOT/'references/evidence/glyph_tail_compact_learning_20261111.json',value)
    normal=[r for r in value['metrics'] if r['condition']=='normal']
    print(json.dumps(dict(metrics=normal,clear_outputs=len(clear),
        examples=len(examples),input_distribution_audit=value['input_distribution_audit']),indent=2))


if __name__=='__main__':export()
