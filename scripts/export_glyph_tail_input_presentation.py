#!/usr/bin/env python3
"""Publish authenticated presentation results and untouched matched images."""
import json
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT,save_json,sha


def load(path):return json.loads(path.read_text())


def export():
    run=ROOT/'artifacts/glyph_tail_input_presentation_v1'
    analysis=ROOT/'artifacts/glyph_tail_input_presentation_learning_v1'
    cold=ROOT/'artifacts/glyph_tail_input_presentation_cold_v1'
    value=load(analysis/'summary.json');public=ROOT/'references/evidence'
    output=public/'glyph_tail_input_presentation_learning_20261111';output.mkdir(exist_ok=True)
    value['original_analysis_sha256']=sha(analysis/'summary.json')
    value['raw_examples_directory']=output.name
    reading=ROOT/'artifacts/glyph_tail_input_presentation_blind_review_v1/readings.json'
    shutil.copyfile(reading,output/'readings.json')
    if sha(output/'readings.json')!=value['readings_sha256']:raise ValueError('Frozen readings changed')
    original=ROOT/'artifacts/glyph_tail_compact_cohort_v1'
    old_inputs={r['id']:r for r in load(original/'registration.json')['inputs']}
    new_inputs={r['id']:r for r in load(run/'registration.json')['inputs']}
    for row in value['raw_examples']:
        key=row['case'];folder=output/'raw_examples'/key;folder.mkdir(parents=True,exist_ok=True)
        paths={
            'answer.png':run/'generation'/key/'answer.png',
            'input.png':run/'evaluation/inputs'/new_inputs[key]['image'],
            'fixed_cell_answer.png':original/'generation'/key/'answer.png',
            'fixed_cell_input.png':ROOT/'artifacts/compact_visual_span_evaluation_v1/inputs'/old_inputs[key]['image']}
        for name,source in paths.items():
            shutil.copyfile(source,folder/name)
            if sha(source)!=sha(folder/name):raise ValueError('Original example copy differs')
        if sha(folder/'answer.png')!=row['image_sha256']:raise ValueError('Actual answer changed')
        row['matched_fixed_cell_image_sha256']=sha(folder/'fixed_cell_answer.png')
        row['matched_fixed_cell_input_sha256']=sha(folder/'fixed_cell_input.png')
    cv=load(cold/'verification.json');cost=load(cold/'inference/report.json')
    cv['original_verification_sha256']=sha(cold/'verification.json')
    cv['measured_costs']={k:cost[k] for k in ['actor_loading_wall_seconds','generation_wall_seconds',
        'elapsed_seconds','resource_wait_seconds','peak_allocated_vram_gib']}
    cold_path=public/'glyph_tail_input_presentation_cold_20261111.json';save_json(cold_path,cv)
    value['public_cold_receipt']=cold_path.name;value['public_cold_receipt_sha256']=sha(cold_path)
    value['prior_control_public_receipt_sha256']=sha(public/'glyph_tail_compact_learning_20261111.json')
    value['retrospective_example_selection_disclosed']=True
    value['all_original_display_and_feedback_pixels_preserved']=True
    value['registration_public_sha256']=sha(public/'glyph_tail_input_presentation_registration_20261111.json')
    value['export_source_sha256']=sha(Path(__file__))
    path=public/'glyph_tail_input_presentation_learning_20261111.json';save_json(path,value)
    print(json.dumps(dict(metrics=[r for r in value['metrics'] if r['condition']=='normal'],
        decision_conditions=value['decision_conditions'],normal_output_counts=value['normal_output_counts']),indent=2))


if __name__=='__main__':export()
