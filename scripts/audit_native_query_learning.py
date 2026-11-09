#!/usr/bin/env python3
"""Audit the completed two-update reader-query preflight without a model call."""
import argparse
import json
import os
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.native_flow_assessment import authenticate_native_png


def audit(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for the saved-tensor audit')
    import torch
    torch.set_num_threads(4)
    run = ROOT/'artifacts/native_query_learning_preflight_v1'
    execution = run/'execution'
    sampling = ROOT/'artifacts/native_flow_positive_sampling_probe_v1/final'
    load = lambda p: json.loads(p.read_text())
    prep, report = load(run/'preparation.json'), load(execution/'report.json')
    cases = load(run/'cases.json')
    before, after = load(sampling/'outputs.json')[:2], load(execution/'answers/outputs.json')
    score, calibration = load(execution/'ocr/report.json'), load(execution/'calibration/report.json')
    for path, digest in {**prep['code_sha256'], **prep['bound_predecessors_sha256']}.items():
        if sha(ROOT/path) != digest:
            raise ValueError('bound source changed: '+path)
    if (sha(run/'preparation.json') != report['preparation_sha256']
            or sha(run/'cases.json') != prep['cases_sha256']
            or sha(execution/'checkpoint.pt') != report['query_checkpoint_sha256']
            or sha(execution/'answers/outputs.json') != report['outputs_sha256']
            or len(cases) != 2 or len(after) != 2
            or len(report['history']) != 2 or report['actual_optimizer_updates'] != 2
            or not all(report[key] for key in ('native_helper_features_exact',
                'differentiated_features_exact', 'initial_native_field_exact',
                'all_frozen_weight_versions_unchanged_during_updates'))
            or not score['complete_cohort']):
        raise ValueError('complete original preflight bindings required')
    references = [json.loads(line) for line in (execution/'references.jsonl').read_text().splitlines()]
    if (len(references) != 2 or score['cohort'] != [r['id'] for r in cases]
            or [r['id'] for r in references] != score['cohort']
            or score['reference_calibration_sha256'] != sha(execution/'calibration/report.json')
            or score['reference_calibration_summary'] != calibration['summary']
            or score['summary']['train'] != {'cases': 2, 'ocr_exact': 0}
            or score['reference_calibration_summary']['train'] != {'cases': 2, 'ocr_exact': 2}):
        raise ValueError('registered two-case score or reference calibration differs')
    checkpoint = torch.load(execution/'checkpoint.pt', map_location='cpu', weights_only=True)
    initial, final = checkpoint['initial_queries'], checkpoint['queries']
    states = list(checkpoint['optimizer']['state'].values())
    if (initial.shape != (128, 2048) or final.dtype != torch.float32
            or not torch.isfinite(final).all() or len(states) != 1
            or int(states[0]['step']) != 2 or checkpoint['completed_updates'] != 2):
        raise ValueError('actual query optimizer state differs')
    update_rows = []
    for row in report['history']:
        path = execution/row['actual_tensor_file']
        if sha(path) != row['actual_tensor_sha256']:
            raise ValueError('saved update tensor file changed')
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        if not all(torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError('nonfinite saved training tensor')
        if (not torch.equal(tensors['noisy'], tensors['noise'])
                or not torch.equal(tensors['target_velocity'], tensors['noise']-tensors['clean'])):
            raise ValueError('saved full-noise supervision differs')
        mse = float((tensors['predicted_velocity'].float()-tensors['target_velocity'].float()).square().mean())
        if mse != row['loss']:
            raise ValueError('saved actual velocity loss does not reproduce')
        grad = tensors['query_gradient']
        update_rows.append(dict(step=row['step'],id=row['id'],actual_tensor_file_sha256=sha(path),
            nonzero_gradient_coordinates=int(torch.count_nonzero(grad)),
            total_query_coordinates=grad.numel(),actual_velocity_MSE=mse,
            all_saved_tensors_finite=True,writer_input_exactly_independent_noise=True))
        if row['step'] == 2 and not torch.equal(tensors['queries_after'], final):
            raise ValueError('checkpoint does not match final saved update')
    output.mkdir(parents=True, exist_ok=False)
    (output/'images').mkdir()
    observed = []
    for i, (case, old, new, ref, scored) in enumerate(zip(cases,before,after,references,score['rows'])):
        if (not case['id'] == old['id'] == new['id'] == ref['id'] == scored['id']
                or old['input_sha256'] != new['input_sha256'] or old['seed'] != new['seed']
                or new['sha256'] != scored['png_sha256']):
            raise ValueError('matched original input, seed or OCR row differs')
        images = {}
        for name, base, value in [('before',sampling,old),('after',execution/'answers',new)]:
            path = base/value['path']
            signature = authenticate_native_png(path,value['sha256'],value['result']['tensor_sha256'])
            images[name] = dict(path=f'images/{i:02}_{name}.png',sha256=sha(path),
                tensor_file_sha256=value['result']['tensor_sha256'],actual_pixel_signature=signature)
            shutil.copyfile(path,output/images[name]['path'])
        target = Path(case['target'])
        if sha(target) != case['target_sha256']:
            raise ValueError('original target changed')
        images['reference'] = dict(path=f'images/{i:02}_reference.png',sha256=sha(target))
        shutil.copyfile(target,output/images['reference']['path'])
        observed.append(dict(id=case['id'],input_sha256=case['own_sha256'],sampling_seed=new['seed'],
            reference_text=ref['reference_text'],raw_OCR_transcript=scored['raw_transcript'],
            generated_OCR_exact=scored['ocr_exact'],images=images,
            PNG_changed=old['sha256'] != new['sha256']))
    if torch.cuda.is_initialized():
        raise ValueError('offline audit initialized CUDA')
    save_json(output/'audit.json',dict(implementation_sha256=sha(__file__),
        source_preparation_sha256=sha(run/'preparation.json'),source_execution_report_sha256=sha(execution/'report.json'),
        source_checkpoint_sha256=sha(execution/'checkpoint.pt'),source_cases_sha256=sha(run/'cases.json'),
        source_scoring_report_sha256=sha(execution/'ocr/report.json'),
        source_calibration_report_sha256=sha(execution/'calibration/report.json'),
        source_matched_baseline_sha256=sha(sampling/'outputs.json'),
        all_22_bound_source_files_unchanged=True,actual_optimizer_updates=2,
        FP32_changed_query_coordinates=int(torch.count_nonzero(final != initial)),
        BF16_changed_query_coordinates=int(torch.count_nonzero(final.bfloat16() != initial.bfloat16())),
        maximum_absolute_query_change=float((final-initial).abs().max()),updates=update_rows,rows=observed,
        generated_OCR_exact=0,reference_OCR_exact=2,cases=2,
        retained_runtime_parameters=report['retained_runtime_parameters']['total'],
        peak_allocated_vram_GiB=report['peak_allocated_vram_GiB'],elapsed_seconds=report['elapsed_seconds'],
        neural_models_loaded=0,new_optimizer_updates=0,new_generations=0,
        raw_image_review_completed=False,semantic_learning_validated=False,general_ilm_goal_complete=False))
    shutil.copyfile(execution/'report.json',output/'execution.json')
    shutil.copyfile(execution/'ocr/report.json',output/'ocr.json')
    # This is the original numerical report; original source files remain untouched.
    print('Audited2 actual updates and4 original generated pages; generated answers0/2.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output.resolve())
