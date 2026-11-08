#!/usr/bin/env python3
"""Export a complete, inspected native-flow copying cohort without generation.

This audit authenticates the saved pixels and the owning OCR calibration. It
does not estimate training gains or replace the frozen 384-image assessment.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.native_flow_assessment import authenticate_native_png
from scripts.audit_native_flow_normal_cohort import load, require
from scripts.evaluate_image_edit_baseline import normalize


def audit(assessment, review, output, arm):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for offline assessment')
    require(not output.exists(), 'use a fresh output directory')
    import torch
    torch.set_num_threads(4)
    prep = load(assessment/'preparation.json')
    training = Path(prep['training_run'])
    meta = load(training/'preparation.json')
    require(sha(training/'preparation.json') == prep['training_preparation_sha256'],
            'training preparation changed')
    for path, value in {**meta['code_sha256'], **prep['assessment_code_sha256']}.items():
        require(sha(ROOT/path) == value, 'frozen source changed: '+path)
    name = arm+'_copy'
    folder = assessment/(name+'_inference')
    native = load(folder/'report.json')
    outputs = load(folder/'outputs.json')
    index = training/'evaluation/copy_inputs.json'
    manifest = training/'evaluation/copy_references.jsonl'
    given = load(index)
    refs = [json.loads(line) for line in manifest.read_text().splitlines()]
    for path in (index, manifest):
        require(sha(path) == prep['evaluation_files_sha256'][path.name],
                'fixed copying evaluation data changed')
    scores_path = assessment/(name+'_ocr/report.json')
    scores = load(scores_path)
    calibration_path = Path(prep['calibrations']['copy']['path'])
    calibration = load(calibration_path)
    require(sha(calibration_path) == prep['calibrations']['copy']['sha256'], 'calibration changed')
    require(scores['stage'] == 'score' and calibration['stage'] == 'calibrate'
            and scores['complete_cohort'] and calibration['complete_cohort']
            and scores['manifest_sha256'] == calibration['manifest_sha256'] == sha(manifest)
            and scores['reference_calibration_sha256'] == sha(calibration_path)
            and scores['engine'] == calibration['engine']
            and scores['implementation'] == calibration['implementation'],
            'owning OCR calibration differs')
    for path, value in scores['implementation'].items():
        require(sha(ROOT/path) == value, 'offline evaluator changed')
    deployment_path = training/'deployment/deployment.json'
    deployment = load(deployment_path)
    require(sha(training/'deployment/adapters.safetensors') == deployment['weights_sha256'],
            'exported adapters changed')
    require(native['arm'] == ('final' if arm == 'final' else 'initial_source')
            and native['learned_adapter_weights_loaded'] == (arm == 'final')
            and native['weights_sha256'] == (deployment['weights_sha256'] if arm == 'final' else None)
            and native['exported_weights_file_sha256'] == deployment['weights_sha256']
            and native['source'] == deployment['source_sha256']
            and native['code_sha256'] == meta['code_sha256']
            and native['deployment_sha256'] == sha(deployment_path)
            and native['inputs_sha256'] == sha(index)
            and native['complete_generations'] == 64, 'complete original deployment/cohort required')
    require(deployment['steps'] == 50 and deployment['guidance'] == 4.
            and deployment['input_canvas'] == deployment['output_canvas'] == [1024, 1024]
            and not deployment['runtime_lexical_vocabulary'], 'native geometry or sampler differs')
    require(not any(native[k] for k in ('training_manifest_loaded', 'optimizer_loaded',
            'target_or_transcript_used', 'changing_text_condition', 'output_repair',
            'external_teacher_or_OCR')), 'input-only inference boundary differs')
    require(all(native['removal'][k] for k in ('vocabulary_table_removed',
            'output_word_head_removed', 'tokenizer_removed')), 'deployed lexical modules remain')
    ids = [r['id'] for r in given]
    require(len(ids) == len(set(ids)) == 64 and all([r['id'] for r in group] == ids
            for group in (refs, outputs, scores['rows'], calibration['rows'])),
            'complete original-order copying cohort required')
    receipt = load(review/'receipt.json')
    inspection = load(review/'raw_inspection.json')
    require(receipt['source_mode'] == 'completed_assessment_cohorts' and receipt['cases'] == 64
            and inspection['review_receipt_sha256'] == sha(review/'receipt.json')
            and inspection['all64_original_pages_actually_viewed']
            and not inspection['manual_accuracy_assigned'], 'complete bounded raw inspection required')
    require(sha(review/'private_key.json') == receipt['private_key_sha256'], 'review key changed')
    key = load(review/'private_key.json')
    require(set(key['groups']) == {name}
            and key['groups'][name]['index_sha256'] == sha(folder/'outputs.json')
            and key['groups'][name]['native_report_sha256'] == sha(folder/'report.json'),
            'review source cohort differs')
    packet = load(review/'packet/index.json')
    require(len(packet) == len(key['cases']) == len(inspection['records']) == 64,
            'all64 reviewed bindings required')
    require(len({r['review_id'] for r in packet}) == 64
            and {r['source_id'] for r in key['cases']} == set(ids), 'review coverage differs')
    actual = {r['id']: r for r in outputs}
    for visible, hidden, seen in zip(packet, key['cases'], inspection['records']):
        out = actual[hidden['source_id']]
        require(visible['review_id'] == hidden['review_id'] == seen['review_id']
                and seen['image_sha256'] == visible['image_sha256'] == hidden['answer_sha256']
                == out['sha256'] and hidden['actual_tensor_sha256'] == out['result']['tensor_sha256']
                and seen['original_page_viewed'], 'reviewed page differs')
    for rel, value in receipt['packet_files_sha256'].items():
        require(sha(review/'packet'/rel) == value, 'review packet changed')
    require(set(inspection['viewed_contact_sheets_sha256']) ==
            {f'pages/page_{i:03}.png' for i in range(1, 17)}, 'all16 contact sheets required')
    for rel, value in inspection['viewed_contact_sheets_sha256'].items():
        require(sha(review/'packet'/rel) == value, 'viewed contact sheet changed')
    rows, copies = [], []
    for i, (inp, out, ref, score, clean) in enumerate(zip(given, outputs, refs,
                                                       scores['rows'], calibration['rows'])):
        require(set(inp) == {'id', 'input', 'input_sha256', 'seed'}
                and all(out[k] == inp[k] for k in ('id', 'input_sha256', 'seed'))
                and sha(index.parent/inp['input']) == inp['input_sha256'] == ref['input_sha256']
                and out['sha256'] == out['result']['sha256'] == score['png_sha256']
                and score['expected'] == clean['expected'] == ref['reference_text']
                and ref['task'] == 'copy' and ref['split'] == score['split'] == clean['split']
                and ref['language'] == score['language'] == clean['language']
                and sha(manifest.parent/ref['target']) == ref['target_sha256'],
                'actual input/output/reference binding differs')
        signature = authenticate_native_png(folder/out['path'], out['sha256'],
                                             out['result']['tensor_sha256'])
        text = normalize(score['raw_transcript'], ref['language'])
        canonical = text == normalize(ref['reference_text'], ref['language'])
        require(text == score['normalized'] and canonical == score['ocr_exact'], 'OCR score differs')
        require(normalize(clean['raw_transcript'], ref['language']) == clean['normalized']
                and clean['ocr_exact'] == (clean['normalized'] ==
                    normalize(ref['reference_text'], ref['language'])), 'reference OCR score differs')
        images = {kind: f'images/{i:03}_{kind}.png' for kind in ('input', 'answer', 'reference')}
        copies.extend([(index.parent/inp['input'], images['input'], inp['input_sha256']),
                       (folder/out['path'], images['answer'], out['sha256']),
                       (manifest.parent/ref['target'], images['reference'], ref['target_sha256'])])
        rows.append(dict(id=ref['id'], split=ref['split'], language=ref['language'], seed=inp['seed'],
            input_sha256=inp['input_sha256'], PNG_sha256=out['sha256'],
            tensor_sha256=out['result']['tensor_sha256'], actual_pixel_signature=signature,
            images=images, expected=ref['reference_text'], raw_transcript=score['raw_transcript'],
            normalized=text, canonical_exact=canonical, reference_OCR_exact=clean['ocr_exact']))
    require([r['split'] for r in rows] == ['train']*32+['validation']*32, 'fixed split order differs')
    summary = {}
    for split in ('train', 'validation'):
        selected = [r for r in rows if r['split'] == split]
        summary[split] = dict(cases=len(selected), canonical_exact=sum(r['canonical_exact'] for r in selected),
            reference_OCR_exact=sum(r['reference_OCR_exact'] for r in selected), languages={})
        for language in ('zh', 'en'):
            group = [r for r in selected if r['language'] == language]
            require(len(group) == 16, 'fixed language balance differs')
            summary[split]['languages'][language] = dict(cases=len(group),
                canonical_exact=sum(r['canonical_exact'] for r in group),
                reference_OCR_exact=sum(r['reference_OCR_exact'] for r in group))
        require(scores['summary'][split] == dict(cases=32, ocr_exact=summary[split]['canonical_exact'])
                and calibration['summary'][split] == dict(cases=32,
                    ocr_exact=summary[split]['reference_OCR_exact']), 'OCR summary differs')
    require(not torch.cuda.is_initialized(), 'offline audit initialized CUDA')
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for path, rel, value in copies:
        target = output/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == value, 'exported original PNG bytes differ')
        files[rel] = value
    save_json(output/'raw_inspection.json', inspection)
    files['raw_inspection.json'] = sha(output/'raw_inspection.json')
    save_json(output/'report.json', dict(cohort=name, cases=64, summary=summary, rows=rows,
        implementation_sha256=sha(Path(__file__)),
        review_binding_utility_sha256=sha(ROOT/'scripts/audit_native_flow_normal_cohort.py'),
        assessment_preparation_sha256=sha(assessment/'preparation.json'),
        native_report_sha256=sha(folder/'report.json'), output_index_sha256=sha(folder/'outputs.json'),
        OCR_report_sha256=sha(scores_path), calibration_sha256=sha(calibration_path),
        deployment_sha256=sha(deployment_path), final_checkpoint_sha256=deployment['final_checkpoint_sha256'],
        files_sha256=files, all_original_input_answer_reference_PNG_bytes_unchanged=True,
        all_answer_PNGs_authenticated_against_actual_decoder_tensors=True,
        complete_copy_cohort_raw_review=True, manual_accuracy_assigned=False,
        full384_assessment_completed=False, initial_final_learning_gain_established=False,
        semantic_learning_validated=False, general_ilm_goal_complete=False))
    print(json.dumps(dict(cohort=name, summary=summary, generated_new_images=0), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--arm', choices=('initial', 'final'), default='final')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.assessment.resolve(), args.review.resolve(), args.output.resolve(), args.arm)
