#!/usr/bin/env python3
"""Fixed-checkpoint C3 image answers, followed by output-first review/scoring.

Generation reads input PNGs and neural weights only. Reference strings and
canonical images enter the later CPU scorer, never the generation loop.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OWN = 'scripts/assess_c3_glyph_language.py'
BASELINE = ROOT / 'artifacts/semantic_glyph_language_v1'
SOURCES = (OWN, 'scripts/train_semantic_glyph_language.py',
           'ilm/visual_lm/semantic_glyph_loop.py', 'ilm/visual_lm/image_context_interface.py',
           'ilm/visual_lm/semantic_image_memory.py', 'ilm/visual_lm/grounded_image_weights.py',
           'ilm/visual_lm/grounded_image_actor.py', 'scripts/score_pixel_language_first_blocks.py',
           'ilm/image_edit/baseline.py', 'ilm/visual_lm/gguf_visual_language.py',
           'ilm/visual_lm/gguf_language_trunk.py', 'scripts/check_native_visual_transfer_v71.py',
           'scripts/train_semantic_glyph_bridge.py')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def identity():
    for name in SOURCES:
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit assessment sources first: ' + name)
    return {n: sha(ROOT / n) for n in SOURCES}


def inputs(cohort):
    """Validate PNGs and reviewed preparation without opening evaluator labels."""
    m = load(cohort / 'preparation.json')
    review = load(cohort / 'input_review.json')
    require(review['preparation_sha256'] == sha(cohort / 'preparation.json')
            and review['all16_paired_preview_boards_actually_viewed']
            and review['input_pages_approved_for_later_assessment'], 'review all actual input pages first')
    require(len(m['cases']) == 64 and len({r['case_id'] for r in m['cases']}) == 64,
            'fixed cohort must contain exactly64distinct cases')
    require(m['genuine_qualification_available_at_preparation'] is False,
            'cohort was not prospective to the readout result')
    for name, digest in m['code_sha256'].items():
        require(sha(ROOT / name) == digest, 'input preparation source changed')
    for name, digest in review['preview_files_sha256'].items():
        require(sha(cohort / name) == digest, 'reviewed input preview changed')
    for row in m['cases']:
        require(sha(cohort / row['prompt']) == row['prompt_sha256'], 'input PNG changed')
        require(row['maximum_glyphs'] == (8 if row['style'] == 'letter' else 128), 'output cap changed')
    return m


def prepare(cohort, candidate, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires CUDA hidden')
    code = identity()
    m = inputs(cohort)
    require(not output.exists(), 'preserve earlier assessments')
    require((candidate / 'qualify/report.json').exists(), 'finish genuine readout qualification first')
    for stage in ['fit', 'qualify', 'assessment']:
        status = load(candidate / stage / 'status.json')
        require(status['phase'] == 'completed' and status['pid'] is None, 'candidate stage still active: ' + stage)
    require(load(candidate / 'qualify/report.json')['readout_gate_passed'], 'genuine readout gate failed')
    candidate_meta = load(candidate / 'preparation.json')
    data_preparation = candidate_meta.get('genuine_data_preparation_sha256', sha(candidate / 'preparation.json'))
    require(data_preparation == m['prospective_genuine_preparation_sha256'],
            'candidate does not descend from the prospectively fixed data')
    lineage = {}
    training_sources = {}
    ancestors = [candidate_meta]
    if 'genuine_data_preparation_sha256' in candidate_meta:
        parent = ROOT / candidate_meta['genuine_run']
        require(candidate_meta['C3_inputs_or_references_used'] is False
                and candidate_meta['development_training'] is False, 'child used evaluation data')
        for filename, key in [('preparation.json', 'genuine_data_preparation_sha256'),
                              ('cache/report.json', 'genuine_cache_report_sha256'),
                              ('qualify/report.json', 'parent_qualification_sha256'),
                              ('qualify/visual_review.json', 'parent_visual_review_sha256')]:
            path = parent / filename
            require(sha(path) == candidate_meta[key], 'fixed data ancestry changed')
            lineage[relative(path)] = candidate_meta[key]
        ancestors.append(load(parent / 'preparation.json'))
    for ancestor in ancestors:
        for key, bindings in ancestor.items():
            if key.endswith('code_sha256') and isinstance(bindings, dict):
                for name, digest in bindings.items():
                    require(sha(ROOT / name) == digest and training_sources.get(name, digest) == digest,
                            'registered training source differs: ' + name)
                    training_sources[name] = digest
    require(load(candidate / 'assessment/status.json')['phase'] == 'completed',
            'complete original40stream assessment first')
    manual = load(candidate / 'raw_review/manual_review.json')
    require(manual['all40_actual_raw_outputs_viewed'] and manual['all10_boards_viewed']
            and (candidate / 'raw_review/canonical_scores.json').is_file(),
            'review and score the original complete streams first')
    versions = {}
    for name, run in [('baseline', BASELINE), ('candidate', candidate)]:
        checkpoint = run / 'fit/checkpoint_latest.pt'
        digest = load(run / 'fit/report.json')['checkpoint_sha256']
        require(sha(checkpoint) == digest, 'fixed checkpoint differs')
        versions[name] = dict(checkpoint=relative(checkpoint), checkpoint_sha256=digest)
    output.mkdir(parents=True)
    save(output / 'preparation.json', dict(
        code_sha256=code, cohort=relative(cohort), cohort_sha256=sha(cohort / 'preparation.json'),
        input_review_sha256=sha(cohort / 'input_review.json'), versions=versions,
        candidate_run=relative(candidate), candidate_preparation_sha256=sha(candidate / 'preparation.json'),
        candidate_data_preparation_sha256=data_preparation, candidate_data_lineage_sha256=lineage,
        candidate_training_sources_sha256=training_sources,
        candidate_qualification_sha256=sha(candidate / 'qualify/report.json'),
        candidate_original_assessment_sha256=sha(candidate / 'assessment/report.json'),
        candidate_original_manual_review_sha256=sha(candidate / 'raw_review/manual_review.json'),
        candidate_original_scores_sha256=sha(candidate / 'raw_review/canonical_scores.json'),
        evaluator_reference_sha256=m['evaluation_references_sha256'],
        primitive=load(candidate / 'preparation.json')['primitive'], maximum_seconds=2400.,
        source_pretraining_exposure_unknown=True, adaptation_after_C3_results_allowed=False,
        checkpoint_frozen_before_first_C3_model_call=True, usable_language_demonstrated=False))


def verify(output):
    meta = load(output / 'preparation.json')
    require(identity() == meta['code_sha256'], 'assessment sources changed')
    cohort = ROOT / meta['cohort']
    require(sha(cohort / 'preparation.json') == meta['cohort_sha256']
            and sha(cohort / 'input_review.json') == meta['input_review_sha256'], 'input preparation changed')
    for row in meta['versions'].values():
        require(sha(ROOT / row['checkpoint']) == row['checkpoint_sha256'], 'checkpoint changed')
    candidate = ROOT / meta['candidate_run']
    for name, digest in {**meta['candidate_data_lineage_sha256'],
                         **meta['candidate_training_sources_sha256']}.items():
        require(sha(ROOT / name) == digest, 'registered training ancestry changed: ' + name)
    for filename, key in [('preparation.json', 'candidate_preparation_sha256'),
                          ('qualify/report.json', 'candidate_qualification_sha256'),
                          ('assessment/report.json', 'candidate_original_assessment_sha256'),
                          ('raw_review/manual_review.json', 'candidate_original_manual_review_sha256'),
                          ('raw_review/canonical_scores.json', 'candidate_original_scores_sha256')]:
        require(sha(candidate / filename) == meta[key], 'registered candidate lineage changed: ' + filename)
    return meta, cohort, inputs(cohort)


def generate(output):
    meta, cohort, cohort_meta = verify(output)
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphLanguageActor
    from scripts.train_semantic_glyph_language import Monitor, reader, bridge_model, image, numerics, atomic_tensor
    from scripts.score_pixel_language_first_blocks import diagnostic_gpu
    folder = output / 'generation'
    folder.mkdir(exist_ok=False)
    monitor = Monitor(folder, meta['maximum_seconds'])
    monitor.status('admitting')
    try:
        records = []
        with diagnostic_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission
            numerics()
            monitor.guard()
            interface, provenance = reader(device)
            hook = interface.actor.context.register_forward_pre_hook(lambda module, args: monitor.guard())
            for version, fixed in meta['versions'].items():
                bridge = bridge_model({}, device, ROOT / fixed['checkpoint'])
                actor = SemanticGlyphLanguageActor(interface, bridge, language_training_complete=True).eval()
                for case in cohort_meta['cases']:
                    key = version + '/' + case['case_id']
                    monitor.position = dict(version=version, case=case['case_id'], completed=len(records), total=128)
                    monitor.guard()
                    value = actor.generate(image(cohort / case['prompt']), maximum_glyphs=case['maximum_glyphs'])
                    stem = version + '_' + case['case_id']
                    atomic_tensor(folder / (stem + '.pt'), value)
                    actual = np.full((32, max(1, len(value['images'])) * 32), 255, dtype=np.uint8)
                    for i, tile in enumerate(value['images']):
                        actual[:, i * 32:(i + 1) * 32] = ((tile[0].numpy() + 1) * 127.5).astype(np.uint8)
                    Image.fromarray(actual).save(folder / (stem + '.png'))
                    records.append(dict(key=key, version=version, case_id=case['case_id'],
                        tensor=stem + '.pt', tensor_sha256=sha(folder / (stem + '.pt')),
                        png=stem + '.png', png_sha256=sha(folder / (stem + '.png')),
                        glyphs=len(value['images']), stopped=value['stopped'], capped=value['capped']))
                    monitor.status('generating_C3_image_answers')
                    print(json.dumps(records[-1]), flush=True)
                del actor, bridge
            hook.remove()
            save(folder / 'report.json', dict(preparation_sha256=sha(output / 'preparation.json'),
                records=records, provenance=provenance, references_read=False, tokenizer=False,
                source_vocabulary_head=False, runtime_font=False, runtime_glyph_bank=False,
                actual_emitted_pixel_feedback=True, quality_not_yet_reviewed=True, **monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed', reason=repr(error))
        raise
    finally:
        status = load(folder / 'status.json')
        save(folder / 'status.json', dict(status, pid=None))


def review(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU raw review requires CUDA hidden')
    meta, cohort, cohort_meta = verify(output)
    import torch
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    folder = output / 'raw_review'
    require(load(output / 'generation/status.json')['phase'] == 'completed', 'finish every generated stream first')
    report = load(output / 'generation/report.json')
    require(len(report['records']) == 128, 'incomplete assessment')
    cases = {r['case_id']: r for r in cohort_meta['cases']}
    require(report['preparation_sha256'] == sha(output / 'preparation.json')
            and {r['key'] for r in report['records']} ==
            {version + '/' + case for version in meta['versions'] for case in cases},
            'generation does not cover the fixed models and cases exactly')
    displays = {}
    for row in report['records']:
        p = output / 'generation' / row['tensor']
        require(sha(p) == row['tensor_sha256'], 'raw tensor changed')
        v = torch.load(p, weights_only=True, map_location='cpu')
        tiles = v['images']
        require(tiles.shape == (row['glyphs'], 1, 32, 32)
                and bool(((tiles == -1) | (tiles == 1)).all()), 'invalid glyph pixels')
        require(len(v['feedback']) == len(tiles) and len(v['stop_probabilities']) == len(tiles) + 1
                and len(v['states']) == len(tiles) + 1, 'feedback/decision counts differ')
        require(all(tensor_hash(tiles[i:i + 1]) == f['image_sha256'] for i, f in enumerate(v['feedback'])),
                'recorded feedback differs from actual pixels')
        prob = v['stop_probabilities']
        require(all(t < .5 for t in prob[:-1]) and (prob[-1] >= .5) == row['stopped']
                and row['stopped'] != row['capped'], 'autonomous stopping differs')
        require(not row['capped'] or len(tiles) == cases[row['case_id']]['maximum_glyphs'], 'wrong output cap')
        strip = np.full((32, max(1, len(tiles)) * 32), 255, np.uint8)
        wrapped = Image.new('L', (1024, max(1, (len(tiles) + 31) // 32) * 32), 255)
        for i, tile in enumerate(tiles):
            a = ((tile[0].numpy() + 1) * 127.5).astype(np.uint8)
            strip[:, i * 32:(i + 1) * 32] = a
            wrapped.paste(Image.fromarray(a), ((i % 32) * 32, (i // 32) * 32))
        png = output / 'generation' / row['png']
        require(sha(png) == row['png_sha256'] and np.array_equal(np.asarray(Image.open(png)), strip),
                'display PNG differs from raw output')
        displays[row['key']] = wrapped
    folder.mkdir(exist_ok=False)
    boards = []
    for document in range(16):
        rows = [r for r in report['records'] if cases[r['case_id']]['document_index'] == document]
        board = Image.new('L', (1036, sum(displays[r['key']].height + 28 for r in rows)), 245)
        draw = ImageDraw.Draw(board)
        y = 0
        for row in rows:
            draw.text((6, y + 3), f"{row['key']} glyphs={row['glyphs']} stop={row['stopped']} cap={row['capped']}", fill=0)
            board.paste(displays[row['key']], (6, y + 22))
            y += displays[row['key']].height + 28
        name = f'document_{document:02}.png'
        board.save(folder / name)
        boards.append(dict(file=name, sha256=sha(folder / name)))
    save(folder / 'report.json', dict(generation_report_sha256=sha(output / 'generation/report.json'),
        records=report['records'], boards=boards, references_read=False,
        all_pixel_and_feedback_image_hashes_verified=True, output_pixels_repaired=False,
        board_wrapping_preserves_every_complete32x32cell=True, actual_visual_review_pending=True))
    print(json.dumps(dict(raw_streams=128, boards=len(boards), references_read=False)))


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU scorer requires CUDA hidden')
    meta, cohort, cohort_meta = verify(output)
    folder = output / 'raw_review'
    report = load(folder / 'report.json')
    manual = load(folder / 'manual_review.json')
    require(report['generation_report_sha256'] == sha(output / 'generation/report.json'),
            'reviewed generation report changed')
    require(manual['raw_review_report_sha256'] == sha(folder / 'report.json')
            and manual['all16_boards_actually_viewed'], 'record actual output-first board review')
    for row in report['boards']:
        require(sha(folder / row['file']) == row['sha256'], 'reviewed board changed')
    require(sha(cohort / 'evaluation_references.json') == meta['evaluator_reference_sha256'], 'evaluator labels changed')
    references = {r['case_id']: r['expected'] for r in load(cohort / 'evaluation_references.json')}
    import torch
    torch.set_num_threads(2)
    primitive = ROOT / meta['primitive']
    pm = load(primitive / 'preparation.json')
    require(sha(primitive / 'training.pt') == pm['training_sha256']
            and sha(primitive / 'inventory.json') == pm['inventory_sha256'], 'canonical evaluator changed')
    data = torch.load(primitive / 'training.pt', weights_only=True, map_location='cpu')
    inventory = load(primitive / 'inventory.json')['records']
    charmap = {c: i for i, r in enumerate(inventory) for c in r['chars']}
    cases = {r['case_id']: r for r in cohort_meta['cases']}
    require(set(manual['output_first_readings']) == {r['key'] for r in report['records']},
            'retain a reading or explicit uncertainty for every stream')
    measured = []
    for row in report['records']:
        p = output / 'generation' / row['tensor']
        require(sha(p) == row['tensor_sha256'], 'raw output changed')
        value = torch.load(p, weights_only=True, map_location='cpu')
        a = (value['images'] < 0).to(torch.uint8)[:, 0].numpy()
        text = references[row['case_id']]
        unsupported = sorted(set(text) - charmap.keys())
        f1, exact = None, False
        if not unsupported:
            g = np.stack([data['ink'][charmap[c], 0].numpy() for c in text])
            n = max(len(a), len(g))
            aa = np.zeros((n, 32, 32), np.uint8)
            gg = aa.copy()
            aa[:len(a)], gg[:len(g)] = a, g
            denominator = int(aa.sum() + gg.sum())
            f1 = 2 * int((aa & gg).sum()) / denominator if denominator else 1.
            exact = bool(row['stopped'] and len(a) == len(g) and np.array_equal(aa, gg))
        reading = manual['output_first_readings'][row['key']]
        require(isinstance(reading['uncertain'], bool) and
                (reading['text'] is None or isinstance(reading['text'], str)), 'invalid manual reading')
        measured.append(dict(**row, style=cases[row['case_id']]['style'],
            document_index=cases[row['case_id']]['document_index'], expected=text,
            unsupported_reference_chars=unsupported, foreground_f1=f1, canonical_complete_exact=exact,
            manual_clear_complete_exact=bool(row['stopped'] and not reading['uncertain'] and reading['text'] == text),
            output_first_reading=reading))
    summaries = {}
    for version in meta['versions']:
        for style in ['letter', 'phrase']:
            rows = [r for r in measured if r['version'] == version and r['style'] == style]
            summaries[version + '/' + style] = dict(cases=len(rows),
                canonical_complete_exact=sum(r['canonical_complete_exact'] for r in rows),
                manual_clear_complete_exact=sum(r['manual_clear_complete_exact'] for r in rows),
                stopped=sum(r['stopped'] for r in rows), capped=sum(r['capped'] for r in rows),
                both_option_orders_manual_clear_correct=sum(
                    all(r['manual_clear_complete_exact'] for r in rows if r['document_index'] == i)
                    for i in range(16)))
    save(folder / 'scores.json', dict(records=measured, summary=summaries,
        manual_review_sha256=sha(folder / 'manual_review.json'), reference_bank_evaluator_only=True,
        runtime_answers_replaced=False, approximate_nearest_glyph_decoding=False,
        independent_human_grading=False, formal_general_language_accuracy_established=False,
        source_pretraining_exposure_unknown=True, model_selection_not_performed_by_evaluator=True))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['check-inputs', 'prepare', 'generate', 'review', 'score'], required=True)
    parser.add_argument('--cohort', type=Path, default=ROOT / 'artifacts/c3_visual_reading_v1')
    parser.add_argument('--candidate-run', type=Path, default=ROOT / 'artifacts/genuine_glyph_readout_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/c3_glyph_language_assessment_v1')
    args = parser.parse_args()
    if args.stage == 'check-inputs':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU check requires CUDA hidden')
        identity()
        m = inputs(args.cohort.resolve())
        print(json.dumps(dict(cases=len(m['cases']), actual_input_review_verified=True,
                              evaluator_references_read=False, model_calls=0)))
    elif args.stage == 'prepare':
        prepare(args.cohort.resolve(), args.candidate_run.resolve(), args.output.resolve())
    else:
        {'generate': generate, 'review': review, 'score': score}[args.stage](args.output.resolve())
