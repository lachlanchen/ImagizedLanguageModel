#!/usr/bin/env python3
"""Freeze every autonomous answer before owner/reference scoring; no model calls."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

from scripts import evaluate_joint_writer_language as evaluation

ROOT, DEST = evaluation.ROOT, evaluation.DEST
DIRECTORY = DEST / 'review'
load, sha, require, write_new = evaluation.load, evaluation.sha, evaluation.require, evaluation.write_new
EVALS = {'fixed_cells': ROOT / 'artifacts/compact_visual_span_evaluation_v1',
         'proportional': ROOT / 'artifacts/glyph_tail_input_presentation_v1/evaluation'}


def live(value):
    pid = value.get('pid')
    if not pid:
        return False
    try:
        s = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return s[0] != 'Z' and s[19] == value['start_ticks'] and value['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError:
        return False


def completed():
    require((DEST / 'report.json').exists(), 'Complete all 708 autonomous streams before review')
    report = load(DEST / 'report.json')
    require(report['complete'] and report['total_streams'] == 708, 'Keep the complete autonomous comparison')
    status = load(DEST / 'status.json')
    require(not live(status) and status['phase'] == 'awaiting_opaque_complete_output_review',
            'Original evaluator must finish before review')
    for path in (DEST / 'attempts').glob('*/start.json'):
        require(not live(load(path)), 'Actual generation owner remains live')
        require(path.with_name('end.json').exists(), 'Reconcile every generation attempt and its cost before review')
    registration = evaluation.validate()
    require(report['registration_sha256'] == sha(evaluation.PUBLIC), 'Wrong generation registration')
    require(Path(__file__).read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:scripts/review_joint_writer_language.py'], cwd=ROOT),
        'Commit the review implementation first')
    return registration, report


def records(registration, report):
    """Enumerate exactly the prescribed input/step/cap combinations."""
    require(len(registration['arms']) == len(report['arms']) == 7, 'All seven generation arms required')
    result = []
    identities = set()
    for planned, actual in zip(registration['arms'], report['arms']):
        require(actual['arm'] == planned['arm'] and actual['step'] == planned['step'], 'Wrong generation arm')
        expected = {row['id']: row for row in planned['inputs']}
        found = {row['input']['id']: row for row in actual['records']}
        require(len(found) == len(actual['records']) and set(found) == set(expected), 'Missing or duplicate complete response')
        for key in expected:
            row = found[key]
            require(row['input'] == expected[key] and row['maximum_glyphs'] == planned['maximum_glyphs'], 'Prompt or cap changed')
            require(row['checkpoint_sha256'] == registration['snapshots'][str(planned['step'])]['sha256'], 'Head differs')
            require(row['actual_emitted_pixel_feedback'] and not row['reference_inputs'] and not row['output_repaired'],
                    'Raw emitted-pixel generation required')
            identity = (planned['step'], planned['arm'], key)
            require(identity not in identities, 'Duplicate response identity')
            identities.add(identity)
            result.append((planned, row))
    require(len(result) == 708, 'Keep all 708 complete streams')
    return result


def guard(began):
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2, 'Retain 24GiB available host RAM')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2, 'CPU review RSS ceiling reached')
    require(time.perf_counter() - began <= 900, 'CPU review operation time ceiling reached')


def prepare(reviewer, kind):
    began = time.perf_counter()
    registration, report = completed()
    require(reviewer.strip() and kind in {'model', 'human'}, 'Identify the actual reviewer')
    require(not DIRECTORY.exists(), 'Preserve previous complete-output review')
    rows = records(registration, report)
    guard(began)
    from PIL import Image, ImageDraw
    from scripts.train_visual_continuation import verify_generation
    guard(began)
    DIRECTORY.mkdir()
    pages = DIRECTORY / 'pages'
    pages.mkdir()
    mapping = {}
    for planned, record in rows:
        guard(began)
        row, step, cap = record['input'], planned['step'], record['maximum_glyphs']
        parent = DEST / f'step_{step:06}' / ('core128' if cap == 128 else 'long640') / row['id']
        actual = verify_generation(parent, row, sha(evaluation.PUBLIC), record['checkpoint_sha256'], cap)
        require(actual == record, 'Generation record changed')
        key = hashlib.sha256(f'{sha(evaluation.PUBLIC)}/{record["checkpoint_sha256"]}/{planned["arm"]}/{row["id"]}'.encode()).hexdigest()[:20]
        require(key not in mapping, 'Opaque response collision')
        os.link(parent / 'answer.png', pages / (key + '.png'))
        prefix = '__book_' in row['id'] and row['condition'] == 'normal'
        if prefix:
            require(sha(ROOT / row['image']) == row['sha256'], 'Visible book prefix changed')
            os.link(ROOT / row['image'], pages / (key + '_prefix.png'))
        mapping[key] = dict(case=row['original_case'], layout=row['layout'], condition=row['condition'], step=step,
            arm=planned['arm'], checkpoint_sha256=record['checkpoint_sha256'], input_id=row['id'],
            image_sha256=record['image_sha256'], visible_prefix=prefix,
            prefix_sha256=row['sha256'] if prefix else None, stopped=record['stopped'], capped=record['capped'],
            emitted_glyphs=record['emitted_glyphs'], elapsed_seconds=record['elapsed_seconds'])
    write_new(DIRECTORY / 'mapping.json', dict(registration_sha256=sha(evaluation.PUBLIC), rows=mapping))
    boards = []
    ids = sorted(mapping)
    for start in range(0, len(ids), 4):
        guard(began)
        chosen = ids[start:start + 4]
        images = [Image.open(pages / (key + '.png')).convert('L') for key in chosen]
        canvas = Image.new('L', (1056, sum(max(40, image.height) + 38 for image in images)), 245)
        draw = ImageDraw.Draw(canvas)
        top = 0
        for key, image in zip(chosen, images):
            draw.text((8, top + 5), key, fill=0)
            canvas.paste(image, (8, top + 28))
            top += max(40, image.height) + 38
        name = f'board_{start // 4 + 1:03}.png'
        canvas.save(DIRECTORY / name)
        board = dict(file=name, sha256=sha(DIRECTORY / name), ids=chosen,
            images={key: mapping[key]['image_sha256'] for key in chosen},
            prefixes={key: mapping[key]['prefix_sha256'] for key in chosen if mapping[key]['visible_prefix']})
        boards.append(board)
        write_new(DIRECTORY / 'templates' / (Path(name).stem + '.json'), dict(
            board_file=name, board_sha256=board['sha256'], all_outputs_actually_viewed=False,
            visible_prefixes_viewed=[], owner_mapping_opened=False, reference_answers_opened=False,
            readings={key: dict(text=None, uncertain=None, coherent_continuation=None) for key in chosen}))
    write_new(DIRECTORY / 'boards.json', dict(boards=boards, responses=708))
    write_new(DIRECTORY / 'session.json', dict(
        reviewer=reviewer, reviewer_kind=kind, registration_sha256=sha(evaluation.PUBLIC),
        generation_report_sha256=sha(DEST / 'report.json'), mapping_sha256=sha(DIRECTORY / 'mapping.json'),
        boards_sha256=sha(DIRECTORY / 'boards.json'), source_sha256=sha(Path(__file__)),
        prior_development_familiarity=True, independent_human_adjudication=False,
        new_model_calls=0, elapsed_seconds=time.perf_counter() - began))
    return dict(responses=708, boards=len(boards), new_model_calls=0, actual_review_pending=True)


def session():
    require((DIRECTORY / 'session.json').exists(), 'Prepare actual complete outputs first')
    s = load(DIRECTORY / 'session.json')
    for path, field in [(evaluation.PUBLIC, 'registration_sha256'), (DEST / 'report.json', 'generation_report_sha256'),
                        (DIRECTORY / 'boards.json', 'boards_sha256'), (Path(__file__), 'source_sha256')]:
        require(sha(path) == s[field], 'Review inputs or implementation changed')
    return s, load(DIRECTORY / 'boards.json')['boards']


def validate_readings(value, board):
    require(value.get('board_file') == board['file'] and value.get('board_sha256') == board['sha256'], 'Wrong review board')
    require(value.get('all_outputs_actually_viewed') is True and value.get('owner_mapping_opened') is False
            and value.get('reference_answers_opened') is False, 'Inspect complete opaque outputs before unmasking/references')
    require(set(value['visible_prefixes_viewed']) == set(board['prefixes'])
            and len(value['visible_prefixes_viewed']) == len(board['prefixes']), 'View every associated book prefix')
    require(set(value['readings']) == set(board['ids']), 'Every response must retain a reading')
    for key, row in value['readings'].items():
        require(set(row) == {'text', 'uncertain', 'coherent_continuation'} and isinstance(row['text'], str)
                and type(row['uncertain']) is bool, 'Complete visible reading or uncertainty required')
        require(type(row['coherent_continuation']) is bool if key in board['prefixes']
                else row['coherent_continuation'] is None, 'Coherence judgment requires the visible normal-book prefix')


def verify_board(board):
    from PIL import Image, ImageChops
    require(sha(DIRECTORY / board['file']) == board['sha256'], 'Review board changed')
    with Image.open(DIRECTORY / board['file']) as canvas:
        top = 0
        for key in board['ids']:
            path = DIRECTORY / 'pages' / (key + '.png')
            require(sha(path) == board['images'][key], 'Complete answer image changed')
            with Image.open(path) as image:
                require(canvas.width == 1056 and image.width <= 1024, 'Original uncropped response required')
                crop = canvas.crop((8, top + 28, 8 + image.width, top + 28 + image.height)).convert('L')
                require(ImageChops.difference(crop, image.convert('L')).getbbox() is None, 'Board crop differs from complete response')
                top += max(40, image.height) + 38
            if key in board['prefixes']:
                require(sha(DIRECTORY / 'pages' / (key + '_prefix.png')) == board['prefixes'][key], 'Visible prefix changed')
        require(canvas.height == top, 'Whole review board required')


def record(path):
    s, boards = session()
    require(not (DIRECTORY / 'frozen_readings.json').exists(), 'Preserve frozen readings')
    value = load(path)
    selected = [board for board in boards if board['file'] == value.get('board_file')]
    require(len(selected) == 1, 'Unknown board')
    validate_readings(value, selected[0])
    verify_board(selected[0])
    write_new(DIRECTORY / 'board_receipts' / (Path(selected[0]['file']).stem + '.json'),
              dict(value, session_sha256=sha(DIRECTORY / 'session.json'), recorded_unix=time.time()))
    return dict(recorded=selected[0]['file'])


def freeze():
    began = time.perf_counter()
    s, boards = session()
    require(not (DIRECTORY / 'frozen_readings.json').exists() and not (DIRECTORY / 'freeze_receipt.json').exists(),
            'Preserve the first frozen complete-output review')
    readings, receipts, prefixes = {}, {}, 0
    for board in boards:
        guard(began)
        path = DIRECTORY / 'board_receipts' / (Path(board['file']).stem + '.json')
        value = load(path)
        require(value['session_sha256'] == sha(DIRECTORY / 'session.json'), 'Review session differs')
        validate_readings(value, board)
        verify_board(board)
        require(not set(readings).intersection(value['readings']), 'Duplicate opaque response')
        readings.update(value['readings'])
        receipts[path.name] = sha(path)
        prefixes += len(board['prefixes'])
    require(len(readings) == 708 and len(receipts) == 177 and prefixes == 84,
            'All 708 responses, 177 boards and 84 visible normal-book prefixes required')
    require(len(receipts) == len(list((DIRECTORY / 'board_receipts').glob('*.json'))), 'Unexpected board receipts')
    # Neither owner rows nor canonical answers have been opened by this operation.
    write_new(DIRECTORY / 'frozen_readings.json', dict(
        evaluator=s['reviewer'], reviewer_kind=s['reviewer_kind'], readings=readings,
        mapping_sha256=s['mapping_sha256'], boards_sha256=s['boards_sha256'],
        session_sha256=sha(DIRECTORY / 'session.json'), board_receipt_sha256=receipts,
        all_boards_actually_viewed=True, all_visible_book_prefixes_actually_viewed=True,
        owner_mapping_opened=False, reference_answers_opened=False, frozen_unix=time.time()))
    write_new(DIRECTORY / 'freeze_receipt.json', dict(
        frozen_readings_sha256=sha(DIRECTORY / 'frozen_readings.json'),
        session_sha256=sha(DIRECTORY / 'session.json'), sealed_unix=time.time()))
    return dict(frozen_responses=708, book_prefixes=84, new_model_calls=0)


def frozen_readings():
    s, boards = session()
    seal = load(DIRECTORY / 'freeze_receipt.json')
    require(seal['frozen_readings_sha256'] == sha(DIRECTORY / 'frozen_readings.json')
            and seal['session_sha256'] == sha(DIRECTORY / 'session.json'), 'Frozen readings or seal changed')
    value = load(DIRECTORY / 'frozen_readings.json')
    require(value['session_sha256'] == seal['session_sha256'] and value['mapping_sha256'] == s['mapping_sha256'], 'Wrong frozen review')
    require(value['all_boards_actually_viewed'] is True and value['all_visible_book_prefixes_actually_viewed'] is True
            and value['owner_mapping_opened'] is False and value['reference_answers_opened'] is False,
            'Complete frozen opaque review required')
    require(set(value['board_receipt_sha256']) == {Path(b['file']).stem + '.json' for b in boards}, 'Missing frozen boards')
    expected = set()
    for board in boards:
        path = DIRECTORY / 'board_receipts' / (Path(board['file']).stem + '.json')
        require(sha(path) == value['board_receipt_sha256'][path.name], 'Frozen board reading changed')
        verify_board(board)
        expected.update(board['ids'])
    require(len(expected) == 708 and set(value['readings']) == expected, 'Missing frozen complete response')
    return value


def finalize(registration, cores, long_rows):
    """Apply the original scoring convention plus the full four-QA-erasure audit."""
    from scripts.audit_visual_continuation_protocol import pilot_checks
    final = max(map(int, registration['snapshots']))
    retention = {}
    for layout, required in registration['exact_retention'].items():
        observed = {row['case']: row for row in cores[layout]['records'] if row['step'] == final and row['condition'] == 'normal'}
        require(set(required) <= set(observed), 'Missing required retained case')
        retention[layout] = dict(required=required, retained=[key for key in required if observed[key]['complete_exact']])
    p = next(row for row in cores['proportional']['results'] if row['step'] == final and row['kind'] == 'qa' and row['condition'] == 'normal')
    erased = [row for row in cores['proportional']['results'] if row['step'] == final and row['kind'] == 'qa' and row['condition'] in ['question_blank', 'passage_blank']]
    book = next(row for row in long_rows if row['step'] == final and row['condition'] == 'normal')
    limits = registration['pilot']
    legacy = (p['complete_exact'] >= limits['proportional_exact_minimum'] and p['paired_both_correct'] >= limits['proportional_pairs_minimum']
        and p['readable'] >= limits['proportional_readable_minimum'] and book['coherent_complete'] >= limits['long_book_coherent_complete_minimum']
        and len(erased) == 2 and all(row['complete_exact'] <= limits['each_QA_erasure_exact_maximum'] for row in erased)
        and p['complete_exact'] - max(row['complete_exact'] for row in erased) >= limits['normal_minus_best_erased_exact_minimum']
        and all(row['required'] == row['retained'] for row in retention.values()))
    temporary = dict(core_results={key: value['results'] for key, value in cores.items()}, long_book_results=long_rows,
                     retention=retention, pilot_passed=legacy)
    audit = pilot_checks(dict(registration, total_steps=final), temporary)
    return dict(temporary, pilot_passed=audit['registered_protocol_pilot_passed'], legacy_proportional_route_passed=legacy,
                four_QA_erasure_audit=audit)


def score():
    began = time.perf_counter()
    registration, _ = completed()
    require(not (DIRECTORY / 'summary.json').exists(), 'Preserve the first complete-language result')
    frozen = frozen_readings()  # Must authenticate the durable seal before unmasking.
    s, _ = session()
    require(sha(DIRECTORY / 'mapping.json') == s['mapping_sha256'], 'Owner mapping changed')
    mapping = load(DIRECTORY / 'mapping.json')['rows']
    require(set(mapping) == set(frozen['readings']) and len(mapping) == 708, 'Complete owner mapping required')
    guard(began)
    from scripts.review_compact_visual_span import score as original_score, normalized
    cores = {}
    for layout, source in EVALS.items():
        guard(began)
        rows = {key: row for key, row in mapping.items() if row['layout'] == layout and row['arm'] == 'candidate_core128'}
        with tempfile.TemporaryDirectory(prefix='ilm-joint-language-score-') as name:
            temp = Path(name)
            (temp / 'pages').symlink_to((DIRECTORY / 'pages').resolve())
            write_new(temp / 'mapping.json', dict(preparation_sha256=sha(source / 'preparation.json'), rows=rows))
            write_new(temp / 'readings.json', dict(evaluator=frozen['evaluator'], readings={key: frozen['readings'][key] for key in rows}))
            with redirect_stdout(io.StringIO()):
                original_score(source, temp, temp / 'readings.json')
            cores[layout] = load(temp / 'scores.json')
        write_new(DIRECTORY / (layout + '_core_scores.json'), cores[layout])
    long = defaultdict(list)
    for key, row in mapping.items():
        if row['arm'] != 'candidate_long640':
            continue
        reading = frozen['readings'][key]
        clear = not reading['uncertain'] and bool(normalized(reading['text'])) and row['emitted_glyphs'] > 0
        long[row['step'], row['condition']].append(dict(row, readable=clear,
            coherent_complete=clear and row['stopped'] and reading['coherent_continuation'] is True))
    long_rows = []
    for (step, condition), rows in sorted(long.items()):
        require(len(rows) == 12, 'Retain all twelve long books per condition')
        long_rows.append(dict(arm='candidate_long640', step=step, condition=condition, cases=12,
            readable=sum(row['readable'] for row in rows), stopped=sum(row['stopped'] for row in rows),
            coherent_complete=sum(row['coherent_complete'] for row in rows) if condition == 'normal' else None))
    result = finalize(registration, cores, long_rows)
    require(sum(len(value['records']) for value in cores.values()) == 660 and sum(len(rows) for rows in long.values()) == 48,
            'No response may be omitted from scoring')
    control_path = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2/review/summary.json'
    control = load(control_path)
    result.update(registration_sha256=sha(evaluation.PUBLIC), frozen_readings_sha256=sha(DIRECTORY / 'frozen_readings.json'),
        reviewed_responses=708, control_reused_without_regeneration=True, control_summary_sha256=sha(control_path),
        unchanged_point_control_final_step=41298, unchanged_point_control_core_results={layout: [row for row in control['core_results'][layout] if row['step'] == 41298] for layout in EVALS},
        prior_development_familiarity=True, independent_human_adjudication=False, untouched_confirmation_required=True,
        new_model_calls=0, new_optimizer_updates=0, elapsed_seconds=time.perf_counter() - began,
        complete_output_review_does_not_prove_untouched_generalization=True, goal_achieved=False)
    guard(began)
    write_new(DIRECTORY / 'summary.json', result)
    return dict(pilot_passed=result['pilot_passed'], reviewed_responses=708, goal_achieved=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'record', 'freeze', 'score'])
    p.add_argument('--reviewer')
    p.add_argument('--reviewer-kind', choices=['model', 'human'])
    p.add_argument('--readings', type=Path)
    a = p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only complete-output review required')
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
    if a.action == 'prepare':
        require(a.reviewer and a.reviewer_kind, 'Name the actual reviewer and kind')
        result = prepare(a.reviewer, a.reviewer_kind)
    elif a.action == 'record':
        require(a.readings is not None, 'Provide the actual board readings')
        result = record(a.readings)
    else:
        result = freeze() if a.action == 'freeze' else score()
    import json
    print(json.dumps(result))


if __name__ == '__main__':
    main()
