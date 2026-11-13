#!/usr/bin/env python3
"""Freeze opaque physical readings before opening the joint-writer owner map.

This CPU-only workflow does not run a model or assign language scores. Review
claims still require actual inspection of every board by the named reviewer.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/continuation_joint_writer_v1'
PUBLIC = ROOT / 'references/evidence/continuation_joint_writer_registration_20261113.json'
GROUPS = {
    ('train', 'qa', 'fixed_cells'), ('train', 'qa', 'proportional'),
    ('train', 'book', 'fixed_cells'), ('train', 'book', 'proportional'),
    ('development', 'qa', 'fixed_cells'), ('development', 'qa', 'proportional'),
    ('development', 'book', 'fixed_cells'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    """Publish a complete JSON atomically; never replace prior evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink()


def validate_readings(readings, ids):
    require(isinstance(readings, dict) and set(readings) == set(ids),
            'Every opaque tile must have exactly one reading')
    for key, row in readings.items():
        require(set(row) == {'text', 'uncertain', 'blank'}, 'Unexpected reading fields: ' + key)
        require(isinstance(row['text'], str) and type(row['uncertain']) is bool
                and type(row['blank']) is bool, 'Complete typed readings required: ' + key)
        require(not row['blank'] or not row['text'].strip(), 'Blank tile cannot have a transcription')
        require(row['blank'] or row['uncertain'] or bool(row['text'].strip()),
                'An unreadable nonblank tile must retain uncertainty')


def board_inventory(qualify, manifest):
    require(isinstance(manifest['boards'], list) and manifest['boards'], 'No review boards')
    all_ids = []
    files = []
    for board in manifest['boards']:
        name = board['file']
        require(re.fullmatch(r'board_\d{3}\.png', name), 'Unexpected board filename')
        require(sha(qualify / name) == board['sha256'], 'Reviewed board changed')
        require(0 < len(board['ids']) <= 64, 'Invalid board size')
        require(all(re.fullmatch(r'[0-9a-f]{20}', key) for key in board['ids']), 'Invalid opaque ID')
        files.append(name)
        all_ids.extend(board['ids'])
    require(len(files) == len(set(files)) and len(all_ids) == len(set(all_ids)),
            'Duplicate board or physical tile')
    require(len(all_ids) == manifest['images'], 'Incomplete review inventory')
    return all_ids


def context():
    """Authenticate completed stages, without opening owner rows or targets."""
    registration = load(PUBLIC)
    require(sha(PUBLIC) == sha(RUN / 'registration.json'), 'Registration differs')
    for name, digest in registration['source_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound model source changed: ' + name)
    controller = load(RUN / 'controller_status.json')
    require(controller.get('pid') is None and controller['phase'] == 'awaiting_opaque_physical_review',
            'Wait for the original pipeline to finish qualification')
    qualify = RUN / 'qualify'
    require(load(qualify / 'status.json').get('pid') is None, 'Qualification worker remains live')
    require(Path(__file__).read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:scripts/review_joint_writer_physical.py'], cwd=ROOT),
        'Commit the review workflow before reviewing actual outputs')
    fit, audit, report = [load(RUN / folder / 'report.json') for folder in ['fit', 'audit', 'qualify']]
    for value in [fit, audit, report]:
        require(value['complete'] and value['registration_sha256'] == sha(PUBLIC), 'Incomplete or foreign stage')
    require(audit['fit_report_sha256'] == sha(RUN / 'fit/report.json'), 'Audited fit changed')
    jobs = registration['qualification_jobs']
    expected = {job['id']: job for job in jobs}
    require(len(expected) == len(jobs) == 124, 'All 124 qualification jobs required')
    actual = {row['job']['id']: row['job'] for row in report['records']}
    require(len(report['records']) == 124 and actual == expected and report['all124_complete'],
            'Qualification cohort differs from registration')
    require(report['boards_sha256'] == sha(qualify / 'boards.json'), 'Board manifest changed')
    require(registration['physical_gate_requires_all_groups'] is True
            and registration['physical_outputs_reviewed_before_owner_unmask'] is True,
            'Preserve the registered physical review rules')
    manifest = load(qualify / 'boards.json')
    ids = board_inventory(qualify, manifest)
    bindings = {
        'registration_sha256': sha(PUBLIC), 'qualification_sha256': sha(qualify / 'report.json'),
        'fit_sha256': sha(RUN / 'fit/report.json'), 'audit_sha256': sha(RUN / 'audit/report.json'),
        'boards_sha256': sha(qualify / 'boards.json'), 'mapping_sha256': report['mapping_sha256'],
        'review_source_sha256': sha(Path(__file__)),
    }
    return qualify, registration, report, manifest, ids, bindings


def session_context():
    values = context()
    session = load(values[0] / 'review/session.json')
    require(session['bindings'] == values[-1], 'Review inputs or code changed after preparation')
    return values, session


def prepare(reviewer, kind):
    qualify, _, _, manifest, ids, bindings = context()
    require(reviewer.strip() and kind in {'model', 'human'}, 'Identify the actual reviewer')
    destination = qualify / 'review'
    destination.mkdir(exist_ok=False)
    write_new(destination / 'session.json', dict(
        schema='joint-writer-opaque-review-session-v1', created_unix=time.time(), bindings=bindings,
        reviewer=reviewer, reviewer_kind=kind, prior_development_familiarity=True,
        independent_human_adjudication=False, reviewer_declares_mapping_unopened=True))
    for board in manifest['boards']:
        write_new(destination / 'templates' / (Path(board['file']).stem + '.json'), dict(
            board_file=board['file'], board_sha256=board['sha256'],
            all_outputs_actually_viewed=False, owner_mapping_opened=False,
            readings={key: dict(text=None, uncertain=None, blank=None) for key in board['ids']}))
    return dict(boards=len(manifest['boards']), images=len(ids), templates=str(destination / 'templates'))


def record(readings_path):
    (qualify, _, _, manifest, _, _), session = session_context()
    require(not (qualify / 'frozen_readings.json').exists(), 'Readings already frozen')
    value = load(readings_path)
    boards = {b['file']: b for b in manifest['boards']}
    require(value.get('board_file') in boards, 'Unknown review board')
    board = boards[value['board_file']]
    require(value.get('board_sha256') == board['sha256'], 'Reading refers to another board')
    require(value.get('all_outputs_actually_viewed') is True and value.get('owner_mapping_opened') is False,
            'Actually inspect the complete opaque board before recording')
    validate_readings(value['readings'], board['ids'])
    write_new(qualify / 'review/boards' / (Path(board['file']).stem + '.json'), dict(
        value, recorded_unix=time.time(), session_sha256=sha(qualify / 'review/session.json'),
        reviewer=session['reviewer'], reviewer_kind=session['reviewer_kind']))
    return dict(recorded=board['file'], images=len(board['ids']))


def freeze(qualify, manifest, ids, bindings):
    """Use only opaque manifests, images and readings. Never read mapping.json."""
    require(not (qualify / 'frozen_readings.json').exists()
            and not (qualify / 'review/freeze_receipt.json').exists(), 'Preserve the first frozen review')
    require(board_inventory(qualify, manifest) == ids, 'Review inventory changed')
    readings = {}
    receipts = {}
    for board in manifest['boards']:
        path = qualify / 'review/boards' / (Path(board['file']).stem + '.json')
        value = load(path)
        require(value['session_sha256'] == sha(qualify / 'review/session.json'), 'Board review session changed')
        require(value['board_file'] == board['file'] and value['board_sha256'] == board['sha256'],
                'Board review differs')
        require(value['all_outputs_actually_viewed'] is True and value['owner_mapping_opened'] is False,
                'Complete actual opaque review required')
        validate_readings(value['readings'], board['ids'])
        require(not set(readings).intersection(value['readings']), 'Duplicate frozen reading')
        readings.update(value['readings'])
        receipts[path.name] = sha(path)
    require(len(receipts) == len(list((qualify / 'review/boards').glob('*.json'))),
            'Unexpected board review receipts')
    validate_readings(readings, ids)
    pixels = {key: sha(qualify / 'pages' / (key + '.png')) for key in ids}
    value = dict(
        schema='joint-writer-frozen-physical-readings-v1', frozen_unix=time.time(), bindings=bindings,
        boards_sha256=bindings['boards_sha256'], mapping_sha256=bindings['mapping_sha256'],
        session_sha256=sha(qualify / 'review/session.json'), board_receipt_sha256=receipts,
        image_sha256=pixels, readings=readings, all_boards_actually_viewed=True,
        owner_mapping_opened=False, workflow_freezes_before_mapping_read=True,
        review_claim_requires_actual_visual_inspection=True,
    )
    write_new(qualify / 'frozen_readings.json', value)
    write_new(qualify / 'review/freeze_receipt.json', dict(
        frozen_readings_sha256=sha(qualify / 'frozen_readings.json'), bindings=bindings,
        session_sha256=sha(qualify / 'review/session.json'), sealed_unix=time.time()))
    return dict(frozen_images=len(ids), frozen_sha256=sha(qualify / 'frozen_readings.json'))


def aggregate(jobs, mapping, readings, registration_sha, threshold):
    """Recount all registered positions and groups; a missing group never passes."""
    expected = {}
    counts = Counter()
    for job in jobs:
        group = (job['split'], job['kind'], job['layout'])
        require(group in GROUPS, 'Unexpected qualification group')
        positions = job['positions']
        require(positions and len(set(positions)) == len(positions), 'Duplicate or empty positions')
        for role in ['candidate', 'parent']:
            for position in positions:
                key = hashlib.sha256(f'{registration_sha}/{role}/{job["id"]}/{position}'.encode()).hexdigest()[:20]
                require(key not in expected, 'Duplicate physical case/position')
                expected[key] = dict(role=role, case=job['id'], split=job['split'], kind=job['kind'],
                                     layout=job['layout'], position=position)
                counts[(role, *group)] += 1
    require({key[1:] for key in counts if key[0] == 'candidate'} == GROUPS, 'Missing registered physical group')
    require(set(mapping) == set(expected), 'Owner mapping omits or adds a registered tile')
    validate_readings(readings, expected)
    groups = defaultdict(list)
    for key, row in expected.items():
        require(all(mapping[key].get(k) == v for k, v in row.items()), 'Owner row differs from registered case')
        v = readings[key]
        groups[(row['role'], row['split'], row['kind'], row['layout'])].append(
            bool(v['text'].strip()) and not v['uncertain'] and not v['blank'])
    required = Fraction(str(threshold))
    require(required == Fraction(9, 10), 'Do not relax the registered 90% gate')
    result = [dict(zip(['role', 'split', 'kind', 'layout'], key), count=len(values),
                   readable_nonblank=sum(values), fraction=sum(values) / len(values),
                   passed=sum(values) * required.denominator >= len(values) * required.numerator)
              for key, values in sorted(groups.items())]
    require(all(row['count'] == counts[tuple(row[k] for k in ['role', 'split', 'kind', 'layout'])]
                for row in result), 'Wrong physical group denominator')
    return result, all(row['passed'] for row in result if row['role'] == 'candidate')


def verify_pixels(qualify, report, manifest, mapping, frozen, guard=lambda: None):
    import numpy as np
    from PIL import Image

    for key, row in mapping.items():
        require(sha(qualify / 'pages' / (key + '.png')) == row['image_sha256'] == frozen['image_sha256'][key],
                'Physical page changed after review')
    for row in report['records']:
        guard()
        path = qualify / 'cases' / row['job']['id'] / 'pixels.npz'
        require(sha(path) == row['pixels_sha256'], 'Complete conditional pixels changed')
        with np.load(path, allow_pickle=False) as arrays:
            for role in ['candidate', 'parent']:
                pixels = arrays[role + '_ink']
                require(pixels.shape == (row['target_glyphs'], 1, 32, 32), 'Incomplete conditional pixels')
                require(pixels.dtype == np.uint8 and np.isin(pixels, [0, 1]).all(), 'Binary physical pixels required')
                for position in row['job']['positions']:
                    key = hashlib.sha256(f'{frozen["bindings"]["registration_sha256"]}/{role}/{row["job"]["id"]}/{position}'.encode()).hexdigest()[:20]
                    require(mapping[key]['pixels_sha256'] == row['pixels_sha256'], 'Page source differs')
                    with Image.open(qualify / 'pages' / (key + '.png')) as image:
                        require(image.mode == 'L' and image.size == (32, 32), 'Original physical page required')
                        require(np.array_equal(np.asarray(image), 255 - 255 * pixels[position, 0]),
                                'Displayed tile differs from generated pixels')
    for board in manifest['boards']:
        guard()
        with Image.open(qualify / board['file']) as image:
            require(image.mode == 'L' and image.size == (1120, 848), 'Original whole board required')
            for index, key in enumerate(board['ids']):
                x, y = (index % 8) * 140 + 36, (index // 8) * 106 + 4
                with Image.open(qualify / 'pages' / (key + '.png')) as tile:
                    scaled = np.repeat(np.repeat(np.asarray(tile), 2, axis=0), 2, axis=1)
                    require(np.array_equal(np.asarray(image.crop((x, y, x + 64, y + 64))), scaled),
                            'Board does not contain the reviewed physical page')


def read_frozen_bundle(qualify, manifest, ids, bindings):
    """Authenticate the seal and every board receipt before unmasking."""
    frozen_path = qualify / 'frozen_readings.json'
    seal = load(qualify / 'review/freeze_receipt.json')
    require(seal['frozen_readings_sha256'] == sha(frozen_path) and seal['bindings'] == bindings,
            'Frozen readings changed after sealing')
    frozen = load(frozen_path)
    require(frozen['bindings'] == bindings and frozen['workflow_freezes_before_mapping_read'] is True,
            'A source-bound frozen review must precede unmasking')
    require(frozen['session_sha256'] == sha(qualify / 'review/session.json'), 'Frozen review session changed')
    require(seal['session_sha256'] == frozen['session_sha256'], 'Freeze seal belongs to another session')
    require(frozen['all_boards_actually_viewed'] is True and frozen['owner_mapping_opened'] is False,
            'Frozen opaque review declaration missing')
    for name, digest in frozen['board_receipt_sha256'].items():
        require(Path(name).name == name and sha(qualify / 'review/boards' / name) == digest,
                'Frozen board receipt changed')
    require(set(frozen['board_receipt_sha256']) == {Path(b['file']).stem + '.json' for b in manifest['boards']},
            'All reviewed boards must be frozen')
    validate_readings(frozen['readings'], ids)
    require(set(frozen['image_sha256']) == set(ids), 'All reviewed page hashes required')
    return frozen


def score():
    began = time.perf_counter()
    (qualify, registration, report, manifest, ids, bindings), session = session_context()
    require(not (qualify / 'gate.json').exists(), 'Preserve the first physical verdict')
    frozen_path = qualify / 'frozen_readings.json'
    frozen = read_frozen_bundle(qualify, manifest, ids, bindings)
    # This is the first owner-map read in this workflow. The frozen file already
    # exists durably and must not be replaced by reference-informed transcriptions.
    require(sha(qualify / 'mapping.json') == bindings['mapping_sha256'], 'Owner mapping changed')
    mapping = load(qualify / 'mapping.json')
    require(mapping['registration_sha256'] == bindings['registration_sha256'], 'Foreign owner mapping')
    groups, passed = aggregate(registration['qualification_jobs'], mapping['rows'], frozen['readings'],
                               bindings['registration_sha256'], registration['physical_gate_readable_nonblank_fraction'])
    def guard():
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2, 'Retain 24GiB available host memory')
        require(time.perf_counter() - began <= 900, 'CPU physical review audit time ceiling reached')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2, 'CPU review RSS ceiling reached')
    guard()
    verify_pixels(qualify, report, manifest, mapping['rows'], frozen, guard)
    guard()
    value = dict(
        schema='joint-writer-physical-gate-v2', passed=passed, groups=groups, bindings=bindings,
        frozen_readings_sha256=sha(frozen_path), session_sha256=sha(qualify / 'review/session.json'),
        reviewer=session['reviewer'], reviewer_kind=session['reviewer_kind'],
        prior_development_familiarity=True, independent_human_adjudication=False,
        complete_registered_positions_recounted=True, original_page_and_board_pixels_verified=True,
        readings_frozen_before_workflow_owner_map_read=True, physical_only=True,
        canonical_match_not_required=True, complete_language_unmeasured=True, goal_achieved=False,
        new_model_calls=0, new_optimizer_updates=0, elapsed_seconds=time.perf_counter() - began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    )
    write_new(qualify / 'gate.json', value)
    return dict(passed=passed, groups=len(groups), physical_only=True, complete_language_unmeasured=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'record', 'freeze', 'score'])
    parser.add_argument('--reviewer')
    parser.add_argument('--reviewer-kind', choices=['model', 'human'])
    parser.add_argument('--readings', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES empty for CPU review')
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    if args.action == 'prepare':
        require(args.reviewer and args.reviewer_kind, 'Name the actual reviewer and kind')
        result = prepare(args.reviewer, args.reviewer_kind)
    elif args.action == 'record':
        require(args.readings is not None, 'Provide complete board readings')
        result = record(args.readings)
    elif args.action == 'freeze':
        values, _ = session_context()
        result = freeze(values[0], values[3], values[4], values[5])
    else:
        result = score()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
