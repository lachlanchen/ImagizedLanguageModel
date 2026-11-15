#!/usr/bin/env python3
"""Opaque physical review with pixel-derived blank flags and a whitespace gate.

Only the score action opens ownership and source pixels, after the first frozen
review. A blank/nonblank flag is a pixel fact, not a character transcription.
"""
import time
import resource
from pathlib import Path

import numpy as np
from PIL import Image

from ilm.visual_lm.white_target_training import ROOT, RUN, PUBLIC, load, sha, require
from scripts import review_joint_writer_physical as review

_original_context = review.context
_original_validate = review.validate_readings


def opaque_blank_flags(pages, ids):
    """Read only binary opaque tiles; no owner map, references or text lookup."""
    result = {}
    for key in ids:
        require(len(key) == 20 and all(c in '0123456789abcdef' for c in key), 'Invalid opaque image ID')
        with Image.open(pages / (key + '.png')) as image:
            pixels = np.asarray(image)
            require(image.mode == 'L' and image.size == (32, 32)
                    and np.isin(pixels, [0, 255]).all(), 'Original binary opaque glyph required')
            result[key] = bool((pixels == 255).all())
    return result


def validate_blank_readings(readings, flags):
    _original_validate(readings, flags)
    for key, blank in flags.items():
        require(readings[key]['blank'] is blank, 'Blank flag contradicts opaque pixels: ' + key)
        require(not blank or (readings[key]['text'] == '' and readings[key]['uncertain'] is False),
                'An empty image must retain an empty certain reading: ' + key)


def checked_readings(readings, ids):
    validate_blank_readings(readings, opaque_blank_flags(RUN / 'qualify/pages', ids))


def context():
    values = _original_context()
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes(), 'Exact registration required')
    bindings = dict(values[-1], white_target_review_source_sha256=sha(Path(__file__)))
    return (*values[:-1], bindings)


def prepare(reviewer, kind):
    qualify, _, _, manifest, ids, bindings = context()
    require(reviewer.strip() and kind in {'model', 'human'}, 'Identify the actual reviewer')
    destination = qualify / 'review'; destination.mkdir(exist_ok=False)
    flags = opaque_blank_flags(qualify / 'pages', ids)
    review.write_new(destination / 'session.json', dict(
        schema='white-target-opaque-review-session-v1', created_unix=time.time(), bindings=bindings,
        reviewer=reviewer, reviewer_kind=kind, prior_development_familiarity=True,
        independent_human_adjudication=False, reviewer_declares_mapping_unopened=True,
        blank_flags_from_original_opaque_pixels=True))
    for board in manifest['boards']:
        review.write_new(destination / 'templates' / (Path(board['file']).stem + '.json'), dict(
            board_file=board['file'], board_sha256=board['sha256'], all_outputs_actually_viewed=False,
            owner_mapping_opened=False, readings={key: dict(text='' if flags[key] else None,
                uncertain=False if flags[key] else None, blank=flags[key]) for key in board['ids']}))
    return dict(boards=len(manifest['boards']), images=len(ids), templates=str(destination / 'templates'),
                source_or_owner_mapping_opened=False)


def whitespace_counts(candidate, gold):
    require(candidate.shape == gold.shape and candidate.ndim == 4 and candidate.shape[1:] == (1, 32, 32),
            'Complete glyph arrays required')
    require(candidate.dtype == gold.dtype == np.uint8 and np.isin(candidate, [0, 1]).all()
            and np.isin(gold, [0, 1]).all(), 'Binary physical pixels required')
    source_empty = ~gold.reshape(len(gold), -1).any(1)
    output_empty = ~candidate.reshape(len(candidate), -1).any(1)
    return dict(positions=len(gold), source_whitespace=int(source_empty.sum()),
                false_ink=int((source_empty & ~output_empty).sum()),
                false_blanks=int((~source_empty & output_empty).sum()))


def physical_count_conditions(counts, registration):
    """Keep the complete whitespace and nonempty safeguards distinct."""
    for row in counts:
        require(all(type(row[k]) is int and row[k] >= 0 for k in
                    ['positions', 'source_whitespace', 'false_ink', 'false_blanks']), 'Integer physical counts required')
        require(row['false_ink'] <= row['source_whitespace'] <= row['positions']
                and row['false_blanks'] <= row['positions'] - row['source_whitespace'], 'Invalid source/output partition')
    positions = sum(r['positions'] for r in counts); whitespace = sum(r['source_whitespace'] for r in counts)
    require(positions == 12597 and whitespace == registration['source_whitespace_positions'] == 252, 'Complete original physical cohort required')
    require(registration['maximum_false_ink_on_source_whitespace'] == 11
            and registration['maximum_false_blanks'] == 37, 'Preserve prospective physical thresholds')
    false_ink = sum(r['false_ink'] for r in counts); false_blanks = sum(r['false_blanks'] for r in counts)
    return dict(source_whitespace=whitespace, false_ink=false_ink, false_blanks=false_blanks,
        whitespace_passed=false_ink <= 11, false_blanks_passed=false_blanks <= 37)


def score():
    began = time.perf_counter()
    (qualify, registration, report, manifest, ids, bindings), session = review.session_context()
    require(not (qualify / 'gate.json').exists(), 'Preserve the first physical verdict')
    frozen = review.read_frozen_bundle(qualify, manifest, ids, bindings)
    checked_readings(frozen['readings'], ids)
    # No preceding action opens this map or any source image array.
    require(sha(qualify / 'mapping.json') == bindings['mapping_sha256'], 'Owner mapping changed')
    mapping = load(qualify / 'mapping.json')
    require(mapping['registration_sha256'] == bindings['registration_sha256'], 'Foreign mapping')
    groups, readable_passed = review.aggregate(registration['qualification_jobs'], mapping['rows'], frozen['readings'],
        bindings['registration_sha256'], registration['physical_gate_readable_nonblank_fraction'])
    def guard():
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2, 'Retain24GiB host RAM')
        require(time.perf_counter() - began <= 900 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2,
                'CPU physical review resource ceiling reached')
    guard(); review.verify_pixels(qualify, report, manifest, mapping['rows'], frozen, guard)
    counts = []; credited_empty = []
    for row in report['records']:
        guard(); path = qualify / 'cases' / row['job']['id'] / 'pixels.npz'
        require(sha(path) == row['pixels_sha256'], 'Complete pixels changed')
        with np.load(path, allow_pickle=False) as arrays:
            counts.append(dict(case=row['job']['id'], split=row['job']['split'], kind=row['job']['kind'],
                layout=row['job']['layout'], **whitespace_counts(arrays['candidate_ink'], arrays['gold_ink'])))
    flags = opaque_blank_flags(qualify / 'pages', ids)
    for key, row in mapping['rows'].items():
        value = frozen['readings'][key]
        if row['role'] == 'candidate' and value['text'].strip() and not value['uncertain'] and not value['blank'] and flags[key]:
            credited_empty.append(key)
    conditions = physical_count_conditions(counts, registration)
    whitespace_passed = conditions['whitespace_passed']; ink_passed = not credited_empty
    passed = readable_passed and whitespace_passed and ink_passed and conditions['false_blanks_passed']
    guard()
    review.write_new(qualify / 'gate.json', dict(
        schema='white-target-physical-gate-v1', passed=passed, groups=groups, bindings=bindings,
        readable_groups_passed=readable_passed, **conditions,
        credited_candidate_ink_passed=ink_passed, credited_empty_candidate_ids=credited_empty,
        maximum_false_ink=11, maximum_false_blanks=37, complete_case_counts=counts,
        frozen_readings_sha256=sha(qualify / 'frozen_readings.json'), session_sha256=sha(qualify / 'review/session.json'),
        reviewer=session['reviewer'], reviewer_kind=session['reviewer_kind'], prior_development_familiarity=True,
        independent_human_adjudication=False, complete_registered_positions_recounted=True,
        original_page_and_board_pixels_verified=True, readings_frozen_before_workflow_owner_map_read=True,
        blank_flags_verified_against_opaque_pixels=True, physical_only=True, canonical_match_not_required=True,
        complete_language_unmeasured=True, goal_achieved=False, new_model_calls=0, new_optimizer_updates=0,
        elapsed_seconds=time.perf_counter() - began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
    return dict(passed=passed, readable_groups_passed=readable_passed, whitespace_passed=whitespace_passed,
                false_blanks_passed=conditions['false_blanks_passed'],
                credited_candidate_ink_passed=ink_passed, complete_language_unmeasured=True)


def configure():
    review.RUN, review.PUBLIC = RUN, PUBLIC
    review.context = context
    review.validate_readings = checked_readings
    review.prepare = prepare
    review.score = score

if __name__ == '__main__':
    configure()
    review.main()
