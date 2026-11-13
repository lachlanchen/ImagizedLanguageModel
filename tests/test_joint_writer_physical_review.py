"""Review ordering, complete groups and original bitmap identity, not language quality."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from scripts import review_joint_writer_physical as review


def reading(text='甲', uncertain=False, blank=False):
    return dict(text=text, uncertain=uncertain, blank=blank)


def group_cases():
    registration = 'f' * 64
    jobs, mapping, readings = [], {}, {}
    for index, (split, kind, layout) in enumerate(sorted(review.GROUPS)):
        job = dict(id=f'case{index}', split=split, kind=kind, layout=layout, positions=list(range(10)))
        jobs.append(job)
        for role in ['candidate', 'parent']:
            for position in job['positions']:
                key = hashlib.sha256(f'{registration}/{role}/{job["id"]}/{position}'.encode()).hexdigest()[:20]
                mapping[key] = dict(role=role, case=job['id'], split=split, kind=kind, layout=layout, position=position)
                readings[key] = reading()
    return registration, jobs, mapping, readings


def test_all_groups_required_and_threshold_uses_exact_counts():
    registration, jobs, mapping, readings = group_cases()
    keys = [key for key, row in mapping.items() if row['role'] == 'candidate' and row['case'] == jobs[0]['id']]
    # One uncertain image is exactly 90%, independent of any canonical answer.
    readings[keys[0]] = reading('', uncertain=True)
    groups, passed = review.aggregate(jobs, mapping, readings, registration, .9)
    assert passed and len(groups) == 14
    readings[keys[1]] = reading('', blank=True)
    assert not review.aggregate(jobs, mapping, readings, registration, .9)[1]
    with pytest.raises(ValueError, match='Missing registered physical group'):
        review.aggregate(jobs[:-1], mapping, readings, registration, .9)
    with pytest.raises(ValueError, match='Do not relax'):
        review.aggregate(jobs, mapping, readings, registration, .89)


def test_parent_failures_do_not_set_candidate_gate_and_owner_tampering_rejected():
    registration, jobs, mapping, readings = group_cases()
    for key, row in mapping.items():
        if row['role'] == 'parent':
            readings[key] = reading('', uncertain=True)
    assert review.aggregate(jobs, mapping, readings, registration, .9)[1]
    mapping[next(iter(mapping))]['position'] = 99
    with pytest.raises(ValueError, match='Owner row differs'):
        review.aggregate(jobs, mapping, readings, registration, .9)


def make_review(tmp_path):
    qualify = tmp_path / 'qualify'
    (qualify / 'review/boards').mkdir(parents=True)
    (qualify / 'pages').mkdir()
    ids = ['0' * 20, '1' * 20]
    Image.new('L', (1120, 848), 255).save(qualify / 'board_001.png')
    for key in ids:
        Image.new('L', (32, 32), 255).save(qualify / 'pages' / (key + '.png'))
    review.write_new(qualify / 'review/session.json', dict(reviewer='unit-test'))
    board = dict(file='board_001.png', sha256=review.sha(qualify / 'board_001.png'), ids=ids)
    manifest = dict(boards=[board], images=2)
    bindings = dict(mapping_sha256='a' * 64, boards_sha256='b' * 64)
    review.write_new(qualify / 'review/boards/board_001.json', dict(
        board_file=board['file'], board_sha256=board['sha256'],
        session_sha256=review.sha(qualify / 'review/session.json'),
        all_outputs_actually_viewed=True, owner_mapping_opened=False,
        readings={ids[0]: reading(), ids[1]: reading('', uncertain=True)}))
    return qualify, manifest, ids, bindings


def test_freeze_and_seal_never_read_owner_map_and_preserve_first_readings(tmp_path):
    qualify, manifest, ids, bindings = make_review(tmp_path)
    # Invalid owner JSON would fail any premature owner-map read.
    (qualify / 'mapping.json').write_text('owner mapping must not be opened')
    result = review.freeze(qualify, manifest, ids, bindings)
    assert result['frozen_images'] == 2
    frozen = review.read_frozen_bundle(qualify, manifest, ids, bindings)
    assert frozen['readings'][ids[1]]['uncertain']
    with pytest.raises(ValueError, match='Preserve the first frozen'):
        review.freeze(qualify, manifest, ids, bindings)
    frozen['readings'][ids[1]] = reading('乙')
    (qualify / 'frozen_readings.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='changed after sealing'):
        review.read_frozen_bundle(qualify, manifest, ids, bindings)


def test_missing_board_changed_board_and_invalid_flags_cannot_be_frozen(tmp_path):
    qualify, manifest, ids, bindings = make_review(tmp_path)
    Image.new('L', (1120, 848), 0).save(qualify / 'board_001.png')
    with pytest.raises(ValueError, match='Reviewed board changed'):
        review.freeze(qualify, manifest, ids, bindings)
    assert not (qualify / 'frozen_readings.json').exists()
    Image.new('L', (1120, 848), 255).save(qualify / 'board_001.png')
    (qualify / 'review/boards/board_001.json').unlink()
    with pytest.raises(FileNotFoundError):
        review.freeze(qualify, manifest, ids, bindings)
    assert not (qualify / 'frozen_readings.json').exists()
    with pytest.raises(ValueError, match='Complete typed readings'):
        review.validate_readings({ids[0]: reading(uncertain='false')}, [ids[0]])
    with pytest.raises(ValueError, match='Blank tile cannot'):
        review.validate_readings({ids[0]: reading('甲', blank=True)}, [ids[0]])
    with pytest.raises(ValueError, match='retain uncertainty'):
        review.validate_readings({ids[0]: reading('')}, [ids[0]])


def test_displayed_pages_and_board_must_match_original_neural_bitmaps(tmp_path):
    qualify = tmp_path
    job = dict(id='case', positions=[0])
    registration = 'f' * 64
    directory = qualify / 'cases/case'
    directory.mkdir(parents=True)
    (qualify / 'pages').mkdir()
    candidate = np.zeros((1, 1, 32, 32), dtype=np.uint8)
    candidate[0, 0, 6:25, 15:18] = 1
    parent = np.transpose(candidate, (0, 1, 3, 2)).copy()
    np.savez_compressed(directory / 'pixels.npz', candidate_ink=candidate, parent_ink=parent)
    digest = review.sha(directory / 'pixels.npz')
    board = Image.new('L', (1120, 848), 255)
    mapping, hashes, ids = {}, {}, []
    for index, (role, pixels) in enumerate([('candidate', candidate), ('parent', parent)]):
        key = hashlib.sha256(f'{registration}/{role}/case/0'.encode()).hexdigest()[:20]
        ids.append(key)
        image = Image.fromarray(255 - 255 * pixels[0, 0])
        image.save(qualify / 'pages' / (key + '.png'))
        hashes[key] = review.sha(qualify / 'pages' / (key + '.png'))
        mapping[key] = dict(image_sha256=hashes[key], pixels_sha256=digest)
        board.paste(image.resize((64, 64), Image.Resampling.NEAREST), (index * 140 + 36, 4))
    board.save(qualify / 'board_001.png')
    manifest = dict(boards=[dict(file='board_001.png', ids=ids)])
    report = dict(records=[dict(job=job, pixels_sha256=digest, target_glyphs=1)])
    frozen = dict(bindings=dict(registration_sha256=registration), image_sha256=hashes)
    review.verify_pixels(qualify, report, manifest, mapping, frozen)
    board.putpixel((36, 4), 0)
    board.save(qualify / 'board_001.png')
    with pytest.raises(ValueError, match='Board does not contain'):
        review.verify_pixels(qualify, report, manifest, mapping, frozen)


def test_atomic_publication_cannot_replace_prior_evidence(tmp_path):
    path = tmp_path / 'frozen.json'
    review.write_new(path, dict(value='original'))
    with pytest.raises(FileExistsError):
        review.write_new(path, dict(value='replacement'))
    assert review.load(path) == dict(value='original')
