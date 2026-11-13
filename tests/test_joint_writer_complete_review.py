"""Complete-cohort review invariants; fixtures are not trained language results."""
import copy
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from scripts import evaluate_joint_writer_language as evaluation
from scripts import review_joint_writer_language as review


def registration():
    return json.loads(evaluation.review.PUBLIC.read_text())


def test_complete_real_plan_requires_every_output_before_review():
    r = registration()
    r['arms'] = evaluation.plan(r)
    r['snapshots'] = {str(step): dict(sha256=str(step).zfill(64)) for step in r['snapshot_steps']}
    arms = []
    for arm in r['arms']:
        arms.append(dict(arm=arm['arm'], step=arm['step'], records=[dict(input=row,
            maximum_glyphs=arm['maximum_glyphs'], checkpoint_sha256=r['snapshots'][str(arm['step'])]['sha256'],
            actual_emitted_pixel_feedback=True, reference_inputs=False, output_repaired=False) for row in arm['inputs']]))
    report = dict(arms=arms)
    assert len(review.records(r, report)) == 708
    report['arms'][-1]['records'].pop()
    with pytest.raises(ValueError, match='Missing or duplicate'):
        review.records(r, report)


def core_fixture(r):
    cores = {}
    for layout in ['fixed_cells', 'proportional']:
        counts = [dict(step=41298, kind='qa', condition='normal', cases=32,
                       complete_exact=12, readable=24, paired_both_correct=4)]
        counts += [dict(step=41298, kind='qa', condition=condition, cases=32, complete_exact=0)
                   for condition in ['question_blank', 'passage_blank']]
        rows = [dict(step=41298, condition='normal', case=key, complete_exact=True)
                for key in r['exact_retention'][layout]]
        cores[layout] = dict(results=counts, records=rows)
    books = [dict(arm='candidate_long640', step=41298, condition='normal', cases=12, coherent_complete=4)]
    return cores, books


def evaluation_metadata(r):
    # Match the actual post-gate evaluation schema, which has no snapshot_steps.
    return dict(pilot=r['pilot'], exact_retention=r['exact_retention'],
                snapshots={str(step): {} for step in r['snapshot_steps']})


def test_full_four_QA_erasure_rule_cannot_be_replaced_by_proportional_only_gate():
    r = registration()
    cores, books = core_fixture(r)
    assert review.finalize(evaluation_metadata(r), cores, books)['pilot_passed']
    cores['fixed_cells']['results'][1]['complete_exact'] = 3
    value = review.finalize(evaluation_metadata(r), cores, books)
    assert value['legacy_proportional_route_passed'] and not value['pilot_passed']
    assert len(value['four_QA_erasure_audit']['all_four_QA_erasure_arms']) == 4
    cores['fixed_cells']['results'].pop()
    with pytest.raises(ValueError, match='Exactly one aggregate'):
        review.finalize(evaluation_metadata(r), cores, books)


def test_required_retained_answer_cannot_disappear_from_an_apparent_pass():
    r = registration()
    cores, books = core_fixture(r)
    cores['proportional']['records'][0]['complete_exact'] = False
    assert not review.finalize(evaluation_metadata(r), cores, books)['pilot_passed']
    cores['fixed_cells']['records'].pop()
    with pytest.raises(ValueError, match='Missing required retained'):
        review.finalize(evaluation_metadata(r), cores, books)


def test_reading_requires_actual_complete_board_and_normal_book_prefix():
    board = dict(file='board_001.png', sha256='a', ids=['opaque'], prefixes={'opaque': 'b'})
    value = dict(board_file='board_001.png', board_sha256='a', all_outputs_actually_viewed=True,
        visible_prefixes_viewed=['opaque'], owner_mapping_opened=False, reference_answers_opened=False,
        readings={'opaque': dict(text='', uncertain=True, coherent_continuation=False)})
    review.validate_readings(value, board)
    altered = copy.deepcopy(value)
    altered['visible_prefixes_viewed'] = []
    with pytest.raises(ValueError, match='View every associated'):
        review.validate_readings(altered, board)
    value['reference_answers_opened'] = True
    with pytest.raises(ValueError, match='before unmasking'):
        review.validate_readings(value, board)


def test_whole_variable_height_answer_pages_must_survive_board_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DIRECTORY', tmp_path)
    (tmp_path / 'pages').mkdir()
    sizes = [(1024, 64), (32, 32)]
    canvas = Image.new('L', (1056, 64 + 38 + 40 + 38), 245)
    images = {}
    top = 0
    for index, size in enumerate(sizes):
        key = str(index)
        image = Image.new('L', size, 255)
        image.putpixel((size[0] - 1, size[1] - 1), 0)
        image.save(tmp_path / 'pages' / (key + '.png'))
        images[key] = review.sha(tmp_path / 'pages' / (key + '.png'))
        canvas.paste(image, (8, top + 28))
        top += max(40, image.height) + 38
    canvas.save(tmp_path / 'board_001.png')
    board = dict(file='board_001.png', sha256=review.sha(tmp_path / 'board_001.png'),
                 ids=['0', '1'], images=images, prefixes={})
    review.verify_board(board)
    canvas.putpixel((8, 28), 0)
    canvas.save(tmp_path / 'board_001.png')
    board['sha256'] = review.sha(tmp_path / 'board_001.png')
    with pytest.raises(ValueError, match='Board crop differs'):
        review.verify_board(board)


def test_prepare_refuses_before_generation_completes_without_creating_review(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DEST', tmp_path)
    monkeypatch.setattr(review, 'DIRECTORY', tmp_path / 'review')
    with pytest.raises(ValueError, match='Complete all 708'):
        review.prepare('fixture-reviewer', 'model')
    assert not (tmp_path / 'review').exists()


def test_full_inventory_freezes_before_owner_map_and_seal_rejects_changed_readings(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DEST', tmp_path / 'language')
    monkeypatch.setattr(review, 'DIRECTORY', tmp_path / 'language/review')
    monkeypatch.setattr(evaluation, 'PUBLIC', tmp_path / 'registration.json')
    directory = review.DIRECTORY
    (directory / 'pages').mkdir(parents=True)
    review.write_new(evaluation.PUBLIC, dict(unit_fixture=True))
    review.write_new(review.DEST / 'report.json', dict(unit_fixture=True))
    white = tmp_path / 'white.png'
    Image.new('L', (32, 32), 255).save(white)
    page_sha = review.sha(white)
    canvas = Image.new('L', (1056, 4 * 78), 245)
    for index in range(4):
        canvas.paste(255, (8, index * 78 + 28, 40, index * 78 + 60))
    board_image = tmp_path / 'board.png'
    canvas.save(board_image)
    board_sha = review.sha(board_image)
    boards, receipts = [], []
    for index in range(177):
        ids = [f'{j:020x}' for j in range(index * 4, index * 4 + 4)]
        prefixes = {}
        for j, key in enumerate(ids):
            os.link(white, directory / 'pages' / (key + '.png'))
            if index * 4 + j < 84:
                os.link(white, directory / 'pages' / (key + '_prefix.png'))
                prefixes[key] = page_sha
        name = f'board_{index + 1:03}.png'
        os.link(board_image, directory / name)
        board = dict(file=name, sha256=board_sha, ids=ids, images={key: page_sha for key in ids}, prefixes=prefixes)
        boards.append(board)
        receipts.append(dict(board_file=name, board_sha256=board_sha, all_outputs_actually_viewed=True,
            owner_mapping_opened=False, reference_answers_opened=False, visible_prefixes_viewed=list(prefixes),
            readings={key: dict(text='', uncertain=False, coherent_continuation=False if key in prefixes else None)
                      for key in ids}))
    review.write_new(directory / 'boards.json', dict(boards=boards, responses=708))
    review.write_new(directory / 'session.json', dict(reviewer='unit-fixture', reviewer_kind='model',
        registration_sha256=review.sha(evaluation.PUBLIC), generation_report_sha256=review.sha(review.DEST / 'report.json'),
        boards_sha256=review.sha(directory / 'boards.json'), mapping_sha256='a' * 64,
        source_sha256=review.sha(Path(review.__file__))))
    for board, value in zip(boards, receipts):
        review.write_new(directory / 'board_receipts' / (Path(board['file']).stem + '.json'),
                         dict(value, session_sha256=review.sha(directory / 'session.json')))
    # Any attempted owner JSON parse before freeze would fail.
    (directory / 'mapping.json').write_text('unopened owner-map fixture')
    assert review.freeze()['frozen_responses'] == 708
    assert len(review.frozen_readings()['readings']) == 708
    value = review.load(directory / 'frozen_readings.json')
    value['readings'][boards[0]['ids'][0]]['text'] = 'changed after freeze'
    (directory / 'frozen_readings.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='seal changed'):
        review.frozen_readings()
