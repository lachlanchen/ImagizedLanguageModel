"""New full-comparison accounting and both physical-regression safeguards."""
import json

import numpy as np
from PIL import Image
import pytest

from ilm.visual_lm import white_target_training as training
from scripts.review_white_target_physical import (
    opaque_blank_flags, validate_blank_readings, whitespace_counts, physical_count_conditions)

RECIPE = dict(ranking_strength=1., white_strength=1.)


def event():
    return dict(image_loss=.3, ranking_loss=.2, white_ranking_loss=.4, stop_loss=.1, loss=.91,
        state_gradient_norm=1., gradient_norm_before_clip=2., learning_rate=.0001,
        elapsed_seconds=1., target_glyphs=3, nonempty_targets=2, white_targets=1)


@pytest.mark.parametrize('change', [dict(loss=.51), dict(white_ranking_loss=-.1), dict(image_loss=float('nan')),
    dict(nonempty_targets=3), dict(white_targets=0), dict(white_targets=True), dict(white_targets=-1)])
def test_source_partition_and_new_loss_tampering_rejects(change):
    training.check_training_event(event(), RECIPE, 2)
    with pytest.raises(ValueError): training.check_training_event(dict(event(), **change), RECIPE, 2)


def test_disjoint_objectives_cannot_penalize_the_wrong_source_type():
    white = dict(event(), nonempty_targets=0, white_targets=3, ranking_loss=0., loss=.71)
    training.check_training_event(white, RECIPE, 0)
    with pytest.raises(ValueError, match='True whitespace'):
        training.check_training_event(dict(white, ranking_loss=.2, loss=.91), RECIPE, 0)
    ink = dict(event(), nonempty_targets=3, white_targets=0, white_ranking_loss=0., loss=.51)
    training.check_training_event(ink, RECIPE, 3)
    with pytest.raises(ValueError, match='Nonempty targets'):
        training.check_training_event(dict(ink, white_ranking_loss=.4, loss=.91), RECIPE, 3)


def test_original_opaque_pixel_safeguard_is_preserved(tmp_path):
    blank, ink = 'a' * 20, 'b' * 20; pixels = np.full((32, 32), 255, np.uint8)
    Image.fromarray(pixels).save(tmp_path / (blank + '.png')); pixels[11, 18] = 0
    Image.fromarray(pixels).save(tmp_path / (ink + '.png'))
    flags = opaque_blank_flags(tmp_path, [blank, ink]); assert flags == {blank: True, ink: False}
    readings = {blank: dict(text='', uncertain=False, blank=True), ink: dict(text='', uncertain=True, blank=False)}
    validate_blank_readings(readings, flags)
    with pytest.raises(ValueError, match='contradicts opaque pixels'):
        validate_blank_readings(dict(readings, **{blank: dict(text='中', uncertain=False, blank=False)}), flags)


def test_complete_counts_keep_false_ink_and_false_blanks_separate():
    gold = np.zeros((4, 1, 32, 32), np.uint8); gold[2:, 0, 4, 5] = 1
    candidate = gold.copy(); candidate[0, 0, 7, 8] = 1; candidate[2] = 0
    assert whitespace_counts(candidate, gold) == dict(positions=4, source_whitespace=2, false_ink=1, false_blanks=1)


@pytest.mark.parametrize('ink,blank,expected', [(11, 37, (True, True)), (12, 37, (False, True)),
                                              (11, 38, (True, False)), (25, 37, (False, True))])
def test_each_full_cohort_boundary_is_binding(ink, blank, expected):
    r = dict(source_whitespace_positions=252, maximum_false_ink_on_source_whitespace=11, maximum_false_blanks=37)
    row = dict(positions=12597, source_whitespace=252, false_ink=ink, false_blanks=blank)
    result = physical_count_conditions([row], r)
    assert (result['whitespace_passed'], result['false_blanks_passed']) == expected
    with pytest.raises(ValueError, match='thresholds'):
        physical_count_conditions([row], dict(r, maximum_false_blanks=38))
    with pytest.raises(ValueError, match='cohort'):
        physical_count_conditions([dict(row, positions=12596)], r)


def test_recovery_costs_and_unterminated_owner_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'RUN', tmp_path); path = tmp_path / 'controller_journal.jsonl'
    rows = [dict(event='started', action='train', child_pid=99999999),
            dict(event='ended', action='train', child_pid=99999999, elapsed_seconds=12.)]
    path.write_text('\n'.join(json.dumps(x) for x in rows)); assert training.previous_seconds('train') == 12.
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match='Reconcile prior unterminated'): training.previous_seconds('train')


def test_old_ranked_checkpoint_cannot_be_used_as_white_target():
    with pytest.raises(ValueError, match='provenance differs'):
        training.check_checkpoint(dict(schema='global-glyph-ranking-checkpoint-v1'), {})


def test_checkpoint_with_changed_white_strength_rejects_before_tensor_restore(tmp_path, monkeypatch):
    path = tmp_path / 'registration.json'; path.write_text('{}'); monkeypatch.setattr(training, 'PUBLIC', path)
    r = dict(source_sha256={}, runtime={}, objective='physical ranking', ranking_strength=1., white_strength=1.)
    value = dict(schema='white-target-ranking-checkpoint-v1', registration_sha256=training.sha(path),
        source_sha256={}, runtime={}, objective=r['objective'], ranking_strength=1., white_strength=0.)
    with pytest.raises(ValueError, match='objective/normalization'):
        training.check_checkpoint(value, r)
