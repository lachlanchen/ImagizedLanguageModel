"""Failure tests for the complete trial's accounting and prospective review."""
import json

import numpy as np
from PIL import Image
import pytest

from ilm.visual_lm import global_ranking_training as training
from scripts.review_global_glyph_ranking import opaque_blank_flags, validate_blank_readings, whitespace_counts


def event():
    return dict(image_loss=.3, ranking_loss=.2, stop_loss=.1, loss=.51,
                state_gradient_norm=1., gradient_norm_before_clip=2., learning_rate=.0001,
                elapsed_seconds=1., target_glyphs=3, nonempty_targets=2)


@pytest.mark.parametrize('change', [dict(loss=.5), dict(ranking_loss=-.1), dict(image_loss=float('nan')),
                                  dict(nonempty_targets=3), dict(nonempty_targets=True)])
def test_source_or_loss_tampering_is_rejected(change):
    good = event(); training.check_training_event(good, dict(ranking_strength=1.), 2)
    with pytest.raises(ValueError):
        training.check_training_event(dict(good, **change), dict(ranking_strength=1.), 2)


def test_whitespace_never_gets_a_ranking_penalty():
    good = dict(event(), nonempty_targets=0, ranking_loss=0., loss=.31)
    training.check_training_event(good, dict(ranking_strength=1.), 0)
    with pytest.raises(ValueError, match='True whitespace'):
        training.check_training_event(dict(good, ranking_loss=.2, loss=.51), dict(ranking_strength=1.), 0)


def test_opaque_pixels_override_an_incorrect_review_flag(tmp_path):
    blank, ink = 'a' * 20, 'b' * 20
    pixels = np.full((32, 32), 255, dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / (blank + '.png'))
    pixels[10, 15] = 0
    Image.fromarray(pixels).save(tmp_path / (ink + '.png'))
    flags = opaque_blank_flags(tmp_path, [blank, ink])
    assert flags == {blank: True, ink: False}
    readings = {blank: dict(text='', uncertain=False, blank=True),
                ink: dict(text='', uncertain=True, blank=False)}
    validate_blank_readings(readings, flags)
    with pytest.raises(ValueError, match='contradicts opaque pixels'):
        validate_blank_readings(dict(readings, **{blank: dict(text='中', uncertain=False, blank=False)}), flags)
    with pytest.raises(ValueError, match='contradicts opaque pixels'):
        validate_blank_readings(dict(readings, **{ink: dict(text='', uncertain=False, blank=True)}), flags)
    Image.fromarray(np.full((32, 32), 254, dtype=np.uint8)).save(tmp_path / (ink + '.png'))
    with pytest.raises(ValueError, match='binary opaque glyph'):
        opaque_blank_flags(tmp_path, [ink])


def test_full_cohort_whitespace_is_not_readability():
    gold = np.zeros((4, 1, 32, 32), dtype=np.uint8)
    gold[2:, 0, 4, 5] = 1
    candidate = gold.copy(); candidate[0, 0, 7, 8] = 1; candidate[2] = 0
    assert whitespace_counts(candidate, gold) == dict(positions=4, source_whitespace=2, false_ink=1, false_blanks=1)
    with pytest.raises(ValueError, match='Binary physical'):
        whitespace_counts(candidate.astype(np.float32), gold)


def test_recovery_costs_and_incomplete_owners_cannot_be_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'RUN', tmp_path)
    path = tmp_path / 'controller_journal.jsonl'
    rows = [dict(event='started', action='train', child_pid=99999999),
            dict(event='ended', action='train', child_pid=99999999, elapsed_seconds=10.)]
    path.write_text('\n'.join(json.dumps(x) for x in rows))
    assert training.previous_seconds('train') == 10.
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match='Reconcile prior unterminated'):
        training.previous_seconds('train')


def test_old_actor_checkpoint_cannot_be_restored_as_ranking():
    with pytest.raises(ValueError, match='provenance differs'):
        training.check_checkpoint(dict(schema='joint-context-checkpoint-v1'), {})
