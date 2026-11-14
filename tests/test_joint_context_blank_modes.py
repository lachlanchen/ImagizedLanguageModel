"""Probability math, fixed strata and actual saved-output admission checks."""
import json
import os
import subprocess
import sys

import pytest

from scripts import analyze_joint_context_blank_modes as probe


def test_import_and_missing_binding_reject_before_neural_runtime(tmp_path):
    program = """
import sys
from pathlib import Path
from scripts import analyze_joint_context_blank_modes as p
p.PUBLIC=Path(sys.argv[1])/'absent.json';p.DEST=Path(sys.argv[1])/'probe'
try:p.validate()
except ValueError as error:assert 'Register probe' in str(error)
else:raise AssertionError('Missing binding admitted')
assert 'torch' not in sys.modules and not p.DEST.exists()
"""
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=True, cwd=probe.ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_original_failure_sizes_and_each_prospective_boundary():
    result = probe.load(probe.RESULT)
    rows = {'/'.join(row[k] for k in ['split', 'kind', 'layout']): row
            for row in result['complete_supplied_history_pixel_diagnostics']}
    groups = {k: dict(false_blank=n, source_preferred_false_blank=m) for k, (n, m) in probe.BOOK_RULES.items()}
    assert all(rows[k]['candidate_false_blank'] == v['false_blank'] for k, v in groups.items())
    assert probe.decision(groups)['search_comparison_eligible']
    for key in groups:
        failed = {k: dict(v) for k, v in groups.items()};failed[key]['source_preferred_false_blank'] -= 1
        assert not probe.decision(failed)['search_comparison_eligible']
    groups['development/book/fixed_cells']['false_blank'] -= 1
    with pytest.raises(ValueError, match='cohort changed'):
        probe.decision(groups)


def test_prior_attempt_never_becomes_an_automatic_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'DEST', tmp_path)
    probe.ensure_first_execution()
    (tmp_path / 'status.json').write_text(json.dumps(dict(pid=None, phase='failed')))
    with pytest.raises(ValueError, match='No automatic probe retry'):
        probe.ensure_first_execution()


def test_scores_match_actual_density_image_nll_and_its_saved_greedy_path():
    """Real image-head computation; no base body, Chinese score or language claim."""
    import numpy as np
    import torch
    from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs
    torch.set_num_threads(2);torch.manual_seed(20261114)
    model = GlyphJointDensity(torch.zeros(4096), torch.ones(4096)).eval().requires_grad_(False)
    states = torch.randn(2, 4096); ink = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.uint8)
    with torch.no_grad():
        emitted = pack_glyphs(model.write(states)); source = pack_glyphs(ink)
        out = probe.score_density(model, states, source, emitted)
        expected = model.image_nll(states, ink).numpy()
        assert np.allclose(-out['source_logp'] / 128, expected, atol=1e-6, rtol=0)
        assert (out['saved_argmax_disagreements'] == 0).all()
        assert (out['saved_greedy_logp_deficit'] == 0).all()
        zero = torch.zeros_like(source)
        out = probe.score_density(model, states, zero, zero)
        assert np.array_equal(out['source_logp'], out['emitted_logp'])
        assert np.allclose(out['blank_logp'], out['emitted_logp'], atol=1e-9, rtol=0)
