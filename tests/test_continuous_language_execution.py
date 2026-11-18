from bisect import bisect_right
import json

import numpy as np
import torch

from scripts.train_continuous_image_language import ROOT, DATA, EVAL, source_arrays
from scripts.evaluate_continuous_image_language import plan
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus


def test_complete_original_packed_source_and_causal_target_accounting():
    torch.set_num_threads(2)
    arrays, offsets = source_arrays()
    assert len(arrays) == 108 and offsets[-1] == 13785
    targets = sum(int((np.diff(a['record_glyph_offsets']) - np.maximum(1, a['prompt_glyphs'])).sum()) for a in arrays)
    assert targets == 2571477
    original = VisualSequenceCorpus(DATA, split='train')
    cases = json.loads((ROOT / 'references/evidence/trainable_visual_context_preflight_registration_20261116.json').read_text())['cases']
    for case in cases:
        i = case['corpus_index']; shard = bisect_right(offsets, i) - 1; local = i - offsets[shard]
        a = arrays[shard]; first, last = a['record_glyph_offsets'][local:local + 2]
        actual = np.unpackbits(a['ink'][first:last], bitorder='big').reshape(-1, 1, 32, 32)
        expected = original[i]
        assert np.array_equal(actual, expected['ink'].numpy())
        assert int(a['prompt_glyphs'][local]) == expected['prompt_glyphs']
        assert len(actual) == case['glyphs']


def test_all_fixed_autonomous_jobs_include_erasure_and_long_book_controls():
    inputs = json.loads((EVAL / 'manifest.json').read_text())['inputs']
    jobs = plan(dict(evaluation_inputs=inputs))
    assert [len(j['inputs']) for j in jobs] == [120, 44, 44, 44, 120, 24]
    assert [j['cap'] for j in jobs] == [128, 128, 128, 128, 128, 640]
    assert sum(len(j['inputs']) for j in jobs) == 396
    assert sum(r['condition'] == 'question_blank' for r in jobs[4]['inputs']) == 32
    assert sum(r['condition'] == 'passage_blank' for r in jobs[4]['inputs']) == 32
    assert sum(r['condition'] == 'prefix_blank' for r in jobs[5]['inputs']) == 12
