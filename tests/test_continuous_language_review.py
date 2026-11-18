"""Review preserves every emitted pixel and freezes readings before scoring."""
import json

import numpy as np
import pytest
from PIL import Image

from scripts import review_continuous_image_language as review


def test_native_page_ink_padding_and_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    n = 33
    tiles = np.zeros((n, 1, 32, 32), dtype=np.uint8)
    tiles[-1, 0, 0, 0] = 1
    page = np.full((64, 1024), 255, dtype=np.uint8)
    page[32, 0] = 0
    raw = tmp_path / 'q.npz'
    png = tmp_path / 'q.png'
    np.savez_compressed(raw, ink=tiles)
    Image.fromarray(page).save(png)
    record = dict(input={'id':'q'}, glyphs=n, cap=n, actual_emitted_pixel_feedback=True,
                  capped=True, stopped=False, output_file=png.name, raw_file=raw.name,
                  output_sha256=review.sha(png), raw_sha256=review.sha(raw))
    assert review.checked_page(tmp_path, record) == (png, True)
    page[32, 32] = 0
    Image.fromarray(page).save(png)
    record['output_sha256'] = review.sha(png)
    with pytest.raises(ValueError, match='pixels or padding'):
        review.checked_page(tmp_path, record)


def test_entire_review_workflow(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    evaluation = tmp_path / 'inputs'
    evaluation.mkdir()
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    monkeypatch.setattr(review, 'RUN', run)
    monkeypatch.setattr(review, 'REVIEW', run / 'review')
    monkeypatch.setattr(review, 'EVAL', evaluation)
    reg_path = tmp_path / 'registration.json'
    monkeypatch.setattr(review, 'REG', reg_path)
    refs, inputs = [], []
    for group in range(1, 17):
        for slot in ('a', 'b'):
            base = f'qa_{group:02d}_{slot}'
            refs.append(dict(id=base, paired_group=group, kind='qa', answer=slot))
            for c in ('normal', 'question_blank', 'passage_blank'):
                inputs.append(dict(id=f'{base}_{c}', condition=c))
    for group in range(1, 13):
        base = f'book_{group:02d}'
        refs.append(dict(id=base, kind='book', answer='x'))
        for c in ('normal', 'prefix_blank'):
            inputs.append(dict(id=f'{base}_{c}', condition=c))
    tile = np.zeros((1, 1, 32, 32), dtype=np.uint8)
    tile[0, 0, 0, 0] = 1
    img = Image.fromarray(255 - 255 * tile[0, 0])
    img.save(evaluation / 'fixture.png')
    for r in inputs:
        r.update(image='fixture.png', glyphs=1, image_sha256=review.sha(evaluation / 'fixture.png'))
    reg = dict(components=[1, 8], evaluation_inputs=inputs)
    review.write_new(reg_path, reg)
    arms = [dict(components=k, snapshots=[{'update':s, 'sha256':f'{k}-{s}'} for s in review.STEPS])
            for k in reg['components']]
    review.write_new(run / 'report.json', dict(arms=arms, registration_sha256=review.sha(reg_path)))
    review.write_new(run / 'audit.json', dict(complete=True, registration_sha256=review.sha(reg_path),
                     report_sha256=review.sha(run / 'report.json')))
    # Keep the references outside the expected location until every reading is frozen.
    ref_file = tmp_path / 'withheld.json'
    review.write_new(ref_file, dict(reference_rows=refs))
    prep = tmp_path / 'preparation.json'
    review.write_new(prep, dict(references_sha256=review.sha(ref_file)))
    monkeypatch.setattr(review, 'PREPARATION_SHA', review.sha(prep))
    streams = []
    for k in reg['components']:
        for job in review.plan(reg):
            step, cap = job['step'], job['cap']
            folder = run / 'evaluation' / f'k{k}_step{step:07d}_cap{cap}'
            folder.mkdir(parents=True)
            for r in job['inputs']:
                png = folder / (r['id'] + '.png')
                raw = png.with_suffix('.npz')
                img.save(png)
                np.savez_compressed(raw, ink=tile)
                rec = dict(components=k, step=step, cap=cap, input=r,
                           snapshot_sha256=f'{k}-{step}', glyphs=1, stopped=True, capped=False,
                           actual_emitted_pixel_feedback=True, raw_file=str(raw.relative_to(tmp_path)),
                           raw_sha256=review.sha(raw), output_file=str(png.relative_to(tmp_path)),
                           output_sha256=review.sha(png))
                streams.append(rec)
                review.write_new(png.with_suffix('.json'), rec)
    review.write_new(run / 'evaluation/report.json', dict(streams=streams, registration_sha256=review.sha(reg_path)))
    review.prepare()
    packet = review.REVIEW
    template = review.load(packet / 'boards/readings_template.json')
    mapping = review.load(packet / 'mapping.json')
    template.update(reviewer='fixture', prior_familiarity='fixture')
    assert 'qa_' not in json.dumps(template)
    for board in template['boards']:
        for label in board['readings']:
            r = mapping[label]
            # Known, synthetic branch cases exercise scoring; never used on real answers.
            text = r['case'].split('_')[2] if r['condition'] == 'normal' else 'x'
            board['readings'][label] = dict(text=text, uncertain=False,
                                            coherent_continuation=True if r['prefix'] else None)
    submission = tmp_path / 'submission.json'
    review.write_new(submission, template)
    review.freeze(submission)
    with pytest.raises(FileExistsError):
        review.freeze(submission)
    ref_file.rename(tmp_path / 'references.json')
    review.score()
    scores = review.load(packet / 'scores.json')
    assert len(scores['rows']) == 792 and scores['fixed_cell_development_passed']
    assert scores['summaries'][-1]['paired'] == 16
    assert scores['summaries'][1]['erased'] is None and scores['summaries'][1]['coherent_books'] is None
    assert not scores['goal_achieved']


def test_empty_and_uncertain_writing():
    row = dict(prefix=True, ink=False)
    item = dict(text='答', uncertain=False, coherent_continuation=False)
    with pytest.raises(ValueError, match='empty'):
        review.validate_reading(item, row)
    review.validate_reading(dict(item, text=''), row)
    with pytest.raises(ValueError, match='uncertain'):
        review.validate_reading(dict(item, text=''), dict(row, ink=True))
