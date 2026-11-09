#!/usr/bin/env python3
"""Locate physical errors in saved image answers, without neural inference.

An existing tensor/PNG audit is required. This analysis compares physical
prefixes and input interventions; it does not grade semantics or resample.
The optional earlier-reader comparison performs image resizing only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')).copy()


def image_key(image):
    return (image.shape, image.tobytes()) if image is not None else None


def earlier_reader(folder, meta, corpus, manifest):
    """Compare original and resized pixels, not a new language-model forward."""
    import torch
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    require(not torch.cuda.is_initialized(), 'CPU image preprocessing only')
    torch.set_num_threads(2)
    preparation = load(folder/'preparation.json')
    verification = load(folder/'verification.json')
    require(verification['generation_report_sha256'] == sha(folder/'generation_report.json'),
            'earlier reader generation report changed')
    require(sha(folder/'structural_vectors.pt') == preparation['structural_vectors_sha256']
        and preparation['structural_vectors_sha256'] in meta['bound_sha256'].values(),
        'the earlier and current fixed controls differ')
    by_id = {r['id']: r for r in manifest['development']}
    cases = []
    for row in preparation['cases']:
        current = by_id[row['id']]
        raw = row['images']['raw/normal']
        resized = row['images']['native_budget/normal']
        require(sha(folder/raw['file']) == raw['sha256']
            and sha(folder/resized['file']) == resized['sha256'], 'earlier input PNG changed')
        original = pixels(folder/raw['file'])
        now = pixels(corpus/current['normal'])
        require(sha(corpus/current['normal']) == current['normal_sha256'], 'current original input changed')
        observed = native_budget_image(torch.from_numpy(now).float()[None, None]/127.5-1)
        restored = ((observed[0, 0]+1)*127.5).round().clamp(0, 255).byte().numpy()
        cases.append(dict(position=row['position'], id=row['id'],
            original_pixels_equal=bool(np.array_equal(original, now)),
            resized_pixels_equal=bool(np.array_equal(pixels(folder/resized['file']), restored))))
    common = {name: sha(ROOT/name) == digest for name, digest in preparation['sources'].items()
              if name.startswith('ilm/')}
    answers = [r for r in verification['measurements'] if r['route'] == 'native_budget/normal']
    return dict(preparation_sha256=sha(folder/'preparation.json'),
        verification_sha256=sha(folder/'verification.json'), cases=cases,
        common_source_files_unchanged=common, earlier_answers=answers,
        earlier_literal_exact=sum(r['exact'] for r in answers),
        earlier_helper_only_symbolic_readout=True, whole_current_reader_forward_replayed=False,
        interpretation='Identical inputs, controls and shared loader source support a bounded reader comparison. This is not a replay or proof of the complete current reader computation.')


def analyze(run, reviews, output, reader):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for saved-output analysis')
    require(not output.exists(), 'preserve existing and partial analyses')
    meta = load(run/'preparation.json')
    corpus = ROOT/meta['corpus']
    require(sha(corpus/'manifest.json') == meta['corpus_sha256']['manifest.json'], 'corpus manifest changed')
    manifest = load(corpus/'manifest.json')
    stages = []
    for review_path in reviews:
        review = load(review_path)
        step = review['update']
        require(step in meta['assessment_updates'] and review['preparation_sha256'] == sha(run/'preparation.json'),
                'a registered, owning reviewed assessment is required')
        require(review['all_PNGs_match_actual_emitted_tensors'] and review['all_complete_pixel_scores_match']
            and review['all_stop_cap_events_checked'], 'the original complete tensor/PNG audit must pass')
        stage = run/'execution'/f'assessment_{step:05}'
        producer = load(stage/'report.json')
        require(sha(stage/'report.json') == review['producer_report_sha256']
            and producer['model_sha256'] == review['model_sha256'], 'owning assessment changed')
        require([r['key'] for r in producer['cases']] == [r['key'] for r in review['complete_cases']],
                'review case order differs from original assessment')
        groups, all_cases, values = defaultdict(list), [], {}
        for row in review['complete_cases']:
            folder = stage/row['key']
            receipt = load(folder/'receipt.json')
            require(receipt['identity']['model_sha256'] == review['model_sha256']
                and receipt['identity']['update'] == step
                and sha(folder/'answer.png') == receipt['hashes']['answer.png'], 'saved original output changed')
            observed = pixels(folder/'answer.png')
            require(observed.shape == (32, 64*max(1, row['blocks']))
                and bool(((observed == 0) | (observed == 255)).all()), 'original binary output geometry differs')
            first = observed[:, :64] if row['blocks'] else None
            source = manifest['development' if row['development'] else 'train'][row['index']]
            refs = source['references'] if row['development'] else [dict(path=source['answer'], sha256=source['answer_sha256'])]
            reference_first = set()
            for ref in refs:
                path = corpus/ref['path']
                require(sha(path) == ref['sha256'], 'original reference changed')
                target = pixels(path)
                require(target.shape[0] == 32 and target.shape[1] >= 64 and target.shape[1] % 64 == 0,
                        'complete reference image blocks required')
                reference_first.add(image_key(target[:, :64]))
            score = dict(key=row['key'], group=row['group'], index=row['index'],
                development=row['development'], route=row['route'], pair=row['pair'],
                blocks=row['blocks'], first_block_exact=bool(first is not None and image_key(first) in reference_first),
                complete_image_exact=row['exact'], reference_first_block_variants=len(reference_first))
            groups[row['group']].append(score)
            all_cases.append(score)
            values[row['key']] = dict(row=row, first=image_key(first),
                stream=(image_key(observed) if row['blocks'] else None, row['stopped'], row['capped']),
                references=reference_first)
        pairs = {}
        for label, selected in [('train', [v for v in values.values() if not v['row']['development']]),
                                ('cmrc_normal', [v for v in values.values() if v['row']['group'] == 'cmrc_normal'])]:
            by_pair = defaultdict(list)
            for v in selected:
                by_pair[v['row']['pair']].append(v)
            items = []
            for pair, cases in sorted(by_pair.items()):
                require(len(cases) == 2, 'an entire same-passage question pair is required')
                a, b = cases
                items.append(dict(pair=pair, keys=[a['row']['key'], b['row']['key']],
                    first_blocks_identical=a['first'] == b['first'], complete_streams_identical=a['stream'] == b['stream'],
                    reference_first_block_sets_overlap=bool(a['references'] & b['references'])))
            pairs[label] = items
        erasure = {}
        for source in ('cmrc', 'dolly'):
            selected = [v for v in values.values() if v['row']['group'] == source+'_normal']
            compared = [(v, values[v['row']['key'].replace('_normal', '_blank')]) for v in selected]
            erasure[source] = dict(cases=len(compared),
                first_block_unchanged=sum(a['first'] == b['first'] for a, b in compared),
                complete_stream_unchanged=sum(a['stream'] == b['stream'] for a, b in compared))
        stages.append(dict(update=step, original_review_sha256=sha(review_path), model_sha256=review['model_sha256'],
            groups={label: dict(cases=len(rows), empty_outputs=sum(r['blocks'] == 0 for r in rows),
                first_block_exact=sum(r['first_block_exact'] for r in rows),
                complete_image_exact=sum(r['complete_image_exact'] for r in rows),
                first_exact_but_complete_not_exact=sum(r['first_block_exact'] and not r['complete_image_exact'] for r in rows))
                for label, rows in groups.items()}, cases=all_cases, paired_question_output_comparisons=pairs,
            question_erasure_output_comparisons=erasure))
    require(len({s['update'] for s in stages}) == len(stages), 'duplicate assessment update')
    report = dict(analysis_code_sha256=sha(Path(__file__)), preparation_sha256=sha(run/'preparation.json'),
        stages=stages, neural_model_forward_calls=0, optimizer_updates=0, output_images_generated=0,
        outputs_repaired=False, semantic_accuracy_measured=False,
        limitation='First-block pixel mismatch is not semantic grading. It precedes feedback of any previous complete image, but not autoregression over the 256 physical pieces inside that first block.',
        earlier_reader_comparison=earlier_reader(reader, meta, corpus, manifest) if reader else None)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(stages=[dict(update=s['update'], groups=s['groups'], erasure=s['question_erasure_output_comparisons'])
                                 for s in stages], reader_pixel_comparisons=report['earlier_reader_comparison']['cases'] if reader else None), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, action='append', required=True)
    parser.add_argument('--earlier-reader', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), [p.resolve() for p in args.review], args.output.resolve(),
            args.earlier_reader.resolve() if args.earlier_reader else None)
