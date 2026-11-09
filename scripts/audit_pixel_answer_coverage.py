#!/usr/bin/env python3
"""Offline source-code-point coverage, authenticated against actual answer PNGs.

This analysis neither supplies labels to the actor nor changes its examples.
Code points in current answer supervision are not the model's glyph inventory.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ilm.visual_lm.full_text_renderer import CompleteLineRenderer

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'train': ('data/visual_qa/cmrc2018_short/train.jsonl', '742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c'),
    'development': ('data/visual_qa/cmrc2018_short/development.jsonl', '8bef093075ac0715d5878a1abe61aa38092893ade8753b935cdf657551ebc6a7'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(corpus, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for source-only analysis')
    require(not output.exists(), 'preserve existing coverage analyses')
    renderer = CompleteLineRenderer()
    preparation = load(ROOT/'data/general_instructions/complete_lines_train_20261105/identity.json')
    require(renderer.provenance == preparation['renderer'], 'original answer renderer differs')
    for name, digest in preparation['source_code_sha256'].items():
        require(sha(ROOT/name) == digest, 'original rendering source changed')
    manifest = load(corpus/'manifest.json')
    require(len(manifest['train']) == 512, 'complete registered cohort required')
    source = {}
    for split, (name, expected) in SOURCES.items():
        require(sha(ROOT/name) == expected, 'owning source text changed')
        source[split] = {r['id']: r for r in map(json.loads, (ROOT/name).read_text().splitlines())}
    bindings = {}
    rendered = 0
    def actual(ref):
        nonlocal rendered
        path = corpus/ref['path']
        require(sha(path) == ref['sha256'], 'prepared complete reference changed')
        bindings[ref['path']] = ref['sha256']
        with Image.open(path) as image:
            return np.asarray(image.convert('L')).copy()
    def render(text):
        nonlocal rendered
        image, _ = renderer.render(text, maximum_blocks=128)
        rendered += 1
        return image
    answers = []
    for row in manifest['train']:
        record = source['train'][Path(row['source']).stem]
        text = record['references'][0]['text']
        require(np.array_equal(render(text), actual(dict(path=row['answer'], sha256=row['answer_sha256']))),
                'source text does not reproduce owning complete target')
        answers.append(text)
    development = []
    for row in manifest['development']:
        if row['source'] != 'cmrc':
            continue
        record = source['development'][row['id']]
        candidates = [(r['text'], render(r['text'])) for r in record['references']]
        matched = []
        for ref in row['references']:
            image = actual(ref)
            texts = sorted({text for text, pixels in candidates if np.array_equal(pixels, image)})
            require(bool(texts), 'a complete development reference has no matching source rendering')
            matched.extend(texts)
        development.append(dict(index=row['index'], id=row['id'], reference_texts=sorted(set(matched))))
    require(len(development) == 16, 'all fixed Chinese development questions required')
    summaries = {}
    for size in (32, 512):
        counts = Counter(''.join(answers[:size]))
        seen = set(counts)
        cases = []
        for row in development:
            options = [dict(text=text, absent_codepoints=''.join(sorted(set(text)-seen))) for text in row['reference_texts']]
            cases.append(dict(index=row['index'], id=row['id'], references=options,
                any_complete_reference_covered=any(not r['absent_codepoints'] for r in options)))
        summaries[str(size)] = dict(training_answers=size, source_codepoints=len(counts),
            source_codepoint_occurrences=sum(counts.values()),
            Chinese_development_cases=len(cases), complete_reference_covered=sum(r['any_complete_reference_covered'] for r in cases),
            cases=cases)
    require(all(sha(corpus/name) == digest for name, digest in bindings.items()), 'analysis modified owning reference images')
    result = dict(corpus_manifest_sha256=sha(corpus/'manifest.json'),
        source_sha256={name: digest for name, digest in SOURCES.values()},
        analysis_code_sha256=sha(__file__), renderer=renderer.provenance,
        all512_training_targets_reproduced_from_owning_text=True,
        all16_Chinese_development_reference_sets_reproduced_from_owning_text=True,
        source_render_calls=rendered, model_calls=0, GPU_calls=0,
        source_labels_enter_model=False, training_protocol_or_data_changed=False,
        original32_development_outcomes_previously_known=True,
        current512_model_outputs_used_for_coverage=False, pretraining_glyph_coverage_measured=False,
        source_licenses=dict(cmrc='CC-BY-SA-4.0'),
        interpretation='Coverage of Unicode code points in this answer-supervision cohort only; not glyph identity, prior model exposure, semantic accuracy or an exemption from unseen-form generation.',
        reference_image_sha256=bindings, cohorts=summaries)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: {n: v for n, v in s.items() if n != 'cases'} for k, s in summaries.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT/'artifacts/pixel_language_diversity_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.corpus.resolve(), args.output.resolve())
