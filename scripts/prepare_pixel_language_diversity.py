#!/usr/bin/env python3
"""Prepare a fixed 512-question visual diversity cohort, without model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'data/general_instructions/complete_lines_train_20261105'
TRAIN_SHA = '0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e'
SEED = 'complete-pixel-diversity-20261109-v1'
SOURCES = ('scripts/prepare_pixel_language_diversity.py', 'ilm/visual_lm/complete_writing_data.py',
           'ilm/visual_lm/causal_visual_prefix.py', 'references/complete_pixel_diversity_direction.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def article(name):
    match = re.fullmatch(r'cmrc/(TRAIN_\d+)_QUERY_(\d+)\.pt', name)
    if not match:
        raise ValueError('unexpected Chinese training identity: ' + name)
    return match[1], int(match[2])


def pixels(tensor):
    if not bool(((tensor == -1) | (tensor == 1)).all()):
        raise ValueError('original binary writing required')
    return (tensor[0, 0].numpy() > 0).astype('uint8') * 255


def prepare(output, original):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for this CPU-only preparation')
    torch.set_num_threads(2)
    if output.exists():
        raise FileExistsError('preserve the existing cohort and choose a fresh output')
    code = {name: sha(ROOT / name) for name in SOURCES}
    for name, digest in code.items():
        raw = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('commit the preparation source first: ' + name)
    status = load(original / 'execution/status.json')
    if status['phase'] != 'completed' or status['pid'] is not None or not (original / 'audit.json').is_file():
        raise ValueError('the completed independently audited original run is required')
    old = load(original / 'cases.json')
    data = CompleteWritingImageData(TRAIN, purpose='train', expected_report_sha256=TRAIN_SHA)
    anchors = [row['id'] for row in old[:32]]
    if len(set(anchors)) != 32 or any(name not in data.names for name in anchors):
        raise ValueError('the original 32 complete Chinese sources must exist')
    old_articles = {article(name)[0] for name in anchors}
    if len(old_articles) != 16:
        raise ValueError('the original 16 passage pairs differ')
    groups = {}
    for name in data.names:
        if name.startswith('cmrc/') and article(name)[0] not in old_articles:
            groups.setdefault(article(name)[0], []).append(name)

    def images(name):
        full, boundary = data.get(name)
        prompt = fold_writing_prompt(full[..., :boundary * 64]).pixels
        answer = full[..., boundary * 64:-128]
        if answer.shape[-1] % 64 or not 1 <= answer.shape[-1] // 64 <= 128:
            raise ValueError('complete answer bounds differ')
        return pixels(prompt), pixels(answer)

    chosen = list(anchors)
    selected_articles = []
    ordering = sorted(groups, key=lambda name: hashlib.sha256((SEED + '\0' + name).encode()).hexdigest())
    for name in ordering:
        pair = []
        targets = set()
        for candidate in sorted(groups[name], key=lambda item: article(item)[1]):
            _, answer = images(candidate)
            digest = hashlib.sha256(answer.tobytes()).hexdigest()
            if digest not in targets:
                pair.append(candidate)
                targets.add(digest)
            if len(pair) == 2:
                break
        if len(pair) == 2:
            chosen.extend(pair)
            selected_articles.append(name)
        if len(selected_articles) == 240:
            break
    if len(chosen) != 512 or len(set(chosen)) != 512:
        raise ValueError('240 additional distinct-answer passage pairs are required')
    output.mkdir(parents=True)
    train = output / 'train'
    train.mkdir()
    rows = []
    prompt_digests = set()
    for index, name in enumerate(chosen):
        prompt, answer = images(name)
        digest = hashlib.sha256(prompt.tobytes()).hexdigest()
        if digest in prompt_digests:
            raise ValueError('duplicate complete prompt pixels')
        prompt_digests.add(digest)
        if index < 32:
            for actual, expected in ((prompt, original / old[index]['normal']),
                                     (answer, original / old[index]['references'][0]['path'])):
                with Image.open(expected) as image:
                    if not np.array_equal(actual, np.asarray(image.convert('L'))):
                        raise ValueError('an original positive-control image changed')
        folder = train / f'{index:04}'
        folder.mkdir()
        Image.fromarray(prompt).save(folder / 'prompt.png')
        Image.fromarray(answer).save(folder / 'answer.png')
        rows.append(dict(index=index, pair=index // 2, source=name, article=article(name)[0],
            original_positive_control=index < 32, split='train', prompt=f'train/{index:04}/prompt.png',
            prompt_sha256=sha(folder / 'prompt.png'), prompt_shape=list(prompt.shape),
            answer=f'train/{index:04}/answer.png', answer_sha256=sha(folder / 'answer.png'),
            answer_blocks=answer.shape[1] // 64, source_payload_sha256=data.report['files_sha256'][name]))
    development = []
    for case in old[32:]:
        index = case['index']
        folder = output / 'development' / f'{index:02}'
        folder.mkdir(parents=True)
        record = dict(index=index, id=case['id'], source=case['source'], pair=case['pair'], split='development', references=[])
        for route in ('normal', 'blank'):
            source, target = original / case[route], folder / (route + '.png')
            if sha(source) != case[route + '_sha256']:
                raise ValueError('the original fixed development input changed')
            shutil.copy2(source, target)
            record[route], record[route + '_sha256'] = str(target.relative_to(output)), sha(target)
            with Image.open(target) as image:
                digest = hashlib.sha256(np.asarray(image.convert('L')).tobytes()).hexdigest()
                if digest in prompt_digests:
                    raise ValueError('a development prompt duplicates a training prompt')
        for j, ref in enumerate(case['references']):
            source, target = original / ref['path'], folder / f'reference_{j:02}.png'
            if sha(source) != ref['sha256']:
                raise ValueError('a complete development reference changed')
            shutil.copy2(source, target)
            record['references'].append(dict(path=str(target.relative_to(output)), sha256=sha(target)))
        development.append(record)
    manifest = dict(seed=SEED, train=rows, development=development, train_questions=512, train_pairs=256,
        original_positive_controls=32, added_articles=240, all32_original_prompt_and_target_pixels_exact=True,
        image_report_sha256=TRAIN_SHA, code_sha256=code, original_cases_sha256=sha(original / 'cases.json'),
        original_audit_sha256=sha(original / 'audit.json'), original_preparation_sha256=sha(original / 'preparation.json'),
        selection='old32 plus240 hash-ordered new training articles, first two distinct complete targets in question order',
        original_source_noise_retained=True, training_and_development_paths_separate=True,
        development_sealed=False, model_loaded=False, optimizer_updates=0, CUDA_initialized=torch.cuda.is_initialized(),
        maximum_answer_blocks=max(r['answer_blocks'] for r in rows),
        maximum_prompt_pixels=max(np.prod(r['prompt_shape']).item() for r in rows),
        language_generalization_qualified=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('train', 'development', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original-run', type=Path, default=ROOT / 'artifacts/complete_pixel_language_v1')
    args = parser.parse_args()
    prepare(args.output.resolve(), args.original_run.resolve())
