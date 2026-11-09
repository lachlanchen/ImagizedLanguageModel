#!/usr/bin/env python3
"""Check a completed assessment and lay out its original output pixels on CPU.

Output-only sheets support reading the writing before reference comparison.
This intermediate review does not replay the encoder or assign semantic grades.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')).copy()


def summary(rows):
    return dict(cases=len(rows), exact=sum(r['exact'] for r in rows),
        foreground_f1=sum(r['foreground_f1'] for r in rows)/len(rows),
        stopped=sum(r['stopped'] for r in rows), capped=sum(r['capped'] for r in rows))


def sheets(output, groups, images):
    """Paste every pixel once, wrapping only the display at 1024 columns."""
    font = ImageFont.load_default(size=14)
    pages = []
    for group, rows in groups.items():
        folder = output/'output_only'/group
        folder.mkdir(parents=True)
        for offset in range(0, len(rows), 4):
            selected = rows[offset:offset+4]
            heights = [max(1, (images[r['key']].shape[1]+1023)//1024) for r in selected]
            page = Image.new('L', (1056, 40+sum(32+36*n for n in heights)), 255)
            draw = ImageDraw.Draw(page)
            draw.text((16, 8), group+' | source order | original output pixels; display wrap only', fill=0, font=font)
            top, placements = 36, []
            for row, count in zip(selected, heights):
                label = row['key']+f" | {row['blocks']} emitted blocks"
                if row['blocks'] == 0:
                    label += ' | empty output; white display placeholder'
                draw.text((16, top), label, fill=0, font=font)
                top += 24
                image = images[row['key']]
                for start in range(0, image.shape[1], 1024):
                    part = image[:, start:start+1024]
                    page.paste(Image.fromarray(part), (16, top))
                    placements.append((16, top, part.copy()))
                    top += 36
                top += 8
            path = folder/f'{offset//4:02}.png'
            page.save(path)
            restored = png(path)
            require(all(np.array_equal(restored[y:y+p.shape[0], x:x+p.shape[1]], p) for x, y, p in placements),
                    'display sheets changed or omitted source pixels')
            pages.append(dict(path=str(path.relative_to(output)), sha256=sha(path),
                              cases=[r['key'] for r in selected]))
    return pages


def review(run, step, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for assessment review')
    require(not output.exists(), 'preserve previous and partial reviews')
    import torch
    from ilm.visual_lm.pixel_language_diversity import state_digest
    torch.set_num_threads(2)
    meta = load(run/'preparation.json')
    require(step in meta['assessment_updates'], 'a fixed registered assessment is required')
    corpus = ROOT/meta['corpus']
    manifest = load(corpus/'manifest.json')
    for name, digest in {**meta['code_sha256'], **meta['bound_sha256']}.items():
        require(sha(ROOT/name) == digest, 'bound source changed: '+name)
    for name, digest in meta['corpus_sha256'].items():
        require(sha(corpus/name) == digest, 'prepared source pixels changed: '+name)
    stage = run/'execution'/f'assessment_{step:05}'
    producer = load(stage/'report.json')
    teacher = load(stage/'teacher_history.json')
    require(producer['update'] == teacher['update'] == step and teacher['questions'] == 512
        and producer['model_sha256'] == teacher['model_sha256'], 'assessment state identity differs')
    if step == 0:
        initial = next(ROOT/name for name, digest in meta['bound_sha256'].items() if digest == meta['initial_state_sha256'])
        require(state_digest(torch.load(initial, map_location='cpu', weights_only=True)) == producer['model_sha256'],
                'initial model fingerprint differs')
    elif step in (4000, 8000):
        receipt = load(run/'milestone_models'/f'actor_update{step:05}.json')
        require(receipt['model_sha256'] == producer['model_sha256'] and receipt['assessment_sha256'] == sha(stage/'report.json')
            and receipt['archive_sha256'] == sha(run/'milestone_models'/f'actor_update{step:05}.pt'), 'owning fixed-state archive required')
    else:
        state = torch.load(run/'execution/checkpoint_latest.pt', map_location='cpu', weights_only=True)
        require(state['training_complete'] and state['update'] == step and state_digest(state['model']) == producer['model_sha256'],
                'completed final own state required')
        del state
    jobs = [(False, i, 'normal') for i in range(512 if step == 16000 else 64)]
    jobs += [(True, i, route) for i in range(24) for route in ('normal', 'blank')]
    require(len(producer['cases']) == len(jobs), 'some registered streams are missing')
    output.mkdir(parents=True)
    (output/'raw_outputs').mkdir()
    scores, images, groups = [], {}, defaultdict(list)
    for position, (dev, index, route) in enumerate(jobs):
        key = ('development' if dev else 'train')+f'_{index:04}_{route}'
        folder = stage/key
        receipt = load(folder/'receipt.json')
        require(receipt['identity'] == dict(index=index, development=dev, route=route, update=step,
            model_sha256=producer['model_sha256'], preparation_sha256=sha(run/'preparation.json'), actual_PNG_prefill=True),
            'owning output receipt differs')
        require(all(sha(folder/name) == digest for name, digest in receipt['hashes'].items()), 'original output changed')
        result = torch.load(folder/'result.pt', map_location='cpu', weights_only=True)
        blocks = result['images']
        n = len(blocks)
        require(blocks.dtype == torch.float32 and blocks.shape == (n, 1, 32, 64)
            and bool(((blocks == -1) | (blocks == 1)).all()), 'complete original binary image blocks required')
        probabilities = result['stop_probabilities']
        stopped, capped = result['stopped'], result['capped']
        require(type(stopped) is bool and type(capped) is bool and stopped != capped and n <= 128
            and len(probabilities) == n+1 and all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            and all(p < .5 for p in probabilities[:n]) and (probabilities[-1] >= .5) == stopped
            and (not capped or n == 128), 'learned stop or cap trace differs')
        ink = np.concatenate([v[0].numpy() < 0 for v in blocks], axis=1) if n else np.zeros((32, 0), bool)
        display = (~ink).astype('uint8')*255 if n else np.full((32, 64), 255, 'uint8')
        require(np.array_equal(png(folder/'answer.png'), display), 'PNG differs from actual emitted pixels')
        row = manifest['development' if dev else 'train'][index]
        references = row['references'] if dev else [dict(path=row['answer'], sha256=row['answer_sha256'])]
        exact, best_f1 = False, 0.
        for reference in references:
            target = png(corpus/reference['path']) == 0
            overlap = min(target.shape[1], ink.shape[1])
            denominator = int(target.sum()+ink.sum())
            best_f1 = max(best_f1, 2*int((target[:, :overlap] & ink[:, :overlap]).sum())/denominator if denominator else 1.)
            exact |= stopped and not capped and target.shape == ink.shape and np.array_equal(target, ink)
        score = dict(key=key, index=index, development=dev, route=route, exact=bool(exact),
                     foreground_f1=best_f1, stopped=stopped, capped=capped, blocks=n)
        require(score == producer['cases'][position], 'independent complete-image score differs')
        group = (row['source']+'_'+route) if dev else ('train_old' if index < 32 else 'train_new')
        scores.append(dict(group=group, source=row['source'], pair=row['pair'], **score))
        groups[group].append(score)
        images[key] = display
        shutil.copy2(folder/'answer.png', output/'raw_outputs'/(key+'.png'))
    pairs = {}
    for label, selected in [('train', [r for r in scores if not r['development']]),
                            ('cmrc_normal', [r for r in scores if r['development'] and r['source'] == 'cmrc' and r['route'] == 'normal'])]:
        identities = sorted({r['pair'] for r in selected})
        pairs[label] = dict(pairs=len(identities), both_exact=sum(all(r['exact'] for r in selected if r['pair'] == p) for p in identities))
    effects = {}
    for source in ('cmrc', 'dolly'):
        by = {(r['index'], r['route']): r for r in scores if r['development'] and r['source'] == source}
        ids = sorted({i for i, _ in by})
        effects[source] = sum(by[i, 'normal']['foreground_f1']-by[i, 'blank']['foreground_f1'] for i in ids)/len(ids)
    report = dict(update=step, preparation_sha256=sha(run/'preparation.json'),
        producer_report_sha256=sha(stage/'report.json'), code_sha256=sha(__file__), model_sha256=producer['model_sha256'],
        correct_history_whole512=teacher, groups={k: summary(v) for k, v in groups.items()},
        paired_questions=pairs, question_erasure_F1_effect=effects, complete_cases=scores,
        output_only_sheets=sheets(output, groups, images), all_PNGs_match_actual_emitted_tensors=True,
        all_complete_pixel_scores_match=True, all_stop_cap_events_checked=True,
        feedback_files_SHA_checked=True, feedback_CPU_reencoded=False, semantic_accuracy_measured=False,
        final_audit_still_required=True, output_only_sheets_preserve_all_original_pixels=True)
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('complete_cases', 'output_only_sheets')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--update', type=int, choices=(0, 4000, 8000, 16000), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.run.resolve(), args.update, args.output.resolve())
