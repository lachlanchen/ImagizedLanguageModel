#!/usr/bin/env python3
"""Independent physical scoring of every registered first-block output.

This audit does not infer semantic accuracy from pixel agreement and does not
replay the inherited visual/contextual model or every neural generation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run(output):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for independent offline scoring')
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    meta, cases = read(output/'preparation.json'), read(output/'cases.json')
    for name, expected in {**meta['code_sha256'], **meta['bound_sha256']}.items():
        if digest(root/name) != expected:
            raise ValueError('bound source changed: ' + name)
    if digest(output/'data.pt') != meta['data_sha256'] or digest(output/'cases.json') != meta['cases_sha256']:
        raise ValueError('prepared data changed')
    execution = output/'execution'
    report = read(execution/'report.json')
    if read(execution/'status.json')['phase'] != 'completed' or read(execution/'status.json')['pid'] is not None:
        raise ValueError('finish GPU execution before offline audit')
    for name, key in (('ink.npy', 'output_sha256'), ('final.pt', 'final_checkpoint_sha256'), ('nll.pt', 'nll_sha256')):
        if digest(execution/name) != report[key]:
            raise ValueError('actual result binding changed')
    data = torch.load(output/'data.pt', map_location='cpu', weights_only=True)
    # Independently unpack every source physical tile in NumPy.
    packed = data['patterns'].numpy()
    reference = (((packed[..., None] >> np.arange(8)) & 1).reshape(704, 16, 16, 2, 4)
        .transpose(0, 1, 3, 2, 4).reshape(704, 32, 64).astype(bool))
    if not np.array_equal(reference, data['ink'].numpy()):
        raise ValueError('independent packing inverse differs')
    normal = data['normal'].reshape(704, -1).numpy()
    if not np.allclose(normal[:512].mean(0), data['mean'].numpy(), atol=1e-4, rtol=1e-5):
        raise ValueError('normalization is not training only')
    actual = np.load(execution/'ink.npy', allow_pickle=False)
    if actual.shape != (896, 32, 64) or actual.dtype != np.bool_:
        raise ValueError('complete binary output cohort required')
    truth = np.concatenate((reference, reference[512:]))
    rows = []
    for i, (a, b) in enumerate(zip(actual, truth)):
        path = execution/f'images/{i:04}.png'
        if digest(path) != report['png_sha256'][f'images/{i:04}.png']:
            raise ValueError('original output PNG changed')
        with Image.open(path) as im:
            pixels = np.asarray(im.convert('L'))
        if not np.array_equal(pixels, (~a).astype('uint8')*255):
            raise ValueError('PNG differs from actual emitted mask')
        tp, fp, fn = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        source_index = i if i < 704 else i-192
        rows.append(dict(output_index=i, source_index=source_index, id=cases[source_index]['id'],
            source=cases[source_index]['source'], split=cases[source_index]['split'],
            route='normal' if i < 704 else 'blank', exact=bool(np.array_equal(a, b)), f1=f1,
            true_positive=tp, false_positive=fp, false_negative=fn))
    groups = {}
    for split in ('train', 'development'):
        for source in ('cmrc', 'dolly'):
            for route in ('normal', 'blank') if split == 'development' else ('normal',):
                part = [r for r in rows if (r['split'], r['source'], r['route']) == (split, source, route)]
                groups[f'{split}_{source}_{route}'] = dict(count=len(part), exact=sum(r['exact'] for r in part),
                    foreground_f1=float(np.mean([r['f1'] for r in part])))
    chinese = [i for i, c in enumerate(cases) if c['split'] == 'development' and c['source'] == 'cmrc']
    articles = sorted({str(cases[i]['article']) for i in chinese})
    if len(chinese) != 128 or len(articles) != 64:
        raise ValueError('all 64 development question pairs required')
    both, effects = 0, []
    for article in articles:
        ix = [i for i in chinese if str(cases[i]['article']) == article]
        if len(ix) != 2:
            raise ValueError('paired source article changed')
        both += int(all(rows[i]['exact'] for i in ix))
        effects.append(np.mean([rows[i]['f1']-rows[i+192]['f1'] for i in ix]))
    rng = np.random.default_rng(202611092)
    intervals = np.mean(np.asarray(effects)[rng.integers(64, size=(2000, 64))], axis=1)
    audit = dict(groups=groups, complete_original_pngs_scored=896, every_png_matches_emitted_bits=True,
        all_reference_packing_independently_verified=True, train_only_mean_independently_checked=True,
        chinese_pairs_both_exact=both, chinese_question_effect_f1=float(np.mean(effects)),
        chinese_question_effect_95_article_bootstrap=np.quantile(intervals, [.025, .975]).tolist(),
        semantic_accuracy_measured=False, inherited_actor_forward_replayed=False,
        all_neural_generations_replayed=False, complete_answer_claim=False,
        general_ilm_goal_complete=False, report_sha256=digest(execution/'report.json'), cases=rows)
    path = output/'audit.json'
    if path.exists():
        raise FileExistsError('completed audits are immutable')
    path.write_text(json.dumps(audit, indent=2)+'\n')
    plates = output/'plates'
    plates.mkdir(exist_ok=False)
    hashes = {}
    for group, indices in [('train', list(range(512))), ('development_zh', chinese),
            ('development_en', [i for i, c in enumerate(cases) if c['split']=='development' and c['source']=='dolly'])]:
        height = 84 if group == 'train' else 116
        for start in range(0, len(indices), 128):
            chosen = indices[start:start+128]
            sheet = Image.new('L', (16*72, math_ceil(len(chosen), 16)*height), 255)
            draw = ImageDraw.Draw(sheet)
            for j, i in enumerate(chosen):
                x, y = (j%16)*72, (j//16)*height
                draw.text((x, y), str(i), fill=0)
                pieces = [reference[i], actual[i]] + ([] if group == 'train' else [actual[i+192]])
                for k, ink in enumerate(pieces):
                    sheet.paste(Image.fromarray((~ink).astype('uint8')*255), (x, y+16+32*k))
            name = f'{group}_{start:04}.png'
            sheet.save(plates/name)
            hashes[name] = digest(plates/name)
    (output/'plates.json').write_text(json.dumps(dict(order='reference; normal; blank where present',
        native_pixel_scale=True, output_changes=False, files=hashes), indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'cases'}, indent=2))


def math_ceil(value, divisor):
    return (value + divisor - 1)//divisor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
