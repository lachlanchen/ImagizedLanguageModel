#!/usr/bin/env python3
"""Read complete autonomous outputs anonymously before opening evaluation answers."""
import argparse
from collections import Counter
from pathlib import Path
from secrets import token_hex
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_continuous_image_language import plan
from scripts.train_continuous_image_language import EVAL, REG, ROOT, RUN, STEPS

REVIEW = RUN / 'review'
PREPARATION_SHA = 'cfa18a7c351f1751fa654cf062acd722509a34e606f36a6e8fc5c6beca92229b'


def checked_page(folder, record):
    """Compare the entire saved PNG (including padding) to actual emitted tiles."""
    case = record['input']['id']
    png, raw = folder / f'{case}.png', folder / f'{case}.npz'
    for path, name in ((png, 'output'), (raw, 'raw')):
        require(ROOT / record[name + '_file'] == path and sha(path) == record[name + '_sha256'],
                'Changed original generation file')
    n, cap = record['glyphs'], record['cap']
    require(type(n) is int and 0 <= n <= cap and record['actual_emitted_pixel_feedback'] is True
            and record['capped'] is (n == cap) and record['stopped'] is (n < cap),
            'Inconsistent stopping or feedback receipt')
    with np.load(raw, allow_pickle=False) as archive:
        tiles = archive['ink']
    require(tiles.shape == (n, 1, 32, 32) and tiles.dtype == np.uint8 and (tiles <= 1).all(),
            'Invalid original ink array')
    cols = max(1, min(32, n))
    rows = max(1, (n + cols - 1) // cols)
    with Image.open(png) as image:
        page = np.asarray(image)
        require(image.mode == 'L' and page.shape == (rows * 32, cols * 32),
                'Original answer dimensions differ')
        cells = page.reshape(rows, 32, cols, 32).transpose(0, 2, 1, 3).reshape(-1, 32, 32)
        require(np.array_equal(cells[:n], 255 - 255 * tiles[:, 0]) and np.all(cells[n:] == 255),
                'Answer pixels or padding differ from original stream')
    return png, bool(tiles.any())


def write_boards(directory, pages):
    """Paste each complete native page and any book prefix at its original scale."""
    boards, labels = [], []
    canvas = Image.new('L', (1032, 1280), 245)
    draw = ImageDraw.Draw(canvas)
    top = 0

    def flush():
        path = directory / f'board_{len(boards):03d}.png'
        canvas.crop((0, 0, canvas.width, top)).save(path)
        boards.append(dict(file=path.name, sha256=sha(path), labels=list(labels)))

    for label in sorted(pages):
        prefix, answer = pages[label]
        with Image.open(answer) as a:
            images = []
            if prefix:
                with Image.open(prefix) as p:
                    images.append(('Book prefix', p.convert('L')))
            images.append(('Output', a.copy()))
        height = sum(image.height + 20 for _, image in images) + 8
        require(height <= canvas.height and all(i.width + 8 <= canvas.width for _, i in images),
                'Native review page exceeds board dimensions')
        if top + height > canvas.height:
            flush()
            canvas.paste(245, (0, 0, canvas.width, canvas.height))
            labels.clear()
            top = 0
        for title, image in images:
            draw.text((4, top + 2), f'{label}   {title}', fill=0)
            top += 20
            canvas.paste(image, (4, top))
            top += image.height
        top += 8
        labels.append(label)
    if labels:
        flush()
    write_new(directory / 'boards.json', boards)
    template = [dict(file=b['file'], sha256=b['sha256'], readings={
        label: dict(text=None, uncertain=None, coherent_continuation=None)
        for label in b['labels']}) for b in boards]
    write_new(directory / 'readings_template.json', dict(reviewer=None, prior_familiarity=None, boards=template))


def prepare():
    report_path = RUN / 'evaluation/report.json'
    report = load(report_path)
    fit, audit, reg = load(RUN / 'report.json'), load(RUN / 'audit.json'), load(REG)
    digest = sha(REG)
    require(digest == report['registration_sha256'] == fit['registration_sha256'] == audit['registration_sha256']
            and audit['complete'] is True and audit['report_sha256'] == sha(RUN / 'report.json'),
            'Require complete audited execution')
    require(sha(EVAL.parent / 'preparation.json') == PREPARATION_SHA, 'Reference preparation receipt changed')
    expected = {(k, job['step'], job['cap'], row['id']): row for k in reg['components']
                for job in plan(reg) for row in job['inputs']}
    snapshots = {(arm['components'], s['update']): s['sha256'] for arm in fit['arms'] for s in arm['snapshots']}
    streams = report['streams']
    require(len(expected) == len(streams) == 792, 'Complete original schedule required')
    require(not REVIEW.exists(), 'Existing review must be preserved')
    mapping, pages = {}, {}
    for record in streams:
        k, step, cap = record['components'], record['step'], record['cap']
        row = record['input']
        case = row['id']
        require(expected.pop((k, step, cap, case), None) == row
                and snapshots[k, step] == record['snapshot_sha256'], 'Unexpected stream, input or snapshot')
        folder = report_path.parent / f'k{k}_step{step:07d}_cap{cap}'
        require(load(folder / f'{case}.json') == record, 'Stream receipt and manifest differ')
        png, ink = checked_page(folder, record)
        prefix = None
        if case.startswith('book_') and row['condition'] == 'normal':
            prefix = EVAL / row['image']
            require(sha(prefix) == row['image_sha256'], 'Changed book prefix')
        label = token_hex(8)
        require(label not in mapping, 'Repeated opaque label')
        pages[label] = (prefix, png)
        mapping[label] = dict(components=k, step=step, cap=cap, case=case, condition=row['condition'],
                              stopped=record['stopped'], glyphs=record['glyphs'], ink=ink,
                              prefix=prefix is not None, output_sha256=record['output_sha256'])
    # Publish the whole packet together. A failed verification cannot leave a partial packet.
    with TemporaryDirectory(dir=RUN) as scratch:
        stage = Path(scratch) / 'review'
        boards = stage / 'boards'
        boards.mkdir(parents=True)
        write_boards(boards, pages)
        write_new(stage / 'mapping.json', mapping)
        write_new(stage / 'packet.json', dict(registration_sha256=digest, report_sha256=sha(report_path),
                  boards_sha256=sha(boards / 'boards.json'), mapping_sha256=sha(stage / 'mapping.json')))
        stage.rename(REVIEW)
    print(f'Prepared {len(streams)} complete streams for anonymous reading.')


def validate_reading(item, row):
    require(set(item) == {'text', 'uncertain', 'coherent_continuation'}, 'Supply all reading fields')
    require(isinstance(item['text'], str) and type(item['uncertain']) is bool,
            'Record actual transcription and explicit uncertainty')
    coherent = item['coherent_continuation']
    require(type(coherent) is bool if row['prefix'] else coherent is None, 'Book judgment scope differs')
    if not row['ink']:
        require(not item['text'].strip() and not coherent, 'Physically empty output cannot contain a reading')
    if coherent:
        require(not item['uncertain'] and bool(item['text'].strip()), 'Complete coherent reading required')
    if not item['uncertain'] and row['ink']:
        require(bool(item['text'].strip()), 'Flag illegible marks as uncertain')


def freeze(reading_path):
    packet = load(REVIEW / 'packet.json')
    board_dir = REVIEW / 'boards'
    require(packet['registration_sha256'] == sha(REG) and packet['report_sha256'] == sha(RUN / 'evaluation/report.json')
            and packet['mapping_sha256'] == sha(REVIEW / 'mapping.json')
            and packet['boards_sha256'] == sha(board_dir / 'boards.json'), 'Changed review packet')
    mapping = load(REVIEW / 'mapping.json')
    incoming = load(reading_path)
    require(isinstance(incoming['reviewer'], str) and incoming['reviewer'].strip()
            and isinstance(incoming['prior_familiarity'], str) and incoming['prior_familiarity'].strip(),
            'Disclose reviewer and prior familiarity')
    boards = load(board_dir / 'boards.json')
    require(len(incoming['boards']) == len(boards), 'Read every board')
    readings = {}
    for board, batch in zip(boards, incoming['boards']):
        require(batch['file'] == board['file'] and batch['sha256'] == board['sha256'] == sha(board_dir / board['file'])
                and set(batch['readings']) == set(board['labels']), 'Incomplete or changed board')
        for label, item in batch['readings'].items():
            require(label not in readings, 'Duplicate reading')
            validate_reading(item, mapping[label])
            readings[label] = item
    require(readings.keys() == mapping.keys() and len(readings) == 792, 'Missing actual readings')
    write_new(REVIEW / 'frozen.json', dict(packet_sha256=sha(REVIEW / 'packet.json'),
              submission_sha256=sha(reading_path), reviewer=incoming['reviewer'],
              prior_familiarity=incoming['prior_familiarity'], readings=readings))


def summarize(rows, components, step):
    short = [r for r in rows if r['components'] == components and r['step'] == step and r['cap'] == 128]
    normal = [r for r in short if r['kind'] == 'qa' and r['condition'] == 'normal']
    pairs = Counter(r['pair'] for r in normal if r['exact'])
    exact = sum(r['exact'] for r in normal)
    erased = {c:sum(r['exact'] for r in short if r['condition'] == c)
              for c in ('question_blank', 'passage_blank')} if step in (0, STEPS[-1]) else None
    books = [r for r in rows if r['components'] == components and r['step'] == step
             and r['cap'] == 640 and r['condition'] == 'normal']
    return dict(components=components, step=step, strict=exact, paired=sum(v == 2 for v in pairs.values()),
                readable=sum(r['readable'] for r in normal), erased=erased,
                dependence_gap=exact-max(erased.values()) if erased is not None else None,
                coherent_books=sum(r['coherent'] for r in books) if books else None,
                book_stops=sum(r['stopped'] for r in books) if books else None)


def score():
    frozen = load(REVIEW / 'frozen.json')
    packet = load(REVIEW / 'packet.json')
    require(frozen['packet_sha256'] == sha(REVIEW / 'packet.json')
            and packet['mapping_sha256'] == sha(REVIEW / 'mapping.json')
            and packet['registration_sha256'] == sha(REG), 'Frozen mapping changed')
    prep, refs = EVAL.parent / 'preparation.json', EVAL.parent / 'references.json'
    require(sha(prep) == PREPARATION_SHA and sha(refs) == load(prep)['references_sha256'],
            'Changed original reference answers')
    refs_by_id = {r['id']: r for r in load(refs)['reference_rows']}
    rows = []
    for label, row in load(REVIEW / 'mapping.json').items():
        item = frozen['readings'][label]
        validate_reading(item, row)
        ref = refs_by_id[row['case'].removesuffix('_' + row['condition'])]
        normalized = ''.join(item['text'].split())
        readable = bool(normalized and row['ink'] and not item['uncertain'])
        exact = readable and row['stopped'] and normalized == ''.join(ref['answer'].split())
        rows.append(dict(**row, label=label, kind=ref['kind'], pair=ref.get('paired_group'),
                         readable=readable, exact=exact, coherent=bool(readable and row['stopped']
                         and item['coherent_continuation'])))
    summaries = [summarize(rows, k, step) for k in (1, 8) for step in STEPS]
    final = summaries[-1]
    passed = (final['strict'] >= 12 and final['paired'] >= 4 and final['readable'] >= 24
              and final['coherent_books'] >= 4 and max(final['erased'].values()) <= 2
              and final['dependence_gap'] >= 8)
    write_new(REVIEW / 'scores.json', dict(frozen_sha256=sha(REVIEW / 'frozen.json'),
              references_sha256=sha(refs), rows=rows, summaries=summaries,
              fixed_cell_development_passed=passed, goal_achieved=False))
    print(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'freeze', 'score'))
    parser.add_argument('--readings', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'score':
        score()
    elif args.readings:
        freeze(args.readings)
    else:
        parser.error('freeze requires --readings')
