#!/usr/bin/env python3
"""Prepare prospective public-data reading inputs, never model outputs.

C3 is restricted to non-commercial research. Raw texts, labels and rasterized
pages stay in ignored data/artifacts directories with their source notice.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata
import urllib.request

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OWN = 'scripts/prepare_c3_visual_reading.py'
PROTOCOL = 'references/c3_visual_reading_protocol.md'
REVISION = '1f41682ebb684d63afe7c583c8f38e137fc8a826'
FILES = {'data/c3-m-dev.json': (1103725, '32f7aa82d01a23e88d1e37eb21b4289fc008975a'),
         'license.txt': (66, 'af9853d32bd24e04354b53270720302b564f7777'),
         'README.md': (5211, '4ccde9d1606c5bbf71aa3fdc4fd0783cfec1238a')}
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
SEED = 'prospective-c3-reading-20261110-v1'
SIZE = (960, 1024)
INSTRUCTIONS = {
    'letter': '根据短文和问题选择答案。只写选项字母，不要解释。',
    'phrase': '根据短文和问题选择答案。只抄写正确选项的内容，不要字母或解释。'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def identity():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation requires CUDA hidden')
    for name in (OWN, PROTOCOL):
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit preparation and protocol first')
    return {name: sha(ROOT / name) for name in (OWN, PROTOCOL)}


def verify_source(source):
    for name, (size, blob) in FILES.items():
        raw = (source / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        require(len(raw) == size and actual == blob, 'pinned C3 source differs: ' + name)


def acquire(source):
    code = identity()
    require(not source.exists(), 'reuse verified source with prepare; preserve partial downloads')
    source.mkdir(parents=True)
    receipts = {}
    for name, (size, blob) in FILES.items():
        url = f'https://raw.githubusercontent.com/nlpdata/c3/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(size + 1)
        require(len(raw) == size and hashlib.sha1(
            b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() == blob,
            'download does not match pinned Git blob')
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        receipts[name] = dict(url=url, bytes=size, git_blob=blob, sha256=sha(path))
    verify_source(source)
    save(source / 'source_receipt.json', dict(revision=REVISION, files=receipts, code_sha256=code,
         license='non-commercial research purpose only', model_loaded=False,
         development_labels_acquired=True, test_labels_acquired=False))
    print(json.dumps(dict(revision=REVISION, downloaded_bytes=sum(v[0] for v in FILES.values()))))


def normalized(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def wrapped(text, font):
    draw = ImageDraw.Draw(Image.new('L', (1, 1)))
    rows = []
    for paragraph in text.replace('\r\n', '\n').replace('\t', '    ').split('\n'):
        line = ''
        for char in paragraph:
            if line and draw.textlength(line + char, font=font) > SIZE[0] - 64:
                rows.append(line)
                line = ''
            line += char
        rows.append(line)
    return rows


def input_page(context, question, choices, style, font):
    """Correct-answer identity is deliberately absent from this API."""
    header = wrapped(INSTRUCTIONS[style], font)
    body = wrapped('短文：\n' + context + '\n\n问题：' + question + '\n' +
                   '\n'.join(f'{chr(65 + i)}. {c}' for i, c in enumerate(choices)), font)
    line_height = font.size + 8
    if len(header) * line_height > 96 or 144 + len(body) * line_height > SIZE[1] - 24:
        return None
    page = Image.new('L', SIZE, 255)
    draw = ImageDraw.Draw(page)
    for start, rows in [(24, header), (144, body)]:
        for i, row in enumerate(rows):
            draw.text((32, start + i * line_height), row, font=font, fill=0, anchor='lt')
    return page


def prepare(source, output):
    code = identity()
    verify_source(source)
    require(not output.exists(), 'preserve earlier cohorts')
    genuine = ROOT / 'artifacts/genuine_glyph_readout_v1'
    require(not (genuine / 'qualify/report.json').exists(),
            'freeze this cohort before genuine-copy qualification results')
    original = ROOT / 'artifacts/semantic_glyph_bridge_v1'
    require(sha(FONT) == load(original / 'preparation.json')['font_sha256'], 'font changed')
    face = TTFont(FONT, fontNumber=2)
    require('SC' in face['name'].getDebugName(1), 'SC instruction face required')
    cmap = face.getBestCmap()
    known_contexts = set()
    for split in ('train', 'development'):
        for line in (ROOT / f'data/visual_qa/cmrc2018_short/{split}.jsonl').read_text().splitlines():
            known_contexts.add(normalized(json.loads(line)['context']))
    inventory = {c for r in load(original / 'inventory.json')['records'] for c in r['chars']}
    docs = load(source / 'data/c3-m-dev.json')
    require(len({str(d[2]) for d in docs}) == len(docs), 'source document IDs are not unique')
    order = sorted(docs, key=lambda d: hashlib.sha256((SEED + '\0' + str(d[2])).encode()).hexdigest())
    selected, skipped, pages = [], [], []
    for document in order:
        require(len(document) == 3 and document[0] and document[1], 'unexpected C3 document')
        context = '\n'.join(document[0])
        item = document[1][0]  # Fixed first question; never selected by its answer.
        choices = item['choice']
        require(2 <= len(choices) <= 26 and len(set(choices)) == len(choices)
                and choices.count(item['answer']) == 1, 'invalid source question')
        doc_id = str(document[2])
        if normalized(context) in known_contexts:
            skipped.append(dict(document=doc_id, reason='exact_normalized_CMRC_context_overlap'))
            continue
        visible = context + item['question'] + ''.join(choices) + ''.join(INSTRUCTIONS.values())
        missing = sorted({c for c in visible if not c.isspace() and ord(c) not in cmap})
        if missing:
            skipped.append(dict(document=doc_id, reason='input_font_coverage', codepoints=[ord(c) for c in missing]))
            continue
        shift = 1 + int(hashlib.sha256(('rotate\0' + doc_id).encode()).hexdigest(), 16) % (len(choices) - 1)
        permutations = {'original': list(range(len(choices))),
                        'rotated': list(range(shift, len(choices))) + list(range(shift))}
        rendered = None
        for font_size in (24, 22, 20):
            font = ImageFont.truetype(str(FONT), font_size, index=2)
            candidate = {}
            for version, permutation in permutations.items():
                for style in INSTRUCTIONS:
                    candidate[version, style] = input_page(context, item['question'],
                        [choices[i] for i in permutation], style, font)
            if all(p is not None for p in candidate.values()):
                rendered = candidate
                break
        if rendered is None:
            skipped.append(dict(document=doc_id, reason='complete_page_exceeds_fixed_canvas_at20px'))
            continue
        selected.append(dict(document=doc_id, question_index=0, font_size=font_size,
                             choice_count=len(choices), permutations=permutations,
                             context_sha256=hashlib.sha256(context.encode()).hexdigest()))
        pages.append((rendered, item))
        if len(selected) == 16:
            break
    require(len(selected) == 16, 'insufficient complete source pages')
    output.mkdir(parents=True)
    (output / 'prompts').mkdir()
    (output / 'previews').mkdir()
    records, references, boards = [], [], []
    for index, (selected_row, (rendered, item)) in enumerate(zip(selected, pages)):
        board = Image.new('L', (960, 1088), 245)
        draw = ImageDraw.Draw(board)
        for col, ((version, style), page) in enumerate(rendered.items()):
            case_id = f'doc{index:02}_{version}_{style}'
            name = 'prompts/' + case_id + '.png'
            page.save(output / name)
            correct_original = item['choice'].index(item['answer'])
            letter = chr(65 + selected_row['permutations'][version].index(correct_original))
            expected = letter if style == 'letter' else item['answer']
            records.append(dict(case_id=case_id, document_index=index, version=version, style=style,
                                prompt=name, prompt_sha256=sha(output / name),
                                maximum_glyphs=8 if style == 'letter' else 128))
            references.append(dict(case_id=case_id, expected=expected,
                                   unsupported_canonical_chars=sorted(set(expected) - inventory)))
            x, y = (col % 2) * 480, (col // 2) * 544
            draw.text((x + 4, y + 4), f'{index:02} {version} {style}', fill=0)
            board.paste(page.resize((480, 512), Image.Resampling.LANCZOS), (x, y + 24))
        name = f'previews/document_{index:02}.png'
        board.save(output / name)
        boards.append(dict(file=name, sha256=sha(output / name)))
    for i in range(16):
        refs = {r['case_id']: r['expected'] for r in references}
        require(refs[f'doc{i:02}_original_letter'] != refs[f'doc{i:02}_rotated_letter']
                and refs[f'doc{i:02}_original_phrase'] == refs[f'doc{i:02}_rotated_phrase'],
                'rotation did not change letter and preserve phrase')
    save(output / 'evaluation_references.json', references)
    save(output / 'preparation.json', dict(
        source_revision=REVISION, source_receipt_sha256=sha(source / 'source_receipt.json'),
        source_file_sha256=sha(source / 'data/c3-m-dev.json'), code_sha256=code,
        font_sha256=sha(FONT), font_face_index=2, canvas_size=list(SIZE), seed=SEED,
        selection='hash-ordered documents; first question; input font/layout feasibility only',
        selected=selected, skipped=skipped, cases=records, preview_boards=boards,
        evaluation_references_sha256=sha(output / 'evaluation_references.json'),
        prospective_genuine_preparation_sha256=sha(genuine / 'preparation.json'),
        genuine_qualification_available_at_preparation=False,
        exact_normalized_CMRC_context_overlap=0,
        current_training_reads_C3=False, source_pretraining_exposure_unknown=True,
        expected_answer_coverage_used_to_select=False, source_answer_used_to_crop=False,
        renderer_receives_correct_answer=False, model_loaded=False, neural_forward_calls=0,
        emitted_model_answers=0, input_visual_review_pending=True,
        license='non-commercial research purpose only'))
    summary = dict(documents=16, normal_inputs=len(records), styles=list(INSTRUCTIONS),
                   fonts=dict(Counter(r['font_size'] for r in selected)),
                   skipped_source_documents=len(skipped),
                   references_outside_canonical_coverage=sum(bool(r['unsupported_canonical_chars']) for r in references),
                   maximum_expected_phrase_length=max(len(r['expected']) for r in references),
                   preparation_sha256=sha(output / 'preparation.json'),
                   evaluation_references_sha256=sha(output / 'evaluation_references.json'),
                   model_results_available=False, input_visual_review_pending=True)
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['acquire', 'prepare'], required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/evaluation/c3_m_development_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/c3_visual_reading_v1')
    args = parser.parse_args()
    if args.stage == 'acquire':
        acquire(args.source.resolve())
    else:
        prepare(args.source.resolve(), args.output.resolve())
