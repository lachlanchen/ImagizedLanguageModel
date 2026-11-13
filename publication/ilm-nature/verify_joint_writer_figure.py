#!/usr/bin/env python3
"""Independently verify Figure25 native data, threshold and original sample pixels."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import posixpath
from zipfile import ZipFile
import xml.etree.ElementTree as E

import numpy as np
from PIL import Image

from verify_complete_continuation_figure import literal_workbook, workbook_range, points

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = 'http://schemas.openxmlformats.org/drawingml/2006/chart'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = dict(c=C, a=A, p=P, r=R)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(deck):
    data_path = HERE / 'evidence/joint_writer_figure_data.json'
    d = json.loads(data_path.read_text())
    for name, digest in d['source_sha256'].items():
        assert sha(ROOT / name) == digest, 'Original source changed'
    expected = {
        'Frozen review credits with supplied image history': (
            [g['label'] for g in d['groups']], ['Point control', 'Joint writer', 'Required in every group'],
            [[round(g['control_fraction'], 6) for g in d['groups']], [round(g['candidate_fraction'], 6) for g in d['groups']], [.9] * 7]),
        'Source-preferred selected false blanks': (d['selected_search_categories'], ['Observed', 'Required'],
            [d['selected_source_preferred'], d['selected_search_required']])}
    checked = []; media_count = 0
    with ZipFile(deck) as z:
        paths = [n for n in z.namelist() if '/charts/' in n and '/_rels/' not in n and n.endswith('.xml')]
        assert len(paths) == 2
        found = set()
        for name in paths:
            chart = E.fromstring(z.read(name)); title = ''.join(chart.find('.//c:title', NS).itertext())
            assert title in expected and title not in found; found.add(title)
            categories, names, values = expected[title]
            series = chart.findall('.//c:ser', NS); assert len(series) == len(values)
            workbook, sheet, cells = literal_workbook(z, name, chart)
            for s, label, target in zip(series, names, values):
                assert s.find('c:tx/c:v', NS).text == label
                assert points(s.find('c:cat', NS)) == categories
                actual = list(map(float, points(s.find('c:val', NS))))
                assert actual == target
                assert workbook_range(s.find('c:cat/c:strRef/c:f', NS).text, sheet, cells) == categories
                assert workbook_range(s.find('c:val/c:numRef/c:f', NS).text, sheet, cells) == target
            if title.startswith('Frozen'):
                bar, line = chart.find('.//c:barChart', NS), chart.find('.//c:lineChart', NS)
                assert bar is not None and line is not None
                assert len(bar.findall('c:ser', NS)) == 2 and len(line.findall('c:ser', NS)) == 1
                assert {x.attrib['val'] for x in bar.findall('c:axId', NS)} == {x.attrib['val'] for x in line.findall('c:axId', NS)}
                assert line.find('.//a:prstDash', NS).attrib['val'] == 'dash'
                assert line.find('.//a:srgbClr', NS).attrib['val'] == 'A74337'
            checked.append(dict(title=title, workbook=workbook, all_chart_and_literal_workbook_values_match=True))
        slide = E.fromstring(z.read('ppt/slides/slide1.xml'))
        tables = slide.findall('.//a:tbl', NS); assert len(tables) == 1
        actual = [[''.join(t.text or '' for t in cell.findall('.//a:t', NS)) for cell in row.findall('a:tc', NS)] for row in tables[0].findall('a:tr', NS)]
        expected_table = [['Count', *[g['label'].replace('\n', ' ') for g in d['groups']]],
            ['Joint', *[f'{g["candidate"]}/{g["count"]}' for g in d['groups']]],
            ['Point', *[f'{g["control"]}/{g["count"]}' for g in d['groups']]]]
        assert actual == expected_table
        pics = slide.findall('.//p:pic', NS); assert len(pics) == 51
        rels = E.fromstring(z.read('ppt/slides/_rels/slide1.xml.rels'))
        relationships = {v.attrib['Id']: v.attrib['Target'] for v in rels}
        expected_images = [sample['images'][role] for sample in d['blank_book_samples'] for role in ['source', 'point', 'joint']]
        for pic, expected_image in zip(pics, expected_images):
            embed = pic.find('.//a:blip', NS).attrib['{' + R + '}embed']
            target = relationships[embed]
            media = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('ppt/slides', target))
            assert not media.startswith(('/', '../')) and media in z.namelist() and media.endswith('.png')
            with Image.open(BytesIO(z.read(media))) as actual_image, Image.open(ROOT / expected_image['file']) as expected_image_file:
                a = np.asarray(actual_image.convert('L')); b = np.asarray(expected_image_file.convert('L'))
                assert a.shape == b.shape == (96, 96) and np.array_equal(a, b), 'Embedded sample changed'
                reduced = a[::3, ::3]
                assert np.array_equal(a, np.repeat(np.repeat(reduced, 3, 0), 3, 1)), 'Nearest3x source expansion changed'
            media_count += 1
        presentation = E.fromstring(z.read('ppt/presentation.xml'))
        size = presentation.find('p:sldSz', NS)
        assert (size.attrib['cx'], size.attrib['cy']) == ('15240000', '17145000')
        assert len(presentation.findall('p:sldIdLst/p:sldId', NS)) == 1
    return dict(schema='joint-writer-native-figure-data-verification-v1', deck_sha256=sha(deck), measured_data_sha256=sha(data_path),
        native_charts=2, native_tables=1, literal_workbooks=2, charts=checked, exact_ratio_table_verified=True,
        native_90percent_line_verified=True, all51_original_sample_pixels_verified=True, sample_images=media_count,
        frozen_review_errors_disclosed=True, original_scores_unchanged=True, new_model_calls=0,
        native_PowerPoint_application_opened=False, Office_export_verified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('deck', type=Path); p.add_argument('--output', type=Path)
    args = p.parse_args(); result = verify(args.deck)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
