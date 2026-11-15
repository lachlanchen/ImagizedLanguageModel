#!/usr/bin/env python3
"""Verify saved-score native chart, literal workbook and complete table."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import shutil
import time
import zipfile

from lxml import etree
from PIL import Image
import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NS = dict(a='http://schemas.openxmlformats.org/drawingml/2006/main',
          p='http://schemas.openxmlformats.org/presentationml/2006/main',
          c='http://schemas.openxmlformats.org/drawingml/2006/chart',
          r='http://schemas.openxmlformats.org/officeDocument/2006/relationships')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def rgb(raw): return Image.open(io.BytesIO(raw)).convert('RGB').tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', type=Path)
    parser.add_argument('--promote', action='store_true'); parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(); stage = args.stage.resolve(); began = time.perf_counter()
    data_path = HERE / 'evidence/glyph_score_gap_figure_data.json'; data = load(data_path)
    for name, digest in data['source_sha256'].items(): assert sha(ROOT / name) == digest
    assert sha(HERE / 'prepare_glyph_score_gap_figure.py') == data['exporter_sha256']
    binding = load(stage / 'figure_bindings.json'); assert binding['data_sha256'] == sha(data_path)
    original = load(ROOT / 'references/evidence/ranked_glyph_score_gap_result_20261116.json')
    for row in data['groups']:
        assert all(row[k] == v for k, v in original['groups'][row['key']].items())
    expected_table = [['Group', 'Mismatches', 'Source', 'Emitted', 'Ties']] + [
        [row['label'], *[f'{row[k]:,}' for k in ['nonempty_mismatches', 'source_preferred', 'emitted_preferred', 'tied']]]
        for row in data['groups']]
    assert binding['tableValues'] == expected_table
    for row in data['books']:
        source = original['decision']['book_groups'][row['key']]
        assert row['observed_fraction'] == round(source['source_preferred']/source['nonempty_mismatches'], 6)
        assert row['required_fraction'] == round(source['required']/source['nonempty_mismatches'], 6)
    assert binding['charts'][0]['categories'] == [r['label'] for r in data['books']]
    assert binding['charts'][0]['series'] == [
        dict(name='Source preferred', values=[r['observed_fraction'] for r in data['books']]),
        dict(name='Required by book rule', values=[r['required_fraction'] for r in data['books']])]
    source = stage / 'output/Glyph_score_gap_figures.pptx'; validation = load(stage / 'presentation_validation.json')
    assert validation['finalSha256'] == sha(source)
    assert validation['nativeChartValidation']['passed'] and validation['nativeChartValidation']['detectedChartCount'] == 1
    quantitative = validation['nativeQuantitativeCharts']['report']
    assert quantitative['passed'] and not quantitative['native_application_open_verified']
    assert all(c['cache_reference_ranges_verified'] for row in quantitative['results'] for c in row['charts'])
    with zipfile.ZipFile(source) as package:
        slide = etree.fromstring(package.read('ppt/slides/slide1.xml'))
        rels = etree.fromstring(package.read('ppt/slides/_rels/slide1.xml.rels'))
        relationships = {row.get('Id'): row.get('Target') for row in rels}
        assert len(slide.findall('.//p:pic', NS)) == 0
        tables = slide.findall('.//a:tbl', NS); assert len(tables) == 1
        values = [[ ''.join(cell.xpath('.//a:t/text()', namespaces=NS)) for cell in row.findall('a:tc', NS)]
                  for row in tables[0].findall('a:tr', NS)]
        assert values == binding['tableValues']
        parts = sorted(n for n in package.namelist() if re.search(r'/charts/chart\d+\.xml$', n))
        assert len(parts) == 1
        for part, expected in zip(parts, binding['charts']):
            chart = etree.fromstring(package.read(part))
            title = ''.join(chart.xpath('.//c:title//a:t/text()', namespaces=NS))
            assert title == expected['title']
            series = chart.findall('.//c:barChart/c:ser', NS); assert len(series) == 2
            for actual, planned in zip(series, expected['series']):
                assert actual.xpath('./c:tx//c:v/text()', namespaces=NS) == [planned['name']]
                assert actual.xpath('./c:cat//c:v/text()', namespaces=NS) == expected['categories']
                numbers = [float(x) for x in actual.xpath('./c:val//c:v/text()', namespaces=NS)]
                assert numbers == planned['values']
            workbook = next(x['workbook_part'] for x in validation['chartDataPackaging']['converted_charts'] if x['chart_part'] == part)
            book = openpyxl.load_workbook(io.BytesIO(package.read(workbook)), read_only=True, data_only=True)
            cells = [v for sheet in book for row in sheet.iter_rows(values_only=True) for v in row if v is not None]
            numbers = [v for v in cells if type(v) in [int, float]]
            assert sorted(numbers) == sorted(x for s in expected['series'] for x in s['values'])
            assert all(s['name'] in cells for s in expected['series']); book.close()
    result = dict(data_sha256=sha(data_path), builder_sha256=sha(HERE / 'glyph_score_gap_figures.mjs'),
        verifier_sha256=sha(Path(__file__)), source_sha256=data['source_sha256'],
        figure_sha256=sha(stage / 'final-reimport.png'), pptx_sha256=sha(source),
        native_charts=1, native_tables=1, native_image_instances=0, literal_workbooks=1,
        all_chart_and_table_values_verified=True, new_source_images=0, all_seven_group_counts_retained=True,
        complete_final_slide_actually_viewed=args.reviewed, visual_reviewer='Codex' if args.reviewed else None,
        Office_application_export_verified=False, dependency_loader_tool_unavailable=True,
        existing_shared_runtime_reused=True, new_model_calls=0, new_optimizer_updates=0,
        original_readings_unchanged=True,
        native_export_attempts=[load(p) for p in sorted(stage.parent.glob('figure_attempt*/costs.json'))],
        verification_seconds=time.perf_counter() - began)
    if args.promote:
        assert args.reviewed, 'Actual final-slide visual review required'
        registration = load(ROOT / 'references/evidence/white_target_ranking_registration_20261115.json')
        assert all(sha(ROOT / n) == d for n, d in registration['source_sha256'].items())
        for src, name in [(source, 'Glyph_score_gap_figures.pptx'),
                          (stage / 'final-reimport.png', 'Figure_28_glyph_score_gaps.png')]:
            target = HERE / 'figures' / name; assert not target.exists(); shutil.copy2(src, target)
        out = HERE / 'evidence/glyph_score_gap_figure_build.json'
        with out.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: result[k] for k in ['native_charts','native_tables','native_image_instances',
        'literal_workbooks','all_chart_and_table_values_verified','verification_seconds']}))


if __name__ == '__main__': main()
