#!/usr/bin/env python3
"""Verify native counts, literal workbooks and every original glyph instance."""
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
    data_path = HERE / 'evidence/ranking_completed_figure_data.json'; data = load(data_path)
    for name, digest in data['source_sha256'].items(): assert sha(ROOT / name) == digest
    assert sha(HERE / 'prepare_ranking_completed_figure.py') == data['exporter_sha256']
    binding = load(stage / 'figure_bindings.json'); assert binding['data_sha256'] == sha(data_path)
    source = stage / 'output/Ranking_completed_figures.pptx'; validation = load(stage / 'presentation_validation.json')
    assert validation['finalSha256'] == sha(source)
    assert validation['nativeChartValidation']['passed'] and validation['nativeChartValidation']['detectedChartCount'] == 2
    quantitative = validation['nativeQuantitativeCharts']['report']
    assert quantitative['passed'] and not quantitative['native_application_open_verified']
    assert all(c['cache_reference_ranges_verified'] for row in quantitative['results'] for c in row['charts'])
    expected_samples = [*data['samples'], *data['whitespace_failures']]
    expected_images = [item['images'][role] for item in expected_samples for role in ['source', 'control', 'ranking']]
    with zipfile.ZipFile(source) as package:
        slide = etree.fromstring(package.read('ppt/slides/slide1.xml'))
        rels = etree.fromstring(package.read('ppt/slides/_rels/slide1.xml.rels'))
        relationships = {row.get('Id'): row.get('Target') for row in rels}
        pictures = slide.findall('.//p:pic', NS); assert len(pictures) == len(expected_images) == 27
        for picture, expected in zip(pictures, expected_images):
            part = picture.find('.//a:blip', NS).get('{'+NS['r']+'}embed')
            target = posixpath.normpath(posixpath.join('ppt/slides', relationships[part])).lstrip('/')
            original = ROOT / expected['file']; assert sha(original) == expected['sha256']
            assert rgb(package.read(target)) == rgb(original.read_bytes())
        tables = slide.findall('.//a:tbl', NS); assert len(tables) == 1
        values = [[ ''.join(cell.xpath('.//a:t/text()', namespaces=NS)) for cell in row.findall('a:tc', NS)]
                  for row in tables[0].findall('a:tr', NS)]
        assert values == binding['tableValues']
        parts = sorted(n for n in package.namelist() if re.search(r'/charts/chart\d+\.xml$', n))
        assert len(parts) == 2
        for part, expected in zip(parts, binding['charts']):
            chart = etree.fromstring(package.read(part))
            title = ''.join(chart.xpath('.//c:title//a:t/text()', namespaces=NS))
            assert title == expected['title']
            series = chart.findall('.//c:barChart/c:ser', NS); assert len(series) == 2
            for actual, planned in zip(series, expected['series']):
                numbers = [float(x) for x in actual.xpath('./c:val//c:v/text()', namespaces=NS)]
                assert numbers == planned['values']
            workbook = next(x['workbook_part'] for x in validation['chartDataPackaging']['converted_charts'] if x['chart_part'] == part)
            book = openpyxl.load_workbook(io.BytesIO(package.read(workbook)), read_only=True, data_only=True)
            cells = [v for sheet in book for row in sheet.iter_rows(values_only=True) for v in row if v is not None]
            numbers = [v for v in cells if type(v) in [int, float]]
            assert sorted(numbers) == sorted(x for s in expected['series'] for x in s['values'])
            assert all(s['name'] in cells for s in expected['series']); book.close()
    result = dict(data_sha256=sha(data_path), builder_sha256=sha(HERE / 'ranking_completed_figures.mjs'),
        verifier_sha256=sha(Path(__file__)), source_sha256=data['source_sha256'],
        figure_sha256=sha(stage / 'final-reimport.png'), pptx_sha256=sha(source),
        native_charts=2, native_tables=1, native_image_instances=27, literal_workbooks=2,
        all_chart_and_table_values_verified=True, all_source_image_instances_embedded_pixel_exact=True,
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
        for src, name in [(source, 'Ranking_completed_figures.pptx'),
                          (stage / 'final-reimport.png', 'Figure_27_ranking_completed.png')]:
            target = HERE / 'figures' / name; assert not target.exists(); shutil.copy2(src, target)
        out = HERE / 'evidence/ranking_completed_figure_build.json'
        with out.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: result[k] for k in ['native_charts','native_tables','native_image_instances',
        'literal_workbooks','all_chart_and_table_values_verified','verification_seconds']}))


if __name__ == '__main__': main()
