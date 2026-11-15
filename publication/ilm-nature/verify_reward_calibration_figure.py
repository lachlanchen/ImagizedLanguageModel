#!/usr/bin/env python3
"""Verify native chart/workbook/table and all unchanged original answer PNGs."""
import argparse
import io
import json
from pathlib import Path
import posixpath
import re
import shutil
import zipfile
from lxml import etree
import openpyxl
from prepare_glyph_score_gap_figure import ROOT, HERE, sha, load, owners
from build_ranking_completed_article import model_sources

NS = dict(a='http://schemas.openxmlformats.org/drawingml/2006/main',
          p='http://schemas.openxmlformats.org/presentationml/2006/main',
          c='http://schemas.openxmlformats.org/drawingml/2006/chart',
          r='http://schemas.openxmlformats.org/officeDocument/2006/relationships')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', type=Path)
    parser.add_argument('--reviewed', action='store_true'); parser.add_argument('--promote', action='store_true')
    args = parser.parse_args(); stage = args.stage.resolve(); protected = owners(); model_sources()
    data_path = HERE / 'evidence/reward_calibration_figure_data.json'; data = load(data_path)
    for n, d in data['source_sha256'].items(): assert sha(ROOT / n) == d
    assert sha(HERE / 'prepare_reward_calibration_figure.py') == data['exporter_sha256']
    assert sha(HERE / 'reward_calibration_figures.mjs') == sha(stage / 'reward_calibration_figures.mjs')
    binding = load(stage / 'figure_bindings.json'); assert binding['data_sha256'] == sha(data_path)
    assert binding['charts'][0]['categories'] == data['categories']
    assert binding['charts'][0]['series'] == [dict(name=r['name'], values=r['values']) for r in data['extrema']]
    assert binding['tableValues'] == [['Frozen review group','Cases','Accepted'],
        *[[r['label'],str(r['cases']),str(r['selective_accepted'])] for r in data['groups']]]
    source = stage / 'output/Reward_calibration_figures.pptx'; validation = load(stage / 'presentation_validation.json')
    assert sha(source) == validation['finalSha256'] and validation['nativeChartValidation']['passed']
    quant = validation['nativeQuantitativeCharts']['report']; assert quant['passed'] and not quant['native_application_open_verified']
    assert all(c['cache_reference_ranges_verified'] for r in quant['results'] for c in r['charts'])
    with zipfile.ZipFile(source) as package:
        slide = etree.fromstring(package.read('ppt/slides/slide1.xml'))
        rels = {r.get('Id'):r.get('Target') for r in etree.fromstring(package.read('ppt/slides/_rels/slide1.xml.rels'))}
        pictures = slide.findall('.//p:pic', NS); assert len(pictures) == 3
        embedded = []
        for pic in pictures:
            rid = pic.find('.//a:blip', NS).get('{'+NS['r']+'}embed')
            target = rels[rid].lstrip('/') if rels[rid].startswith('/') else posixpath.normpath(posixpath.join('ppt/slides', rels[rid]))
            raw = package.read(target)
            import hashlib
            embedded.append(hashlib.sha256(raw).hexdigest())
        assert sorted(embedded) == sorted(r['sha256'] for r in data['samples'])
        tables = slide.findall('.//a:tbl', NS); assert len(tables) == 1
        table = [[''.join(c.xpath('.//a:t/text()', namespaces=NS)) for c in r.findall('a:tc', NS)] for r in tables[0].findall('a:tr', NS)]
        assert table == binding['tableValues']
        charts = [n for n in package.namelist() if re.search(r'/charts/chart\d+\.xml$', n)]; assert len(charts) == 1
        chart = etree.fromstring(package.read(charts[0])); expected = binding['charts'][0]
        assert ''.join(chart.xpath('.//c:title//a:t/text()', namespaces=NS)) == expected['title']
        for actual, plan in zip(chart.findall('.//c:barChart/c:ser', NS), expected['series']):
            assert actual.xpath('./c:tx//c:v/text()', namespaces=NS) == [plan['name']]
            assert actual.xpath('./c:cat//c:v/text()', namespaces=NS) == expected['categories']
            assert [float(v) for v in actual.xpath('./c:val//c:v/text()', namespaces=NS)] == plan['values']
        workbook = next(r['workbook_part'] for r in validation['chartDataPackaging']['converted_charts'] if r['chart_part'] == charts[0])
        book = openpyxl.load_workbook(io.BytesIO(package.read(workbook)), read_only=True, data_only=True)
        values = [v for sheet in book for row in sheet.iter_rows(values_only=True) for v in row if v is not None]
        assert sorted(v for v in values if type(v) in (int,float)) == sorted(v for r in expected['series'] for v in r['values'])
        assert all(r['name'] in values for r in expected['series']); book.close()
    before = load(ROOT / 'artifacts/reward_calibration_publication_v1/article_attempt01/before_manifest.json')
    assert all(sha(HERE / n) == d for n, d in before['figures_sha256'].items())
    result = dict(data_sha256=sha(data_path), builder_sha256=sha(HERE / 'reward_calibration_figures.mjs'),
        verifier_sha256=sha(__file__), source_sha256=data['source_sha256'], pptx_sha256=sha(source),
        figure_sha256=sha(stage / 'final-reimport.png'), native_charts=1, literal_workbooks=1, native_tables=1,
        original_PNG_instances=3, original_PNG_bytes_preserved=True, all64cases_retained_in_source_data=True,
        all_chart_table_workbook_values_verified=True, prior_figure_assets_preserved=len(before['figures_sha256']),
        complete_final_slide_actually_viewed=args.reviewed, Office_application_export_verified=False,
        new_model_calls=0, new_optimizer_updates=0, original_readings_unchanged=True,
        native_export_costs=load(stage / 'costs.json'))
    for child in result['native_export_costs']['children']:
        child['command']=[s.replace(str(ROOT)+'/', '').replace(str(Path.home()), '~') for s in child['command']]
    model_sources(); assert owners() == protected
    if args.promote:
        assert args.reviewed
        for src,name in [(source,'Reward_calibration_figures.pptx'),(stage/'final-reimport.png','Figure_29_reward_calibration.png')]:
            target=HERE/'figures'/name; assert not target.exists(); shutil.copy2(src,target)
        with (HERE/'evidence/reward_calibration_figure_build.json').open('x') as stream:
            json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({k:result[k] for k in ['native_charts','literal_workbooks','native_tables','original_PNG_instances','original_PNG_bytes_preserved','prior_figure_assets_preserved']}))
