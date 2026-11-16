#!/usr/bin/env python3
"""Verify native numerical objects, literal workbook and exact original PNG bytes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import shutil
import zipfile
from lxml import etree
import openpyxl

from prepare_visual_context_evidence import ROOT,HERE,STAGE,protected_sources
from prepare_glyph_score_gap_figure import owners
from build_ranking_completed_article import load,sha

NS=dict(a='http://schemas.openxmlformats.org/drawingml/2006/main',p='http://schemas.openxmlformats.org/presentationml/2006/main',
        c='http://schemas.openxmlformats.org/drawingml/2006/chart',r='http://schemas.openxmlformats.org/officeDocument/2006/relationships')

def verify(stage,reviewed,promote):
    protected=owners();sources=protected_sources();data=load(HERE/'evidence/visual_context_completed_figure_data.json')
    assert sha(HERE/'prepare_visual_context_evidence.py')==data['exporter_sha256']
    assert sha(HERE/'visual_context_evidence_figures.mjs')==sha(stage/'visual_context_evidence_figures.mjs')
    b=load(stage/'figure_bindings.json');assert b['data_sha256']==sha(HERE/'evidence/visual_context_completed_figure_data.json')
    validation=load(stage/'presentation_validation.json');ppt=stage/'output/Visual_context_evidence_figures.pptx'
    assert validation['finalSha256']==sha(ppt) and validation['nativeQuantitativeCharts']['report']['passed']
    tables=[];pictures=[]
    with zipfile.ZipFile(ppt) as z:
        for index in (1,2):
            slide=etree.fromstring(z.read(f'ppt/slides/slide{index}.xml'))
            rels={r.get('Id'):r.get('Target') for r in etree.fromstring(z.read(f'ppt/slides/_rels/slide{index}.xml.rels'))}
            for pic in slide.findall('.//p:pic',NS):
                rid=pic.find('.//a:blip',NS).get('{'+NS['r']+'}embed');target=rels[rid]
                target=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('ppt/slides',target))
                pictures.append(hashlib.sha256(z.read(target)).hexdigest())
            for t in slide.findall('.//a:tbl',NS):
                tables.append([[''.join(c.xpath('.//a:t/text()',namespaces=NS)) for c in r.findall('a:tc',NS)] for r in t.findall('a:tr',NS)])
        assert tables==b['tables'] and len(pictures)==11 and sorted(pictures)==sorted(b['original_png_sha256'])
        charts=[n for n in z.namelist() if re.search(r'/charts/chart\d+\.xml$',n)];assert len(charts)==1
        chart=etree.fromstring(z.read(charts[0]));expected=b['charts'][0]
        assert ''.join(chart.xpath('.//c:title//a:t/text()',namespaces=NS))==expected['title']
        actual=chart.findall('.//c:barChart/c:ser',NS);assert len(actual)==3
        for row,plan in zip(actual,expected['series']):
            assert row.xpath('./c:tx//c:v/text()',namespaces=NS)==[plan['name']]
            assert row.xpath('./c:cat//c:v/text()',namespaces=NS)==expected['categories']
            assert [float(x) for x in row.xpath('./c:val//c:v/text()',namespaces=NS)]==plan['values']
        book=next(c['workbook_part'] for c in validation['chartDataPackaging']['converted_charts'] if c['chart_part']==charts[0])
        wb=openpyxl.load_workbook(io.BytesIO(z.read(book)),read_only=True,data_only=True)
        cells=[v for s in wb for row in s.iter_rows(values_only=True) for v in row if v is not None];wb.close()
        assert sorted(v for v in cells if type(v) in (int,float))==sorted(v for s in expected['series'] for v in s['values'])
    before=load(STAGE/'before_manifest.json');assert all(sha(HERE/n)==h for n,h in before['figures_sha256'].items())
    for sample in data['white']['samples']:
        flags=[]
        from PIL import Image
        import numpy as np
        for a in sample['images']:
            tile=(np.array(Image.open(ROOT/a['file']))<128).astype(np.uint8)
            assert hashlib.sha256(tile.tobytes()).hexdigest()==a['original_binary_tile_sha256']
            flags.append(bool(tile.any()))
        expected_flags={'residual_false_ink':(False,None,True),'cleared_white':(False,True,False),'introduced_false_blank':(True,True,False)}[sample['rule']]
        assert all(b is None or a==b for a,b in zip(flags,expected_flags))
    protected_sources();assert owners()==protected
    result=dict(data_sha256=sha(HERE/'evidence/visual_context_completed_figure_data.json'),
        builder_sha256=sha(HERE/'visual_context_evidence_figures.mjs'),verifier_sha256=sha(__file__),
        pptx_sha256=sha(ppt),figures={str(i):sha(stage/f'figure-{i}.png') for i in (30,31)},native_charts=1,
        native_tables=3,literal_workbooks=1,original_PNG_instances=11,original_PNG_bytes_preserved=True,
        outcome_labels_checked_against_original_ink=True,native_values_and_workbook_verified=True,
        prior_figure_assets_preserved=len(before['figures_sha256']),both_complete_slides_actually_viewed=reviewed,
        Office_application_export_verified=False,protected_model_sources=sources,
        new_model_calls=0,new_optimizer_updates=0,original_readings_unchanged=True,costs=load(stage/'costs.json'))
    for child in result['costs']['children']:
        child['command']=[x.replace(str(ROOT)+'/', '').replace(str(Path.home()),'~') for x in child['command']]
    if promote:
        assert reviewed
        for source,name in [(ppt,'Visual_context_evidence_figures.pptx'),(stage/'figure-30.png','Figure_30_white_target_completed.png'),
                            (stage/'figure-31.png','Figure_31_visual_context_evidence.png')]:
            target=HERE/'figures'/name;assert not target.exists();shutil.copy2(source,target)
        with (HERE/'evidence/visual_context_evidence_figure_build.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('native_charts','native_tables','original_PNG_instances','prior_figure_assets_preserved','both_complete_slides_actually_viewed')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('--reviewed',action='store_true');p.add_argument('--promote',action='store_true');a=p.parse_args()
    verify(a.stage,a.reviewed,a.promote)
