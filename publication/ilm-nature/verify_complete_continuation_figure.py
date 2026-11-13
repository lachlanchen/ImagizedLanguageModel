#!/usr/bin/env python3
"""Check native Figure24 caches/table against the bound measured values."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import posixpath
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
NS={'c':C,'a':A}
S='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
P='http://schemas.openxmlformats.org/package/2006/relationships'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def points(node):
    result=sorted(node.findall('.//c:pt',NS),key=lambda p:int(p.attrib['idx']))
    return [p.find('c:v',NS).text for p in result]


def literal_workbook(package, chart_name, chart):
    """Read the workbook addressed by this chart, without evaluating formulas."""
    base, filename = posixpath.split(chart_name)
    relations = ET.fromstring(package.read(f'{base}/_rels/{filename}.rels'))
    external = chart.find('c:externalData', NS)
    assert external is not None
    relation = next(x for x in relations if x.attrib['Id'] == external.attrib[f'{{{R}}}id'])
    assert relation.attrib['Type'] == f'{R}/package'
    assert relation.attrib.get('TargetMode') != 'External'
    name = posixpath.normpath(posixpath.join(base, relation.attrib['Target']))
    assert name.startswith('ppt/embeddings/') and name.endswith('.xlsx')
    with ZipFile(BytesIO(package.read(name))) as workbook:
        book = ET.fromstring(workbook.read('xl/workbook.xml'))
        sheets = book.findall(f'{{{S}}}sheets/{{{S}}}sheet')
        assert len(sheets) == 1
        sheet = sheets[0]
        rels = ET.fromstring(workbook.read('xl/_rels/workbook.xml.rels'))
        sheet_rel = next(x for x in rels if x.attrib['Id'] == sheet.attrib[f'{{{R}}}id'])
        target = sheet_rel.attrib['Target']
        sheet_name = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        document = ET.fromstring(workbook.read(sheet_name))
        assert not document.findall(f'.//{{{S}}}f'), 'Literal data required'
        cells = {}
        for cell in document.findall(f'.//{{{S}}}c'):
            if cell.attrib.get('t') == 'inlineStr':
                value = ''.join(n.text or '' for n in cell.findall(f'.//{{{S}}}t'))
            else:
                assert cell.attrib.get('t') not in {'s', 'str', 'e', 'b'}
                value = float(cell.find(f'{{{S}}}v').text)
            cells[cell.attrib['r']] = value
    return name, sheet.attrib['name'], cells


def workbook_range(formula, sheet, cells):
    match = re.fullmatch(r"'([^']+)'!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)", formula)
    assert match and match[1] == sheet and match[2] == match[4], formula
    return [cells[f'{match[2]}{row}'] for row in range(int(match[3]), int(match[5]) + 1)]


def verify(deck):
 data=json.loads((HERE/'evidence/complete_continuation_measured_data.json').read_text())
 expected={
  'Complete exact answers / 32':([str(x) if not x else f'{x:,}' for x in data['steps']],
    [data['qa_series'][k]['complete_exact'] for k in ['fixed_cells','proportional']],'0'),
  'Readable complete answers / 32':([str(x) if not x else f'{x:,}' for x in data['steps']],
    [data['qa_series'][k]['readable'] for k in ['fixed_cells','proportional']],'0'),
  'Book tiles above clean ink threshold':(['Initial free','Final free','Final supplied'],
    [[round(x['fraction'],6) for x in data['book_density']]],'0.0%')}
 found={}; checked_workbooks=[]
 with ZipFile(deck) as z:
  names=[n for n in z.namelist() if '/charts/' in n and '/_rels/' not in n and n.endswith('.xml')]
  assert len(names)==3
  for name in names:
   root=ET.fromstring(z.read(name));title=''.join(root.find('.//c:title',NS).itertext())
   assert title in expected and title not in found
   categories,values,fmt=expected[title];series=root.findall('.//c:ser',NS)
   assert len(series)==len(values)
   workbook_name,sheet,cells=literal_workbook(z,name,root)
   expected_names=['Fixed cells','Proportional'] if len(values)==2 else ['Fraction of tiles']
   for item,source_values,series_name in zip(series,values,expected_names):
    assert item.find('c:tx/c:v',NS).text==series_name
    assert points(item.find('c:cat',NS))==categories
    actual=list(map(float,points(item.find('c:val',NS))))
    assert actual==source_values,(title,actual,source_values)
    assert workbook_range(item.find('c:cat/c:strRef/c:f',NS).text,sheet,cells)==categories
    value_formula=item.find('c:val/c:numRef/c:f',NS).text
    assert workbook_range(value_formula,sheet,cells)==source_values
    value_column=re.search(r'!\$([A-Z]+)\$',value_formula)[1]
    assert cells[f'{value_column}1']==series_name
   assert root.findall('.//c:dLbls/c:numFmt',NS)
   assert all(n.attrib['formatCode']==fmt for n in root.findall('.//c:dLbls/c:numFmt',NS))
   found[title]=dict(categories=categories,values=values,label_format=fmt)
   checked_workbooks.append(dict(chart=name,workbook=workbook_name,source_cache_literal_cells_match=True))
  slide=ET.fromstring(z.read('ppt/slides/slide1.xml'));tables=slide.findall('.//a:tbl',NS);assert len(tables)==1
  actual=[[''.join(n.text or '' for n in cell.findall('.//a:t',NS))
           for cell in row.findall('a:tc',NS)] for row in tables[0].findall('a:tr',NS)]
  assert actual==data['final_table'],actual
  workbooks=[n for n in z.namelist() if n.startswith('ppt/embeddings/') and n.endswith('.xlsx')]
  assert len(workbooks)==3
  assert len({x['workbook'] for x in checked_workbooks})==3
 return dict(deck_sha256=sha(deck),measured_data_sha256=sha(HERE/'evidence/complete_continuation_measured_data.json'),
   charts=found,table=actual,native_charts=3,native_tables=1,literal_workbooks=3,
   verified_workbooks=checked_workbooks,
   all_displayed_data_checked=True,source_values_not_illustrative=True,new_model_calls=0,
   office_application_review_claimed=False,active_article_updated=False)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deck',type=Path,
  default=HERE/'figures/Complete_continuation_figures.pptx');p.add_argument('--output',type=Path);a=p.parse_args()
 result=verify(a.deck)
 if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(verified=True,deck_sha256=result['deck_sha256'],charts=3,table=1)))
