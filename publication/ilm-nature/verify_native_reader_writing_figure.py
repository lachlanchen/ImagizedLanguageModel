#!/usr/bin/env python3
"""Verify native chart snapshots and original glyph pixels across PPT/PDF."""
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import fitz
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STAGE=ROOT/'artifacts/native_reader_writing_publication_v2'
FIGURE=STAGE/'figure_v5'
NS=dict(c='http://schemas.openxmlformats.org/drawingml/2006/chart',
    a='http://schemas.openxmlformats.org/drawingml/2006/main',
    p='http://schemas.openxmlformats.org/presentationml/2006/main',
    r='http://schemas.openxmlformats.org/officeDocument/2006/relationships')


def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return digest(Path(path).read_bytes())
def load(path):return json.loads(Path(path).read_text())


def relationships(archive,part):
    p=PurePosixPath(part);name=str(p.parent/'_rels'/(p.name+'.rels'))
    rows=ET.fromstring(archive.read(name))
    return {x.attrib['Id']:(x.attrib['Target'].lstrip('/') if x.attrib['Target'].startswith('/')
        else posixpath.normpath(str(p.parent/x.attrib['Target']))) for x in rows}


def cell_values(book,formula):
    sheet,area=formula.split('!',1);sheet=sheet.strip("'").replace("''", "'")
    a,b,c,d=range_boundaries(area)
    return [cell.value for row in book[sheet].iter_rows(min_col=a,max_col=c,min_row=b,max_row=d) for cell in row]


def cache_values(element,kind):
    points=element.findall('.//c:pt',NS)
    values=[p.find('c:v',NS).text for p in sorted(points,key=lambda p:int(p.attrib['idx']))]
    return [float(x) for x in values] if kind=='number' else values


def pdf_pixels(path,expected):
    doc=fitz.open(path);matches=[]
    for index,page in enumerate(doc):
        observed=Counter()
        for info in page.get_image_info(xrefs=True):
            if info['width']!=32 or info['height']!=32 or info['xref']==0:continue
            pix=fitz.Pixmap(doc,info['xref'])
            mode='L' if pix.n==1 else 'RGBA' if pix.n==4 else 'RGB'
            gray=Image.frombytes(mode,(pix.width,pix.height),pix.samples).convert('L')
            observed[digest(gray.tobytes())]+=1
        if all(observed[k]>=n for k,n in expected.items()):matches.append(index+1)
    if len(matches)!=1:raise ValueError(f'Exactly one figure page must preserve all12original tile occurrences: {path}: {matches}')
    return dict(pages=len(doc),figure_page=matches[0],original_tile_occurrences_verified=12,sha256=sha(path))


def main():
    bindings=load(FIGURE/'figure_bindings.json')
    ppt=HERE/'figures/Native_reader_writing_figures.pptx'
    charts=[];images=[]
    with ZipFile(ppt) as archive:
        part='ppt/slides/slide1.xml';slide=ET.fromstring(archive.read(part));rels=relationships(archive,part)
        nodes=slide.findall('.//c:chart',NS)
        if len(nodes)!=4:raise ValueError('Four editable native charts required')
        for node,expected in zip(nodes,bindings['charts']):
            chart_part=rels[node.attrib['{'+NS['r']+'}id']]
            xml=ET.fromstring(archive.read(chart_part));series=xml.findall('.//c:ser',NS)
            external=xml.find('c:externalData',NS)
            if external is None:raise ValueError('Native workbook snapshot missing')
            workbook_part=relationships(archive,chart_part)[external.attrib['{'+NS['r']+'}id']]
            book=load_workbook(BytesIO(archive.read(workbook_part)),data_only=True)
            if len(series)!=len(expected['series']):raise ValueError('Chart series coverage changed')
            for actual,source in zip(series,expected['series']):
                name=actual.find('c:tx/c:v',NS).text
                categories=cache_values(actual.find('c:cat',NS),'string')
                values=cache_values(actual.find('c:val',NS),'number')
                if (name!=source['name'] or categories!=expected['categories'] or values!=source['values']):
                    raise ValueError('Native chart cache differs from declared source')
                category_formula=actual.find('c:cat/c:strRef/c:f',NS).text
                value_formula=actual.find('c:val/c:numRef/c:f',NS).text
                if cell_values(book,category_formula)!=categories or cell_values(book,value_formula)!=values:
                    raise ValueError('Workbook literal cells differ from native chart cache')
                sheet,area=value_formula.split('!',1)
                column,first_row,_,_=range_boundaries(area)
                sheet=sheet.strip("'").replace("''", "'")
                if book[sheet].cell(row=first_row-1,column=column).value!=name:
                    raise ValueError('Workbook series heading differs from native chart')
            title=''.join(xml.find('.//c:title',NS).itertext())
            if 'Mean first-tile' in title:
                fmt=xml.find('.//c:dLbls/c:numFmt',NS)
                if fmt is None or fmt.attrib!={'formatCode':'0.000000','sourceLinked':'0'}:
                    raise ValueError('Physical-F1 labels must retain six displayed decimal places')
            charts.append(dict(part=chart_part,workbook=workbook_part,series=len(series),cache_and_cells_match=True))
        pictures=slide.findall('.//p:pic',NS)
        if len(pictures)!=12:raise ValueError('All12editable original tile objects required')
        for picture,source in zip(pictures,bindings['pictures']):
            blip=picture.find('.//a:blip',NS);media=rels[blip.attrib['{'+NS['r']+'}embed']]
            raw=archive.read(media)
            if digest(raw)!=source['sha256']:raise ValueError('Original PNG bytes changed in native deck')
            image=Image.open(BytesIO(raw)).convert('L')
            if image.size!=(32,32) or digest(image.tobytes())!=source['gray_pixels_sha256']:
                raise ValueError('Original tile pixels changed in native deck')
            transform=picture.find('.//a:xfrm',NS)
            values={**transform.find('a:off',NS).attrib,**transform.find('a:ext',NS).attrib}
            for axis,key in [('x','left'),('y','top'),('cx','width'),('cy','height')]:
                if abs(int(values[axis])-round(source['position'][key]*9525))>1:raise ValueError('Image geometry changed')
            images.append(dict(case=source['case'],kind=source['kind'],PNG_bytes_and_pixels_preserved=True))
    expected=Counter(x['gray_pixels_sha256'] for x in bindings['pictures'])
    paths=[HERE/'figures/Figure_20_native_reader_writing.pdf',STAGE/'reading/manuscript.pdf',
           STAGE/'review/manuscript_review.pdf',HERE/'redlined_manuscript.pdf']
    PDFs={str(p.relative_to(ROOT)):pdf_pixels(p,expected) for p in paths}
    preserved=load(STAGE/'preserved_assets.json')
    for name,expected_sha in preserved['files_sha256'].items():
        if sha(HERE/name)!=expected_sha:raise ValueError('Prior figure/style/supplement asset changed: '+name)
    result=dict(native_charts=charts,original_images=images,PDFs=PDFs,preserved_previous_assets=len(preserved['files_sha256']),
        prior_assets_unchanged=True,PPT_sha256=sha(ppt),figure_source_bindings_sha256=sha(FIGURE/'figure_bindings.json'),
        source_sha256=sha(Path(__file__)),chart_F1_display_decimal_places=6,
        full_precision_F1_retained=bindings['full_precision_mean_first_F1'],native_PowerPoint_application_opened=False,
        structural_and_pixel_verification_only=True,manual_visual_review_required=True)
    (STAGE/'figure_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native_charts=4,original_tiles=12,PDFs=len(paths),preserved_assets=len(preserved['files_sha256']),all_checks_passed=True)))


if __name__=='__main__':main()
