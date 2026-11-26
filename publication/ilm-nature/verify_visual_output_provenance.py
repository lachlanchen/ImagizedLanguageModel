"""Verify the current reading edition against original answer pixels and model scores."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree
from openpyxl import load_workbook
from PIL import Image


PAPER=Path(__file__).resolve().parent
FIG=PAPER/'figures'
ARCHIVES=PAPER/'evidence'
EVIDENCE=PAPER.parents[1]/'references/evidence'


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def bitmap(blob):
    with Image.open(BytesIO(blob)) as image:
        return image.size,sha256(image.convert('RGB').tobytes()).hexdigest()


def shown(doc,page):
    return Counter(bitmap(doc.extract_image(info['xref'])['image']) for info in page.get_image_info(xrefs=True))


def main():
    folder=EVIDENCE/'book_order_prior_v1'
    charts=read(folder/'figures.json')
    for name,receipt in [('book_order_translations.zip','generation'),('book_order_continuations.zip','prose')]:
        assert hash_file(ARCHIVES/name)==read(folder/(receipt+'.json'))['archive']
    assert read(folder/'meaning.json')['archive']==read(folder/'generation.json')['archive']
    assert hash_file(folder/'grammar_scores.npz')==read(folder/'grammar.json')['score_file']
    assert hash_file(ARCHIVES/'book_order_discourse.npz')==read(folder/'discourse.json')['arrays']
    news=read(folder/'meaning.json')
    assert news['totals']['news']=={'baseline':1,'intact':1,'scrambled':1,'reference':107,'foil':0} and not news['passed']
    raw_origins=read(EVIDENCE/'pictured_path_translation_v1/origins.json')
    plates=[(x['file'],x['items'],x['file']+'.pptx') for x in charts]
    memory=EVIDENCE/'book_page_memory_v1'
    details=read(memory/'figure.json'); outcome=read(memory/'result.json')
    assert hash_file(ARCHIVES/'book_page_words.zip')==details['archive']==outcome['archive']
    assert hash_file(ARCHIVES/'book_page_memory_examples.zip')==details['examples']
    assert read(memory/'audit.json')['output_recounts']==5*outcome['eligible']
    plates.insert(1,(details['file'],details['items'],details['file']+'.pptx'))
    plates.append(('Pictured_path_origins',[r for r in raw_origins['items'] if r['type']=='strip'],'Pictured_source_paths.pptx'))
    pdf=PAPER/'manuscript.pdf'
    assert hash_file(pdf)==hash_file(PAPER/'visual_language_paper.pdf')
    index=[]
    with fitz.open(pdf) as paper:
        assert len(paper)==5, len(paper)
        for page,(filename,items,deck) in zip((0,1,2,4),plates,strict=True):
            expected=Counter()
            for row in items:
                with ZipFile(ARCHIVES/row['archive']) as original:
                    blob=original.read(row['member'])
                assert sha256(blob).hexdigest()==row['sha256']
                expected[bitmap(blob)]+=1
            with fitz.open(FIG/(filename+'.pdf')) as plate,ZipFile(FIG/deck) as slides:
                assert expected==shown(plate,plate[0])==shown(paper,paper[page])
                media={bitmap(slides.read(n)) for n in slides.namelist() if n.startswith('ppt/media/')}
                assert set(expected)<=media
                assert f'Fig. {len(index)+1}' in ' '.join(paper[page].get_text().split())
                if page==0:
                    source=read(folder/'prose.json')['scores']
                    series=[[source[arm+'_'+stage]['bits'] for arm in ('baseline','scrambled','intact')] for stage in ('prior','adapted')]
                    assert series==list(charts[0]['series'].values())
                    bars=[d['rect'].width for d in plate[0].get_drawings() if d['type']=='f']
                    assert len(bars)==6 and all(abs(width-67*value)<.002 for width,value in zip(bars,sum(series,[]),strict=True))
                    chart=etree.fromstring(slides.read('ppt/charts/chart1.xml'))
                    values=[[float(n.text) for n in s.xpath('.//*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]')] for s in chart.xpath('//*[local-name()="ser"]')]
                    assert values==series
                    workbook=load_workbook(BytesIO(slides.read(next(n for n in slides.namelist() if n.endswith('.xlsx')))),read_only=True)
                    rows=list(workbook.active.values)[1:];workbook.close()
                    assert all(abs(r[j+1]-series[j][i])<1e-12 for i,r in enumerate(rows) for j in range(2))
            index.append(dict(page=page+1,file=filename+'.pdf',pdf=hash_file(FIG/(filename+'.pdf')),editable=deck,
                              pptx=hash_file(FIG/deck),original_png_instances=sum(expected.values())))
        assert '66,987' in paper[3].get_text() and '256 pairs' in ' '.join(paper[3].get_text().split())
    result=dict(paper=hash_file(pdf),pages=5,figures=index,page_memory=outcome['archive'],scores={n:hash_file(folder/(n+'.json')) for n in
        ('data','prose','generation','meaning','source_interaction','counts','grammar','distant_context','context_clues','discourse','optical_replay','figures')})
    (ARCHIVES/'visual_output_figure_index.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified four figure panels against their output archives and the editable book-prediction chart.')


if __name__=='__main__':
    main()
