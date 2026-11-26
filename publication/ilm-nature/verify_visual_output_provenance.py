"""Verify the current reading edition against original answer pixels and model scores."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
import numpy as np
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


def verify_passage_strips(details):
    def array(z, name):
        return np.load(BytesIO(z.read('fiction/'+name+'.npy')), allow_pickle=False)

    def strips(z, name):
        x = array(z, name+'-pixels')
        offsets = array(z, name+'-offsets')
        return [x[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]

    def check(z, member, cells):
        with Image.open(BytesIO(z.read(member))) as im:
            ink = np.asarray(im.convert('L')) < 128
        h, w = ink.shape
        packed = np.packbits(ink.reshape(h//32, 32, w//32, 32).transpose(0, 2, 1, 3).reshape(-1, 1024), axis=1)
        assert np.array_equal(packed[:len(cells)], cells) and not packed[len(cells):].any()

    with ZipFile(ARCHIVES/'learned_passage_words.zip') as z, ZipFile(ARCHIVES/'learned_passage_examples.zip') as small:
        gold = [strips(z, 'gold'+str(j)) for j in range(3)]
        emitted = strips(z, 'intact')
        present = array(z, 'present')
        first = next(i for i in range(len(emitted)) if not present[i] and np.array_equal(gold[0][i], emitted[i]))
        check(small, f'{first}/reference.png', gold[0][first])
        check(small, f'{first}/output.png', emitted[first])
        stream = [strips(z, 'stream'+str(j)) for j in range(3)]
        rows = array(z, 'stream-rows')
        ends = array(z, 'ends')
        j = next(j for j,i in enumerate(rows) if not ends[i,:2].any() and all(np.array_equal(stream[k][j], gold[k][i]) for k in (0,1)))
        idx = int(rows[j])
        assert details['examples'] == [first, idx]
        space = np.zeros((1,128),np.uint8)
        for name, words in [('output',[r[j] for r in stream]), ('reference',[r[idx] for r in gold])]:
            check(small, f'{idx}/{name}.png', np.concatenate((words[0],space,words[1],space,words[2])))


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
    passage=EVIDENCE/'passage_selector_v1'
    learned=read(passage/'figure.json')
    scored=read(passage/'scores.json')
    assert hash_file(ARCHIVES/'learned_passage_words.zip')==learned['archive']==scored['archive']
    assert hash_file(ARCHIVES/'learned_passage_examples.zip')==learned['clips']
    verify_passage_strips(learned)
    plates.insert(0,(learned['file'],learned['items'],learned['file']+'.pptx'))
    plates.insert(1,(details['file'],details['items'],details['file']+'.pptx'))
    plates.append(('Pictured_path_origins',[r for r in raw_origins['items'] if r['type']=='strip'],'Pictured_source_paths.pptx'))
    pdf=PAPER/'manuscript.pdf'
    assert hash_file(pdf)==hash_file(PAPER/'visual_language_paper.pdf')
    index=[]
    with fitz.open(pdf) as paper:
        assert len(paper)==6, len(paper)
        for page,(filename,items,deck) in zip((0,1,2,3,5),plates,strict=True):
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
                if page==2:
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
                if page==0:
                    series=[[100*scored['results'][k]['strata']['all']['correct'][arm]/scored['results'][k]['strata']['all']['n'] for k in ('fiction','gutenberg','lambada')] for arm in ('salience','local_counts','intact')]
                    assert series==list(learned['series'].values())
                    bars=[d['rect'].width for d in plate[0].get_drawings() if d['type']=='f']
                    assert len(bars)==9 and all(abs(x-18*y)<.002 for x,y in zip(bars,[s[j] for j in range(3) for s in series],strict=True))
                    chart=etree.fromstring(slides.read('ppt/charts/chart1.xml'))
                    vals=[[float(v.text) for v in r.xpath('.//*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]')] for r in chart.xpath('//*[local-name()="ser"]')]
                    assert all(abs(x-y)<1e-12 for r,t in zip(vals,series,strict=True) for x,y in zip(r,t,strict=True))
                    book=load_workbook(BytesIO(slides.read(next(n for n in slides.namelist() if n.endswith('.xlsx')))),read_only=True)
                    data=list(book.active.values)[1:];book.close()
                    assert all(abs(r[k+1]-series[k][j])<1e-12 for j,r in enumerate(data) for k in range(3))
            index.append(dict(page=page+1,file=filename+'.pdf',pdf=hash_file(FIG/(filename+'.pdf')),editable=deck,
                              pptx=hash_file(FIG/deck),original_png_instances=sum(expected.values())))
        assert '66,987' in paper[4].get_text() and '256 pairs' in ' '.join(paper[4].get_text().split())
    result=dict(paper=hash_file(pdf),pages=6,figures=index,passage_words=scored['archive'],page_memory=outcome['archive'],scores={n:hash_file(folder/(n+'.json')) for n in
        ('data','prose','generation','meaning','source_interaction','counts','grammar','distant_context','context_clues','discourse','optical_replay','figures')})
    (ARCHIVES/'visual_output_figure_index.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified five figure panels against their output archives and both editable prediction charts.')


if __name__=='__main__':
    main()
