"""Publish physical writing and extractive reading from measured PNGs."""
import argparse
import fcntl
import hashlib
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import build_visual_word_order_article as prior
from scripts.diagnose_visual_context_gradients import require, write_new

previous, base = prior.previous, prior.base
ROOT, HERE = base.ROOT, base.HERE
WORK = ROOT/'artifacts/visual_reading_writing_publication_v1'
STAGE = WORK/'article'
FIG = 'Figure_34_visual_reading_writing'
DECK = 'Visual_reading_writing.pptx'
Y = ROOT/'references/evidence/native_yan_cold_20261118'
X = ROOT/'references/evidence/visual_extractive_reading_20261118'
previous.WORK = WORK
previous.STAGE = base.STAGE = STAGE
previous.BASELINE = '50c3c001'
base.FIGURES.append((FIG,'visual_reading_writing_caption.tex','fig:visual-reading-writing'))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace('build_visual_word_order_article.py', 'build_visual_reading_writing_article.py')


def figure(attempt):
    folder = WORK/f'figure_{attempt:02d}'
    folder.mkdir(exist_ok=False)
    (folder/'node_modules').symlink_to(previous.NODE/'node_modules',target_is_directory=True)
    source = HERE/'visual_reading_writing_figure.mjs'
    shutil.copy2(source,folder/source.name)
    base.STAGE = folder
    base.run([str(previous.NODE/'bin/node'),str(folder/source.name),str(ROOT),str(folder),str(previous.SKILL),sys.executable],
             folder/'export.log',previous.environment(folder),previous.costs())


def promote():
    before = base.preserved()
    inspection,review,build = (base.load(STAGE/n) for n in ('inspection.json','visual_review.json','build_receipt.json'))
    require(all(base.sha(HERE/n)==v for n,v in build['sources'].items()),'Compiled inputs changed')
    deck = HERE/'figures'/DECK
    pictures = [Y/(n+'.png') for n in ('prompt','greedy_answer','sampled_answer')]
    pictures += [X/(n+'.png') for n in ('first_exact_question','first_exact_answer','first_case_question','first_case_answer','first_case_reference')]
    result=base.load(X/'result.json')['comparisons']
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(deck) as archive:
        media={hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('ppt/media/')}
        require(all(base.sha(p) in media for p in pictures),'Changed original image')
        table=ET.fromstring(archive.read('ppt/slides/slide1.xml')).find('.//a:tbl',ns)
        values=[[c.find('.//a:t',ns).text for c in r.findall('a:tc',ns)] for r in table.findall('a:tr',ns)]
        for row,key in zip(values[1:],('blank','swapped'),strict=True):
            comp=result[key]
            require(row[1:] == [f"{comp['normal']} / {comp['questions']:,}",f"{comp['control']} / {comp['questions']:,}"],'Table differs')
    promotions={}
    for folder,name in base.DOCUMENTS:
        src=STAGE/folder/(name+'.pdf')
        d=inspection['documents'][name]
        require(base.sha(src)==d['pdf_sha256']==review[name]['pdf_sha256']==build['documents'][name], 'Edition mismatch')
        require(d['pages']==review[name]['overview_pages'] and review[name]['detailed_pages'],'Missing visual inspection')
        promotions[HERE/src.name]=src
    promotions[HERE/'redlined_manuscript.tex']=STAGE/'redlines/redlined_manuscript/redlined_manuscript.tex'
    receipt=HERE/'evidence/visual_reading_writing_article_build.json'
    require(not receipt.exists(),'Keep existing edition')
    for dst,src in promotions.items():
        require(base.sha(dst) in (before['files_sha256'][dst.name],base.sha(src)),'Externally modified edition')
    for dst,src in promotions.items():
        tmp=dst.with_suffix(dst.suffix+'.promoting')
        shutil.copy2(src,tmp)
        tmp.replace(dst)
    write_new(receipt,dict(promoted=True,baseline_git=previous.BASELINE,cumulative_redline_git=build['redline_baseline'],
            source_results={p.name:base.sha(p) for p in (X/'result.json',Y/'code_cycle.json')},
            deck_sha256=base.sha(deck),native_table=values,original_images={p.name:base.sha(p) for p in pictures},
            inspection=inspection,visual_review=review,prior_figure_files_verified=len(before['figures_sha256']),
            office_application_export_verified=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('baseline','figure','captions','build','inspect','promote'))
    cli.add_argument('--attempt',type=int,default=1)
    args=cli.parse_args()
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(base.host()['available_gib']>=48,'Host RAM entry reserve')
        if args.action=='baseline':
            previous.baseline()
        elif args.action=='figure':
            figure(args.attempt)
        elif args.action=='captions':
            (HERE/'complete_continuation_figures.tex').write_text(base.literal_figures())
        elif args.action=='build':
            previous.build()
        elif args.action=='inspect':
            base.inspect()
        else:
            promote()
