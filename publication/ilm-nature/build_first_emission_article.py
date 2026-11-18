"""Publish the first-glyph learning study."""
import argparse
import hashlib
import os
from PIL import Image
import fcntl
from xml.etree import ElementTree
from zipfile import ZipFile
import build_sentence_choice_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/first_emission_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = 'ce5e6c3c'
paper.SOURCE = paper.ROOT/'references/evidence/first_emission_20261118'
paper.FIG, paper.DECK = 'Figure_56_first_emission', 'First_emission.pptx'
base.FIGURES.append((paper.FIG,'first_emission_caption.tex','fig:first-emission'))
OLD = 'Figure_54_conditional_ink'
NEW = OLD + '_aligned'
base.FIGURES[:] = [(NEW if name == OLD else name, caption, label) for name, caption, label in base.FIGURES]
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_sentence_choice_article.py','build_first_emission_article.py')


def aligned_figure():
    paper.FIG, paper.DECK = NEW, 'Conditional_ink_aligned.pptx'
    os.environ['ILM_ALIGNED_EXCHANGE'] = '1'
    paper.figure('conditional_ink_figure.mjs')
    ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with ZipFile(paper.HERE/'figures'/paper.DECK) as deck:
        chart = next(n for n in deck.namelist() if '/charts/chart' in n and n.endswith('.xml'))
        root = ElementTree.fromstring(deck.read(chart))
        values = [[int(v.text) for v in series.findall('c:val//c:pt/c:v',ns)] for series in root.findall('.//c:ser',ns)]
        require(values == base.load(paper.SOURCE/'corrected_chart.json'), 'Unaligned historical exchange chart')
        require(any(n.endswith('.xlsx') for n in deck.namelist()), 'Corrected chart lacks workbook')


def inspect():
    with Image.open(paper.HERE/'figures'/(OLD+'.png')) as old:
        base.inspect(replaced_rgb_hashes=[hashlib.sha256(old.convert('RGB').tobytes()).hexdigest()])


def verify_table(cells):
    evidence = base.load(paper.SOURCE/'figure.json')
    require(cells == evidence['table'], 'Editable table differs from the archived physical counts')

    expected = evidence['exchange_counts']
    namespace = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with ZipFile(paper.HERE / 'figures' / paper.DECK) as package:
        charts = sorted(n for n in package.namelist() if '/charts/chart' in n and n.endswith('.xml'))
        values = []
        for chart in charts:
            xml = ElementTree.fromstring(package.read(chart))
            for series in xml.findall('.//c:ser', namespace):
                values.append([float(v.text) for v in series.findall('c:val//c:pt/c:v', namespace)])
        workbooks = [n for n in package.namelist() if n.endswith('.xlsx')]
        require(values == expected and len(workbooks) == 1, 'Editable chart differs from physical phrase accuracy')


if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('first_emission_figure.mjs'),build=previous.build,
        aligned=aligned_figure,inspect=inspect,promote=lambda:paper.promote(verify_table,'first_emission_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
