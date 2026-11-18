"""Publish the passage-conditioned glyph planner comparison."""
import argparse
import fcntl
from xml.etree import ElementTree
from zipfile import ZipFile
import build_ordered_neighborhood_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/glyph_planner_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '46bf3c2b'
paper.SOURCE = paper.ROOT/'references/evidence/glyph_planner_20261118'
paper.FIG, paper.DECK = 'Figure_46_glyph_planner', 'Glyph_planner.pptx'
base.FIGURES.append((paper.FIG,'glyph_planner_caption.tex','fig:glyph-planner'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_ordered_neighborhood_article.py','build_glyph_planner_article.py')


def verify_table(cells):
    evidence = base.load(paper.SOURCE/'figure.json')
    require(cells == evidence['table'], 'Editable table differs from the archived physical counts')

    main = base.load(paper.SOURCE / 'evaluation.json')['arms']
    search = base.load(paper.SOURCE / 'search.json')['arms']['aligned']
    expected = [main['uniform']['prefix_exact'], main['aligned']['prefix_exact'], search['prefix_exact']]
    namespace = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with ZipFile(paper.HERE / 'figures' / paper.DECK) as package:
        charts = sorted(n for n in package.namelist() if '/charts/chart' in n and n.endswith('.xml'))
        values = []
        for chart in charts:
            xml = ElementTree.fromstring(package.read(chart))
            for series in xml.findall('.//c:ser', namespace):
                values.append([int(v.text) for v in series.findall('c:val//c:pt/c:v', namespace)])
        workbooks = [n for n in package.namelist() if n.endswith('.xlsx')]
        require(values == expected and len(workbooks) == 1, 'Editable curves differ from the three pixel recounts')


if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('glyph_planner_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'glyph_planner_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
