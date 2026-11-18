"""Compile the pictured Chinese idiom results and verify the generation examples."""
import argparse
import fcntl
import build_visual_title_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/idiom_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '1685c98b'
paper.SOURCE = paper.ROOT/'references/evidence/idiom_paper_20261118'
paper.FIG, paper.DECK = 'Figure_40_pictured_idioms', 'Pictured_idioms.pptx'
base.FIGURES.append((paper.FIG,'idiom_caption.tex','fig:pictured-idioms'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_visual_title_article.py','build_pictured_idiom_article.py')


def verify_table(cells):
    evidence=base.load(paper.SOURCE/'measurements.json')
    expected=[[row['label'],str(row['exact']),str(row['exchanged']),str(row['distinct'])] for row in evidence['models']]
    require(cells[1:]==expected,'Native table differs from exact whole-image results')



if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('idiom_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'idiom_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
