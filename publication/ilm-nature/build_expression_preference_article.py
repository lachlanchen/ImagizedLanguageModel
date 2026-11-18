"""Publish the local teacher and physical trajectory preference comparisons."""
import argparse
import fcntl
import build_source_plate_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/expression_preference_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '52bb48c1'
paper.SOURCE = paper.ROOT/'references/evidence/expression_preference_20261118'
paper.FIG, paper.DECK = 'Figure_43_expression_preference', 'Expression_preference.pptx'
base.FIGURES.append((paper.FIG,'expression_preference_caption.tex','fig:expression-preference'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_source_plate_article.py','build_expression_preference_article.py')


def verify_table(cells):
    evidence = base.load(paper.SOURCE/'summary.json')
    require(cells == evidence['table'], 'Editable table differs from the archived physical counts')


if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('expression_preference_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'expression_preference_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
