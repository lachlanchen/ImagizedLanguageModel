"""Publish the matched passage diversity comparison."""
import argparse
import fcntl
import build_expression_preference_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/passage_diversity_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = 'c9aa2f77'
paper.SOURCE = paper.ROOT/'references/evidence/passage_diversity_20261118'
paper.FIG, paper.DECK = 'Figure_44_passage_diversity', 'Passage_diversity.pptx'
base.FIGURES.append((paper.FIG,'passage_diversity_caption.tex','fig:passage-diversity'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_expression_preference_article.py','build_passage_diversity_article.py')


def verify_table(cells):
    evidence = base.load(paper.SOURCE/'figure.json')
    require(cells == evidence['table'], 'Editable table differs from the archived physical counts')


if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('passage_diversity_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'passage_diversity_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
