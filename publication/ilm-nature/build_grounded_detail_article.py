"""Build the manuscript with the complete grounded visual answers and Figure 38."""
import argparse
import fcntl

import build_shared_layout_article as prior
paper = prior.paper
from scripts.diagnose_visual_context_gradients import require

base, previous = paper.base, paper.previous
work = paper.ROOT/'artifacts/grounded_detail_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '09b89920'
paper.SOURCE = paper.ROOT/'references/evidence/grounded_detail_writing_20261118'
paper.FIG, paper.DECK = 'Figure_38_grounded_detail', 'Grounded_detail.pptx'
base.FIGURES.append((paper.FIG, 'grounded_detail_caption.tex', 'fig:grounded-detail'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_shared_layout_article.py', 'build_grounded_detail_article.py')


def verify_table(cells):
    scores = base.load(paper.SOURCE/'qa.json')['scores']
    expected = [[label,str(scores[k]['count']),str(scores[k]['source_exact']),str(scores[k]['detail_exact'])]
                for k,label in [('normal','Intact question'),('exchange','Exchanged question'),
                                ('blank_question','Blank question'),('blank_passage','Blank passage')]]
    require(cells[1:] == expected, 'Native table differs from measured images')


if __name__ == '__main__':
    actions = dict(baseline=previous.baseline, figure=lambda: paper.figure('grounded_detail_figure.mjs'),
                   build=previous.build, inspect=base.inspect,
                   promote=lambda: paper.promote(verify_table, 'grounded_detail_article_build.json'),
                   captions=lambda: (paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
