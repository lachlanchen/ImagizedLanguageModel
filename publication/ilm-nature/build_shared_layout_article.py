"""Build the manuscript with the complete coarse-layout experiment and Figure 37."""
import argparse
import fcntl

import build_patch_count_article as paper
from scripts.diagnose_visual_context_gradients import require

base, previous = paper.base, paper.previous
work = paper.ROOT/'artifacts/shared_layout_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = 'c82d13f6'
paper.SOURCE = paper.ROOT/'references/evidence/shared_layout_writing_20261118'
paper.FIG, paper.DECK = 'Figure_37_shared_layout', 'Shared_layout.pptx'
base.FIGURES.append((paper.FIG, 'shared_layout_caption.tex', 'fig:shared-layout'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_patch_count_article.py', 'build_shared_layout_article.py')


def verify_table(cells):
    arm = base.load(paper.SOURCE/'result.json')['arms']['ordered']
    expected = [[label, str(arm['previous'][key]), str(arm['free'][key])] for key, label in
                [('known_nonempty','Known nonempty glyph'), ('blank','Blank'), ('exact','Exact successor')]]
    require(cells[1:] == expected, 'Native table differs from measured images')


if __name__ == '__main__':
    actions = dict(baseline=previous.baseline, figure=lambda: paper.figure('shared_layout_figure.mjs'),
                   build=previous.build, inspect=base.inspect,
                   promote=lambda: paper.promote(verify_table, 'shared_layout_article_build.json'),
                   captions=lambda: (paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
