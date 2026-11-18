"""Compile the native-language and article-title results with verified whole-image examples."""
import argparse
import fcntl
import build_grounded_detail_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/visual_title_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '0ffed538'
paper.SOURCE = paper.ROOT/'references/evidence/visual_titles_20261118'
paper.FIG, paper.DECK = 'Figure_39_visual_titles', 'Visual_titles.pptx'
base.FIGURES.append((paper.FIG,'visual_title_caption.tex','fig:visual-titles'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_grounded_detail_article.py','build_visual_title_article.py')


def verify_table(cells):
    scores=base.load(paper.SOURCE/'result.json')['scores']
    expected=[[label,str(scores[k]['n']),*[str(scores[k][arm]['exact']) for arm in ('normal','contiguous','exchange')]]
              for k,label in [('contiguous','One span'),('distributed','Distributed'),('missing_glyph','Absent glyph')]]
    require(cells[1:]==expected,'Editable table differs from archived outputs')


if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('visual_title_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'visual_title_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
