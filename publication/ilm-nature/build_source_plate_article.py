"""Publish the source-grounded historical form comparison."""
import argparse
import fcntl
import build_writer_transfer_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/source_plate_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '40ac4747'
paper.SOURCE = paper.ROOT/'references/evidence/historical_plates_20261118'
paper.FIG, paper.DECK = 'Figure_42_historical_plates', 'Historical_plates.pptx'
base.FIGURES.append((paper.FIG,'source_plate_caption.tex','fig:historical-plates'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_writer_transfer_article.py','build_source_plate_article.py')


def verify_table(cells):
    spans = base.load(paper.SOURCE/'spans.json')['arms']
    expected = []
    for key, label in [('intact','Original frame'),('paraphrase','Second frame'),('third','Third frame'),('relocated','Pictures moved')]:
        arm = spans[key]
        expected.append([label, *[str(arm['groups'][k]) for k in ('zh_1','zh_2','en_1','en_2')], str(arm['exact'])])
    require(cells[1:]==expected, 'Historical table differs from the archived outputs')


if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('source_plate_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'source_plate_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
