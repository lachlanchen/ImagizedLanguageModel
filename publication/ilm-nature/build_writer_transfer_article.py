"""Publish the matched visual writer comparison and its held-out group confirmation."""
import argparse
import fcntl
import build_pictured_idiom_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT/'artifacts/writer_transfer_publication_v1'
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work/'article'
previous.BASELINE = '19ebb535'
paper.SOURCE = paper.ROOT/'references/evidence/writer_transfer_20261118'
paper.FIG, paper.DECK = 'Figure_41_writer_transfer', 'Writer_transfer.pptx'
base.FIGURES.append((paper.FIG,'writer_transfer_caption.tex','fig:writer-transfer'))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace('build_pictured_idiom_article.py','build_writer_transfer_article.py')


def verify_table(cells):
    result = base.load(paper.SOURCE/'result.json')
    expected = []
    for name, label in [('published','Initial writer'), ('original_targets','Original targets'), ('distilled_targets','Reader-guided targets')]:
        dev = result['development']['arms'][name]
        fresh = result['fresh']['arms'][name]
        expected.append([label, str(dev['exact']), str(dev['reader_correct']), str(fresh['exact']), str(fresh['reader_correct'])])
    require(cells[1:]==expected,'Editable table differs from the complete image counts')



if __name__=='__main__':
    actions=dict(baseline=previous.baseline,figure=lambda:paper.figure('writer_transfer_figure.mjs'),build=previous.build,
        inspect=base.inspect,promote=lambda:paper.promote(verify_table,'writer_transfer_article_build.json'),
        captions=lambda:(paper.HERE/'complete_continuation_figures.tex').write_text(base.literal_figures()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    with (paper.ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
