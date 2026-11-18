"""Build and verify the Mandarin word-order figure and manuscript editions."""
import argparse
import fcntl
import hashlib
import io
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import build_full_context_comparison_article as previous
from scripts.diagnose_visual_context_gradients import require, write_new

base = previous.base
HERE, ROOT = base.HERE, base.ROOT
WORK = ROOT / 'artifacts/visual_word_order_publication_v1'
STAGE = WORK / 'article'
FIG = 'Figure_33_visual_word_order'
DECK = 'Visual_word_order.pptx'
SOURCE = ROOT / 'references/evidence/visual_word_order_20261118.json'
previous.WORK = WORK
previous.STAGE = base.STAGE = STAGE
previous.BASELINE = '4794a6a1'
base.FIGURES.append((FIG, 'visual_word_order_caption.tex', 'fig:visual-word-order'))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace('build_full_context_comparison_article.py', 'build_visual_word_order_article.py')


def figure(attempt):
    import numpy as np
    from PIL import Image
    from scripts.prepare_visual_sequence_corpus import Rasterizer
    stage = WORK / f'figure_{attempt:02d}'
    stage.mkdir(exist_ok=False)
    roster = ROOT / 'artifacts/visual_word_order_v1/pairs.json'
    result = base.load(SOURCE)
    require(base.sha(roster) == result['files']['pairs.json'], 'Word-order roster changed')
    selected = base.load(roster)[0]
    archive = ROOT / 'data/evaluation_only/zhoblimp_e8402762/ZhoBLiMP.tar.gz'
    require(base.sha(archive) == result['archive_sha256'], 'Benchmark archive changed')
    with tarfile.open(archive) as entries:
        name = 'ZhoBLiMP/' + selected['paradigm'] + '.jsonl'
        pair, = [row for row in map(json.loads, entries.extractfile(name)) if row['pairID'] == selected['pair_id']]
    renderer = Rasterizer()
    for key in ('good', 'bad'):
        tiles = b''.join(renderer.tile(c) for c in pair['sentence_' + key])
        require(hashlib.sha256(tiles).hexdigest() == selected[key + '_sha256'], 'Prompt drawing changed')
        ink = np.unpackbits(np.frombuffer(tiles, dtype=np.uint8)).reshape(-1,32,32)
        Image.fromarray((1 - np.concatenate(ink, axis=1)) * 255).save(stage / (key + '.png'))
    (stage/'node_modules').symlink_to(previous.NODE/'node_modules', target_is_directory=True)
    module = stage/'visual_word_order_figure.mjs'
    shutil.copy2(HERE/module.name, module)
    base.STAGE = stage
    base.run([str(previous.NODE/'bin/node'), str(module), str(ROOT), str(stage), str(previous.SKILL), sys.executable],
             stage/'export.log', previous.environment(stage), previous.costs())


def promote(attempt):
    from openpyxl import load_workbook
    old = base.preserved()
    inspection = base.load(STAGE/'inspection.json')
    review = base.load(STAGE/'visual_review.json')
    build = base.load(STAGE/'build_receipt.json')
    require(all(base.sha(HERE/n) == h for n,h in build['sources'].items()), 'Compiled source changed')
    deck = HERE/'figures'/DECK
    result = base.load(SOURCE)
    stages = ['0','3447','6893','10339','13785']
    checked = {}
    with zipfile.ZipFile(deck) as package:
        workbook, = [n for n in package.namelist() if n.endswith('.xlsx')]
        sheet = load_workbook(io.BytesIO(package.read(workbook)), read_only=True, data_only=True).active
        rows = list(sheet.values)[1:]
        series = {'Reader': [100 * result['metrics'][s]['macro_accuracy'] for s in stages],
                  'Visual bigram': [100 * result['metrics']['bigram']['macro_accuracy']] * 5,
                  'Chance': [50.] * 5}
        require(len(rows) == 5,'Checkpoint count')
        for index, (name, expected) in enumerate(series.items()):
            actual = [r[2*index+1] for r in rows]
            require(all(abs(a-b)<1e-8 for a,b in zip(actual, expected)), 'Chart values differ')
            require([int(str(r[2*index]).replace(',','')) for r in rows] == list(map(int,stages)), 'Checkpoint labels differ')
            checked[name] = actual
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        table = ET.fromstring(package.read('ppt/slides/slide1.xml')).find('.//a:tbl', ns)
        contents = [[c.find('.//a:t',ns).text for c in r.findall('a:tc',ns)] for r in table.findall('a:tr',ns)]
        require([r[1:] for r in contents[1:]] == [['51.4%', '43.8 to 59.4%'], ['−2.7 pp','−19.6 to +14.7 pp'], ['−5.7 pp','−16.8 to +6.3 pp']], 'Interval table differs')
        media = {hashlib.sha256(package.read(n)).hexdigest() for n in package.namelist() if n.startswith('ppt/media/')}
        sample_hashes = [base.sha(WORK/f'figure_{attempt:02d}'/(s+'.png')) for s in ('good','bad')]
        require(set(sample_hashes) <= media,'One or more input images differ')
    promotions = {}
    for folder, name in base.DOCUMENTS:
        path = STAGE/folder/(name+'.pdf')
        d = inspection['documents'][name]
        require(base.sha(path) == d['pdf_sha256'] == review[name]['pdf_sha256'] == build['documents'][name], 'Edition differs')
        require(review[name]['overview_pages'] == d['pages'] and review[name]['detailed_pages'], 'Missing review')
        promotions[HERE/path.name] = path
    promotions[HERE/'redlined_manuscript.tex'] = STAGE/'redlines/redlined_manuscript/redlined_manuscript.tex'
    receipt = HERE/'evidence/visual_word_order_article_build.json'
    require(not receipt.exists(), 'Preserve original promotion')
    for dst, src in promotions.items():
        require(base.sha(dst) in (old['files_sha256'][dst.name],base.sha(src)), 'Current edition changed')
    for dst, src in promotions.items():
        temp = dst.with_suffix(dst.suffix+'.promoting')
        shutil.copy2(src, temp)
        temp.replace(dst)
    write_new(receipt, dict(promoted=True, baseline_git=previous.BASELINE, cumulative_redline_git=build['redline_baseline'],
              source_result_sha256=base.sha(SOURCE), figure=dict(deck_sha256=base.sha(deck), chart=checked,
              tables=1, workbooks=1, sample_png_sha256=sample_hashes), inspection=inspection, visual_review=review,
              prior_figure_files_verified=len(old['figures_sha256']), office_application_export_verified=False,
              all_root_tex_sha256={p.name:base.sha(p) for p in HERE.glob('*.tex')}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=['baseline','figure','captions','build','inspect','promote'])
    cli.add_argument('--attempt', type=int, default=1)
    args = cli.parse_args()
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(base.host()['available_gib'] >= 48, 'Host memory reserve')
        if args.action == 'baseline':
            previous.baseline()
        elif args.action == 'figure':
            figure(args.attempt)
        elif args.action == 'captions':
            (HERE/'complete_continuation_figures.tex').write_text(base.literal_figures())
        elif args.action == 'build':
            previous.build()
        elif args.action == 'inspect':
            base.inspect()
        else:
            promote(args.attempt)
