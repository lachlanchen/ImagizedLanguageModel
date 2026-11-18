"""Build and verify Figure 36 and the reading, review and redline articles."""
import argparse
import fcntl
import os
import shutil
import sys
import zipfile
from xml.etree import ElementTree

import build_visual_successor_article as prior
from scripts.diagnose_visual_context_gradients import require, write_new

base, previous = prior.base, prior.previous
HERE, ROOT = base.HERE, base.ROOT
WORK = ROOT/'artifacts/patch_count_publication_v1'
STAGE = WORK/'article'
SOURCE = ROOT/'references/evidence/patch_count_writing_20261118'
FIG = 'Figure_36_patch_writing'
DECK = 'Patch_writing.pptx'
previous.WORK, previous.STAGE, previous.BASELINE = WORK, STAGE, '7b8d76fa'
base.STAGE = STAGE
base.CPU_PUBLICATION_RESERVE_GIB = 32
base.FIGURES.append((FIG, 'patch_count_caption.tex', 'fig:patch-writing'))
old_literal = base.literal_figures
base.literal_figures = lambda: old_literal().replace('build_visual_successor_article.py', 'build_patch_count_article.py')


def figure(module_name='patch_count_figure.mjs'):
    folder = WORK / ('figure' if not (WORK/'figure').exists() else 'figure_retry')
    folder.mkdir(exist_ok=False)
    (folder/'node_modules').symlink_to(previous.NODE/'node_modules', target_is_directory=True)
    module = folder/module_name
    shutil.copy2(HERE/module.name, module)
    base.STAGE = folder
    base.run([str(previous.NODE/'bin/node'), str(module), str(ROOT), str(folder), str(previous.SKILL), sys.executable],
             folder/'export.log', previous.environment(folder), previous.costs())
    for src in (folder/(FIG+'.png'), folder/'output'/DECK):
        dest = HERE/'figures'/src.name
        require(not dest.exists(), 'Preserve existing figure')
        shutil.copy2(src, dest)


def verify_patch_table(cells):
    stats = base.load(SOURCE/'result.json')['stats']
    audit = base.load(SOURCE/'audit.json')
    expected = [['Exact recorded successor', *[str(stats[a]['exact']) for a in ('ordered','rotated')]]]
    expected.extend([[label, *[str(source[a][key]) for a in ('ordered','rotated')]]
                     for label,source,key in [('Known nonempty glyph',audit,'observed_nonempty'),
                                              ('Blank output',stats,'blank_emissions')]])
    require(cells[1:] == expected, 'Measured native table differs')


def promote(verify_table=verify_patch_table, receipt_name='patch_count_article_build.json'):
    before = base.preserved()
    inspection, review, build = (base.load(STAGE/n) for n in ('inspection.json','visual_review.json','build_receipt.json'))
    require(all(base.sha(HERE/n) == h for n,h in build['sources'].items()), 'Compiled TeX changed')
    deck = HERE/'figures'/DECK
    import hashlib
    ns = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(deck) as z:
        hashes = {hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
        require(all(base.sha(p) in hashes for p in SOURCE.glob('*.png')), 'Source pixels differ')
        table = ElementTree.fromstring(z.read('ppt/slides/slide1.xml')).find('.//a:tbl', ns)
        cells = [[c.find('.//a:t', ns).text for c in r.findall('a:tc', ns)] for r in table.findall('a:tr', ns)]
        verify_table(cells)
    promotions = {}
    for folder, name in base.DOCUMENTS:
        path = STAGE/folder/(name+'.pdf')
        doc = inspection['documents'][name]
        require(base.sha(path) == review[name]['pdf_sha256'] == doc['pdf_sha256'] == build['documents'][name],
                'Edition differs')
        require(doc['pages'] == review[name]['overview_pages'] and review[name]['detailed_pages'], 'Review incomplete')
        promotions[HERE/path.name] = path
    promotions[HERE/'redlined_manuscript.tex'] = STAGE/'redlines/redlined_manuscript/redlined_manuscript.tex'
    for dest, src in promotions.items():
        require(base.sha(dest) == before['files_sha256'][dest.name], 'Previous edition changed')
    for dest, src in promotions.items():
        stage = dest.with_suffix(dest.suffix+'.promoting')
        shutil.copy2(src, stage)
        os.replace(stage, dest)
    write_new(HERE/'evidence'/receipt_name, dict(
        promoted=True, baseline_git=previous.BASELINE, cumulative_redline_git=build['redline_baseline'],
        table=cells, deck_sha256=base.sha(deck), source_hashes={str(p.relative_to(SOURCE)):base.sha(p) for p in SOURCE.rglob('*') if p.is_file()},
        inspection=inspection, visual_review=review, prior_figure_files_verified=len(before['figures_sha256']),
        office_application_export_verified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = dict(baseline=previous.baseline, figure=figure, build=previous.build, inspect=base.inspect, promote=promote)
    parser.add_argument('action', choices=[*actions, 'captions'])
    action = parser.parse_args().action
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action == 'captions':
            (HERE/'complete_continuation_figures.tex').write_text(base.literal_figures())
        else:
            actions[action]()
