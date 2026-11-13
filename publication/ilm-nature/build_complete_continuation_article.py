#!/usr/bin/env python3
"""Serial CPU article builds alongside the unchanged, separately owned fit.

Builds remain private until a source-bound visual review explicitly promotes
them. This program neither imports a model library nor acquires its GPU lock.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT / 'artifacts/complete_continuation_publication_v1/active_article_v1'
DOCUMENTS = [('reading', 'manuscript'), ('review', 'manuscript_review'),
             ('redlines/redlined_manuscript', 'redlined_manuscript')]
FIGURES = [
    ('Figure_23_continuation_history_completed', 'continuation_history_caption.tex',
     'fig:continuation-history'),
    ('Figure_24_complete_continuation', 'complete_continuation_caption.tex',
     'fig:complete-continuation'),
    ('Figure_25_joint_writer_physical', 'joint_writer_caption.tex',
     'fig:joint-writer-physical'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def host():
    memory = {r.split(':')[0]: int(r.split()[1])
              for r in Path('/proc/meminfo').read_text().splitlines()}
    pressure = {r.split()[0]: float(r.split()[1].split('=')[1])
                for r in Path('/proc/pressure/memory').read_text().splitlines()}
    return dict(available_gib=memory['MemAvailable'] / 1048576, **pressure)


def identity(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    parts = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, state=parts[0], start_ticks=parts[19])


def model_sources():
    source = ROOT / 'references/evidence/joint_context_learning_registration_20261114.json'
    hashes = load(source)['source_sha256']
    assert len(hashes) == 59
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items())
    return sha(source)


def literal_figures():
    parts = ['% Literal captions copied from their source inputs by build_complete_continuation_article.py.',
             '% Native PowerPoint sources and reviewed PNG pixels remain unchanged.']
    for figure, caption, label in FIGURES:
        text = (HERE / caption).read_text().strip()
        if text.startswith('%'):
            text = '\n'.join(line for line in text.splitlines()
                             if not line.startswith('%')).strip()
        if text.startswith(r'\plate{'):
            prefix, suffix = r'\plate{' + figure + '}{', '}{' + label + '}'
            assert text.startswith(prefix) and text.endswith(suffix)
            text = text[len(prefix):-len(suffix)]
        parts += [r'\begin{figure*}[p]', r'\centering',
                  r'\includegraphics[width=.92\textwidth]{figures/' + figure + '.png}',
                  r'\caption{' + text + '}',
                  r'\label{' + label + '}', r'\end{figure*}']
    return '\n'.join(parts) + '\n'


def preserved():
    before = load(STAGE / 'before_manifest.json')
    for name, digest in before['files_sha256'].items():
        assert sha(STAGE / 'before' / name) == digest, name
    for name, digest in before['figures_sha256'].items():
        assert sha(HERE / name) == digest, name
    assert (HERE / 'complete_continuation_figures.tex').read_text() == literal_figures()
    return before


def tree_rss(pid):
    total = 0
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            text = Path(f'/proc/{current}/status').read_text()
            match = re.search(r'^VmRSS:\s+(\d+)', text, re.M)
            total += int(match[1]) if match else 0
            pending += [int(p) for p in Path(
                f'/proc/{current}/task/{current}/children').read_text().split()]
        except FileNotFoundError:
            pass
    return total / 1048576


def run(command, log, env, costs):
    start = time.monotonic()
    assert host()['available_gib'] >= 48
    with log.open('x') as output:
        process = subprocess.Popen(['nice', '-n', '10', *command], cwd=HERE,
                                   env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                observed = host()
                costs['minimum_available_host_gib'] = min(
                    costs['minimum_available_host_gib'], observed['available_gib'])
                costs['peak_child_tree_rss_gib'] = max(
                    costs['peak_child_tree_rss_gib'], tree_rss(process.pid))
                assert observed['available_gib'] >= 48, 'host reserve'
                assert costs['peak_child_tree_rss_gib'] < 2, 'child tree RSS ceiling'
                assert costs['child_seconds'] + time.monotonic() - start < 180, 'build time ceiling'
                time.sleep(.2)
            assert process.returncode == 0, str(log)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            costs['child_seconds'] += time.monotonic() - start
            costs['children'].append(dict(command=command,
                seconds=time.monotonic() - start, returncode=process.returncode,
                log=str(log.relative_to(ROOT))))
            (STAGE / 'child_costs.json').write_text(json.dumps(costs, indent=2) + '\n')


def build():
    preserved()
    registration = model_sources()
    initial = host()
    assert initial['available_gib'] >= 48 and initial['some'] <= .5 and initial['full'] <= .1, initial
    handoff = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']
    protected = [identity(handoff[key]) for key in ['active_controller', 'active_model_worker_pid']]
    assert protected[0]['start_ticks'] == handoff['active_controller_start_ticks']
    assert protected[1]['start_ticks'] == handoff['active_model_worker_start_ticks']
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            cwd = (path / 'cwd').resolve()
            comm = (path / 'comm').read_text().strip()
            assert not (ROOT in (cwd, *cwd.parents) and comm in
                        ['pdflatex', 'latexmk', 'soffice', 'soffice.bin']), 'another publication job'
        except (FileNotFoundError, PermissionError):
            continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               ILM_MOTOR_PAPER_STAGE=str(STAGE), ILM_MOTOR_PAPER_CHECK_ONLY='1')
    costs = dict(child_seconds=0., minimum_available_host_gib=initial['available_gib'],
                 peak_child_tree_rss_gib=0., children=[])
    if (STAGE / 'child_costs.json').exists():
        costs = load(STAGE / 'child_costs.json')
    for folder, name in DOCUMENTS[:2]:
        output = STAGE / folder
        output.mkdir(exist_ok=False)
        run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
             '-outdir=' + str(output), name + '.tex'], output / 'build.log', env, costs)
    run([os.sys.executable, str(HERE / 'build_motor_diagnostics_redline.py')],
        STAGE / 'redline_console.log', env, costs)
    details = {}
    for folder, name in DOCUMENTS:
        output = STAGE / folder
        log = (output / (name + '.log')).read_text()
        bad = [line for line in log.splitlines() if any(token in line for token in
               ['Overfull', 'undefined references', 'undefined citations', 'multiply defined', 'LaTeX Error'])]
        assert not bad, (name, bad)
        details[name] = dict(pdf_sha256=sha(output / (name + '.pdf')),
                             log_sha256=sha(output / (name + '.log')))
    for value in protected:
        after = identity(value['pid'])
        assert after['start_ticks'] == value['start_ticks'] and after['state'] != 'Z'
    model_sources()
    receipt = dict(build_complete=True, promoted=False, visual_review_complete=False,
        source_sha256={p.name: sha(p) for p in HERE.glob('*.tex')},
        builder_sha256=sha(Path(__file__)), documents=details, costs=costs,
        model_registration_sha256=registration, protected_model_processes=protected,
        model_sources_unchanged=True, GPU_lock_acquired=False,
        inherited_baseline_commit=load(STAGE / 'redline_build.json')['baseline_commit'],
        numerical_language_results_changed=False, Office_export_verified=False)
    (STAGE / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(build_complete=True, costs=costs, documents=details), indent=2))


def inspect():
    """Authenticate image contents and render all pages; does not claim review."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    preserved()
    model_sources()
    began = time.monotonic()
    prior_seconds = load(STAGE / 'child_costs.json')['child_seconds']
    def guard():
        assert host()['available_gib'] >= 48
        assert prior_seconds + time.monotonic() - began < 180, 'cumulative build/inspection ceiling'
        assert tree_rss(os.getpid()) < 2, 'inspection RSS ceiling'
    result = dict(documents={}, figure_assets_preserved=230,
                  automated_checks_are_not_visual_review=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for folder, name in DOCUMENTS:
        path = STAGE / folder / (name + '.pdf')
        rendering = STAGE / 'inspection' / name
        rendering.mkdir(parents=True, exist_ok=False)
        with fitz.open(path) as pdf:
            bounds, contents, image_hashes, images = [], [], set(), {}
            for index, page in enumerate(pdf):
                guard()
                contents.append(page.get_text())
                for block in page.get_text('dict')['blocks']:
                    if block['type'] != 0:
                        continue
                    x0, y0, x1, y1 = block['bbox']
                    if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                        bounds.append(dict(page=index + 1, bbox=block['bbox']))
                page.get_pixmap(matrix=fitz.Matrix(.65, .65)).save(rendering / f'page-{index+1:03}.png')
                for row in page.get_images(full=True):
                    xref = row[0]
                    if xref in images:
                        continue
                    image = Image.open(io.BytesIO(pdf.extract_image(xref)['image'])).convert('RGB')
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    image_hashes.add(digest)
                    images[xref] = dict(digest=digest, mask=row[1], size=image.size)
            assert not bounds, (name, bounds)
            figures = {}
            for figure, _, label in FIGURES:
                source = Image.open(HERE / 'figures' / (figure + '.png'))
                digest = hashlib.sha256(source.convert('RGB').tobytes()).hexdigest()
                found = [(xref, row) for xref, row in images.items() if row['digest'] == digest]
                assert found, (name, figure)
                for _, row in found:
                    assert row['size'] == source.size
                    if source.mode == 'RGBA':
                        mask = Image.open(io.BytesIO(pdf.extract_image(row['mask'])['image'])).convert('L')
                        assert mask.tobytes() == source.getchannel('A').tobytes()
                figures[label] = dict(RGB_and_alpha_exact=True,
                    pages=[i + 1 for i, page in enumerate(pdf)
                           if any(row[0] in [v[0] for v in found] for row in page.get_images(full=True))])
            with fitz.open(STAGE / 'before' / (name + '.pdf')) as old:
                old_hashes = set()
                for page in old:
                    guard()
                    for row in page.get_images(full=True):
                        image = Image.open(io.BytesIO(old.extract_image(row[0])['image'])).convert('RGB')
                        old_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
                assert old_hashes <= image_hashes, (name, 'missing earlier image contents')
            text = '\n'.join(contents)
            for anchor in ['970/1,040', '7/32', '17.795', 'softplus', '5,005,760', '7,553,448,370']:
                assert anchor in text, (name, anchor)
            (rendering / 'text.txt').write_text(text)
            # Thirty-six thumbnails per sheet provide a complete layout overview.
            for first in range(0, len(pdf), 36):
                sheet = Image.new('RGB', (6 * 170, 6 * 260), 'white')
                draw = ImageDraw.Draw(sheet)
                for slot, number in enumerate(range(first + 1, min(first + 36, len(pdf)) + 1)):
                    image = Image.open(rendering / f'page-{number:03}.png').convert('RGB')
                    image.thumbnail((165, 235))
                    x, y = (slot % 6) * 170, (slot // 6) * 260
                    sheet.paste(image, (x, y + 20))
                    draw.text((x + 5, y + 2), str(number), fill='black', font=font)
                sheet.save(rendering / f'overview-{first//36+1}.png')
            result['documents'][name] = dict(pages=len(pdf), pdf_sha256=sha(path),
                all_text_bounds_inside_page=True, all_prior_image_contents_retained=len(old_hashes),
                figures=figures, overview_sheets=len(list(rendering.glob('overview-*.png'))))
    result['inspection_seconds'] = time.monotonic() - began
    result['build_plus_inspection_seconds'] = prior_seconds + result['inspection_seconds']
    (STAGE / 'inspection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def promote():
    build = load(STAGE / 'build_receipt.json')
    inspection = load(STAGE / 'inspection.json')
    review = load(STAGE / 'visual_review.json')
    assert review['complete'] and not review['Office_application_export_verified']
    for name, info in inspection['documents'].items():
        seen = review['documents'][name]
        assert seen['pdf_sha256'] == info['pdf_sha256']
        assert seen['overview_pages'] == list(range(1, info['pages'] + 1))
        assert seen['detailed_pages']
    for name, digest in build['source_sha256'].items():
        assert sha(HERE / name) == digest, ('changed compiled input', name)
    preserved()
    model_sources()
    for folder, name in DOCUMENTS:
        source = STAGE / folder / (name + '.pdf')
        assert sha(source) == inspection['documents'][name]['pdf_sha256']
        target = HERE / (name + '.pdf')
        temporary = target.with_suffix('.promoting.pdf')
        shutil.copy2(source, temporary)
        temporary.replace(target)
    source = STAGE / 'redlines/redlined_manuscript/redlined_manuscript.tex'
    shutil.copy2(source, HERE / 'redlined_manuscript.tex')
    build.update(promoted=True, visual_review_complete=True, inspection=inspection,
                 visual_review=review, immediate_baseline=load(STAGE / 'before_manifest.json'),
                 active_article_updated=True, new_optimizer_updates=0, new_model_calls=0,
                 language_of_running_context_fit_unmeasured=True,
                 builder_with_inspection_sha256=sha(Path(__file__)))
    # Live runtime identities belong only in the private build/handoff records.
    build.pop('protected_model_processes')
    build['model_ownership_verified_before_and_after_build'] = True
    for child in build['costs']['children']:
        child['command'] = [('python' if word == os.sys.executable else
                             word.replace(str(ROOT) + '/', ''))
                            for word in child['command']]
    # The generated redline is an output, not an input of the active article.
    build['compiled_previous_redline_source_sha256'] = build['source_sha256'].pop('redlined_manuscript.tex')
    build['new_redline_source_sha256'] = sha(HERE / 'redlined_manuscript.tex')
    (HERE / 'evidence/complete_continuation_article_build.json').write_text(
        json.dumps(build, indent=2) + '\n')
    print('Promoted the reviewed reading, review and cumulative redline PDFs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-figures', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--promote', action='store_true')
    arguments = parser.parse_args()
    if arguments.check_figures:
        preserved()
        print('All 230 previous figure assets and all literal captions match.')
    else:
        with (ROOT / 'artifacts/publication_cpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if arguments.inspect:
                inspect()
            elif arguments.promote:
                promote()
            else:
                build()
