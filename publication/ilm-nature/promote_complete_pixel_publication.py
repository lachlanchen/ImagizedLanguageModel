#!/usr/bin/env python3
"""Promote reviewed complete-learning artifacts and verify local Nutstore sync."""
import argparse
from pathlib import Path

from complete_pixel_publication import BASE, CODE, DOCS, FIGURES, HERE, ROOT, RESULT_INSERT, DISCUSSION_ANCHOR, baseline, build_slot, load, require, save, sha, verify
from complete_pixel_learning_text import DISCUSSION
from promote_native_flow_manuscript import ALIASES, DEST, atomic_copy

NAMES = ('frontmatter.tex', 'article_content.tex', 'supplementary_information.tex',
         'complete_pixel_learning_results.tex', 'complete_pixel_learning_note.tex',
         'redlined_manuscript.tex', 'redlined_supplementary_information.tex',
         *(stem + '.pdf' for stem in DOCS), 'ILM_Figures_editable.pptx', 'ILM_Figures_review.pdf',
         *('figures/' + name + '.' + suffix for name in FIGURES for suffix in ('pdf', 'png', 'svg')))


def reviewed(stage):
    built, review = load(stage / 'build.json'), load(stage / 'review.json')
    require(review['build_receipt_sha256'] == sha(stage / 'build.json')
            and review['all_changed_PDF_pages_actually_viewed']
            and review['PDFs_sha256'] == built['PDFs_sha256'], 'actual matching PDF review required')
    require(set(review['staged_files_sha256']) == set(NAMES), 'complete fixed promotion set required')
    for name, digest in review['staged_files_sha256'].items():
        require(sha(stage / 'paper' / name) == digest, 'reviewed file changed: ' + name)
    return built, review


def promote(stage):
    with build_slot():
        meta = verify(stage)
        built, review = reviewed(stage)
        article = (stage / 'paper/article_content.tex').read_text().replace(RESULT_INSERT, '')
        article = article.replace(DISCUSSION_ANCHOR + '\n' + DISCUSSION + '\n', DISCUSSION_ANCHOR)
        require(article == baseline('article_content.tex').decode(), 'article changed outside the planned insertions')
        supplement = (stage / 'paper/supplementary_information.tex').read_text().replace('\\input{complete_pixel_learning_note.tex}\n\n', '')
        require(supplement == baseline('supplementary_information.tex').decode(), 'prior supplement body changed')
        require((stage / 'paper/frontmatter.tex').read_text().split(r'\newcommand{\PaperSummary}')[0]
                == baseline('frontmatter.tex').decode().split(r'\newcommand{\PaperSummary}')[0], 'title or prior value inputs changed')
        folder = HERE / 'baselines/pre-complete-pixel-d387a49'
        folder.mkdir(exist_ok=False)
        save(folder / 'baseline_manifest.json', dict(source_commit=BASE, files_sha256=meta['active_before_sha256'], bytes_recoverable_from_Git=True))
        for name in NAMES:
            atomic_copy(stage / 'paper' / name, HERE / name)
        promoted = {name: sha(HERE / name) for name in NAMES}
        require(promoted == review['staged_files_sha256'], 'active promotion differs from reviewed bytes')
        for name, digest in meta['active_before_sha256'].items():
            if name not in NAMES:
                require(sha(HERE / name) == digest, 'unrelated publication asset changed')
        for name in ('preparation', 'render', 'figure_review', 'build', 'review'):
            atomic_copy(stage / (name + '.json'), HERE / 'evidence' / ('complete_pixel_' + name + '.json'))
        receipt = dict(baseline_commit=BASE, promoted_files_sha256=promoted,
            preparation_sha256=sha(stage / 'preparation.json'), build_receipt_sha256=sha(stage / 'build.json'),
            review_receipt_sha256=sha(stage / 'review.json'), prior48_slide_appearances_preserved=True,
            prior53_native_charts_preserved=True, current_native_slides=50, current_native_charts=54,
            title_and_Introduction_preserved=True, active_publication_updated=True, Nutstore_updated=False,
            general_ilm_goal_complete=False)
        save(stage / 'promotion.json', receipt)
        save(HERE / 'evidence/complete_pixel_promotion.json', receipt)
        print('Promoted the reviewed complete-learning manuscript and 50-slide native deck.')


def sync(stage):
    with build_slot():
        _, review = reviewed(stage)
        meta = load(stage / 'preparation.json')
        promotion = load(stage / 'promotion.json')
        for name, digest in promotion['promoted_files_sha256'].items():
            require(sha(HERE / name) == digest, 'active file changed before local sync')
        require(DEST.is_dir(), 'existing authorized Nutstore ILM destination required')
        for name, alias in ALIASES.items():
            path = DEST / alias
            if path.exists():
                require(sha(path) in (meta['active_before_sha256'][name], sha(HERE / name)), 'a different Nutstore revision exists: ' + alias)
        package = DEST / 'ILM_NatureStyle_20261109_CompletePixelLearning'
        package.mkdir(exist_ok=False)
        files = {}
        for path in sorted((stage / 'paper').rglob('*')):
            if path.is_file():
                relative = path.relative_to(stage / 'paper')
                atomic_copy(path, package / relative)
                files[str(relative)] = sha(package / relative)
        for name in ('build', 'review', 'promotion', 'render', 'figure_review'):
            atomic_copy(stage / (name + '.json'), package / 'evidence' / (name + '.json'))
            files['evidence/' + name + '.json'] = sha(stage / (name + '.json'))
        aliases = {}
        for name, alias in ALIASES.items():
            path = DEST / alias
            if path.exists() and sha(path) != sha(HERE / name):
                old = sha(path)
                backup = path.with_name(path.stem + '_before_complete_pixel_' + old[:12] + path.suffix)
                if backup.exists():
                    require(sha(backup) == old, 'alias backup differs')
                else:
                    atomic_copy(path, backup)
            atomic_copy(HERE / name, path)
            aliases[alias] = sha(path)
            require(aliases[alias] == sha(HERE / name), 'local latest alias differs')
        receipt = dict(package_folder=package.name, files_sha256=files, aliases_sha256=aliases,
            verified_local_copy=True, cloud_propagation_verified=False, PowerPoint_application_inspected=False,
            general_ilm_goal_complete=False)
        save(stage / 'sync.json', receipt)
        save(HERE / 'evidence/complete_pixel_sync.json', receipt)
        print('Verified the local Nutstore package and five latest aliases.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('promote', 'sync'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.input.resolve())
