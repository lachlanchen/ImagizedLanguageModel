#!/usr/bin/env python3
"""Deliver the reviewed article locally without changing old packages or models."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path.home() / 'Nutstore Files/Projects/ILM'
NAME = 'ILM_NatureStyle_20261116_ScoreGaps'
OLD_RECEIPT = HERE / 'evidence/ranking_completed_local_sync.json'
RECEIPT = HERE / 'evidence/glyph_score_gap_local_sync.json'
STAGE = ROOT / 'artifacts/glyph_score_gap_local_delivery_v1'
ALIASES = {'ILM_Manuscript_LATEST.pdf': 'manuscript.pdf',
           'ILM_Manuscript_Review_LATEST.pdf': 'manuscript_review.pdf'}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source(name):
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    assert not any(part in ['.git', '.private', '.cache', 'artifacts',
                            'node_modules', 'office_profile', 'baseline_sources']
                   for part in path.parts)
    assert path.suffix not in ['.pt', '.pth', '.bin', '.gguf', '.safetensors']
    result = ROOT / path if path.parts[0] in ['references', 'docs'] else HERE / path
    resolved = result.resolve(strict=True)
    assert ROOT in resolved.parents and result.is_file()
    return result


def available_host_gib():
    for row in Path('/proc/meminfo').read_text().splitlines():
        if row.startswith('MemAvailable:'):
            return int(row.split()[1]) / 1048576
    raise RuntimeError('MemAvailable is unavailable')


def authenticate_model():
    handoff = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == handoff['active_model_worker_boot_id']
    for key, start in [('active_controller', 'active_controller_start_ticks'),
                       ('active_model_worker_pid', 'active_model_worker_start_ticks')]:
        raw = Path(f"/proc/{handoff[key]}/stat").read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        assert fields[19] == handoff[start] and fields[0] != 'Z'
    registration = ROOT / 'references/evidence/white_target_ranking_registration_20261115.json'
    assert sha(registration) == '04b908f488f5175bff1df8da1a7d38676696c03a3ab5f70faa6a199a1757d828'
    registry = load(registration)
    assert len(registry['source_sha256']) == 115
    assert all(sha(ROOT / name) == digest for name, digest in registry['source_sha256'].items())


def previous_unchanged():
    assert sha(OLD_RECEIPT) == '893a23258bca22781135fa914275c66ac7badd64ace00d9a0144e120968ccba3'
    old = load(OLD_RECEIPT)
    for name, digest in old['files_sha256'].items():
        assert sha(BASE / old['package_name'] / name) == digest, ('old package changed', name)
    return old


def payload(old, build):
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    names = set(old['files_sha256']) | set(build['source_sha256'])
    names |= set(build['immediate_baseline']['figures_sha256'])
    names |= {name + '.pdf' for name in build['documents']}
    names |= {'redlined_manuscript.tex', 'build_glyph_score_gap_article.py',
              'sync_glyph_score_gap_publication.py',
              'evidence/glyph_score_gap_article_build.json',
              'references/glyph_score_gap_local_delivery_plan.md',
              'references/glyph_score_gap_publication_revision_plan.md',
              'evidence/realized_writer_local_sync.json',
              'docs/first-imagized-language-model-goal.md'}
    publication_prefixes = ('complete_continuation', 'continuation_history', 'joint_writer',
        'prepare_complete_continuation', 'prepare_continuation_history', 'prepare_joint_writer',
        'verify_complete_continuation', 'verify_continuation_history', 'verify_joint_writer',
        'format_joint_writer', 'context_failure', 'prepare_context_failure', 'build_context_failure', 'sync_context_failure',
        'ranking_completed', 'prepare_ranking_completed', 'verify_ranking_completed',
        'build_ranking_completed', 'sync_ranking_completed', 'white_target_methods',
        'glyph_score_gap', 'prepare_glyph_score_gap', 'verify_glyph_score_gap',
        'build_glyph_score_gap', 'sync_glyph_score_gap')
    for path in HERE.iterdir():
        if path.is_file() and path.name.startswith(publication_prefixes) and path.suffix in ['.py', '.mjs', '.tex']:
            names.add(path.name)
    for path in (HERE / 'evidence').glob('*.json'):
        if path.name.startswith(('complete_continuation', 'continuation_history', 'joint_writer', 'joint_context_staged', 'context_failure', 'ranking_completed', 'glyph_score_gap')):
            names.add('evidence/' + path.name)
    reference_prefixes = ('visual_continuation_', 'continuation_', 'joint_writer_', 'joint_context_', 'global_glyph_ranking_', 'ranked_writer_whitespace_', 'white_target_', 'ranked_glyph_score_gap_', 'glyph_score_gap_')
    for name in tracked:
        path = Path(name)
        if (path.parent in [Path('references'), Path('references/evidence')]
                and path.name.startswith(reference_prefixes)
                and path.suffix in ['.md', '.json', '.png', '.svg', '.pdf', '.tex', '.csv']):
            names.add(name)
    names |= {'figures/Glyph_score_gap_figures.pptx', 'figures/Figure_28_glyph_score_gaps.png'}
    names.discard('evidence/' + RECEIPT.name)
    files = {}
    for name in sorted(names):
        path = source(name)
        relative = str(path.relative_to(ROOT))
        assert relative in tracked or path == Path(__file__).resolve() or name == 'references/glyph_score_gap_local_delivery_plan.md', ('untracked payload', relative)
        files[name] = sha(path)
    return files


def deliver():
    assert available_host_gib() >= 48, 'host reserve'
    authenticate_model()
    old = previous_unchanged()
    destination = BASE / NAME
    assert not destination.exists(), 'Preserve an existing destination; verify it instead of overwriting'
    assert not RECEIPT.exists(), 'Preserve the existing receipt'
    build_path = HERE / 'evidence/glyph_score_gap_article_build.json'
    assert sha(build_path) == '0092069d15a99890e546718dcb172ae4eb628ee4fb04f72e3035fc9b026f663b'
    build = load(build_path)
    assert build['promoted'] and build['visual_review_complete'] and build['active_article_updated']
    for name, digest in build['source_sha256'].items():
        assert sha(HERE / name) == digest, ('compiled source changed', name)
    for name, row in build['documents'].items():
        assert sha(HERE / (name + '.pdf')) == row['pdf_sha256']
    for name, digest in old['latest_alias_sha256'].items():
        assert sha(BASE / name) == digest, ('owned alias changed', name)
    files = payload(old, build)
    available = shutil.disk_usage(BASE).free
    size = sum(source(name).stat().st_size for name in files)
    reserve = load(ROOT / 'references/evidence/white_target_ranking_registration_20261115.json')['minimum_free_disk_gib']
    assert available > size + max(reserve, 1) * 1024 ** 3, 'Preserve the registered model disk reserve'
    STAGE.mkdir(parents=True, exist_ok=False)
    journal = dict(package_name=NAME, phase='copying', files_sha256=files,
                   began_unix=time.time(), previous_receipt_sha256=sha(OLD_RECEIPT))
    (STAGE / 'journal.json').write_text(json.dumps(journal, indent=2) + '\n')
    destination.mkdir()
    for name, digest in files.items():
        assert available_host_gib() >= 48, 'host reserve'
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source(name), target)
        assert sha(target) == digest, ('copied file differs', name)
    previous_unchanged()
    authenticate_model()
    aliases = {}
    for name, filename in ALIASES.items():
        assert sha(BASE / name) == old['latest_alias_sha256'][name]
        temporary = BASE / (name + '.glyph-score-gap.tmp')
        assert not temporary.exists(), 'Preserve an unexpected staged alias'
        shutil.copy2(HERE / filename, temporary)
        assert sha(temporary) == sha(HERE / filename)
        temporary.replace(BASE / name)
        aliases[name] = sha(BASE / name)
    receipt = dict(schema='glyph-score-gap-local-delivery-v1', package_name=NAME,
        publication_source_commit='b9f7a3d86f1104567f2f359c1a18428ff9994fd9',
        files_sha256=files, previous_package_name=old['package_name'],
        previous_sync_receipt_sha256=sha(OLD_RECEIPT), previous_package_retained=True,
        previous_package_all_manifest_files_reverified=len(old['files_sha256']),
        previous_owned_alias_sha256=old['latest_alias_sha256'], latest_alias_sha256=aliases,
        selected_publication_and_public_evidence_only=True, receipt_self_excluded_from_manifest=True,
        publication_build_receipt_sha256=sha(build_path),
        PDF_pages={name: row['pages'] for name, row in build['inspection']['documents'].items()},
        local_hashes_verified=True, cloud_propagation_verified=False,
        Office_application_export_verified=False, frozen_source_snapshot=True,
        current_model_ownership_and115sources_preserved=True, new_model_calls=0,
        new_optimizer_updates=0, bytes_copied=size,
        elapsed_seconds=time.time() - journal['began_unix'])
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    target = destination / 'evidence' / RECEIPT.name
    shutil.copy2(RECEIPT, target)
    assert sha(target) == sha(RECEIPT)
    journal.update(phase='complete', receipt_sha256=sha(RECEIPT))
    (STAGE / 'journal.json').write_text(json.dumps(journal, indent=2) + '\n')
    verify()


def verify():
    receipt = load(RECEIPT)
    destination = BASE / receipt['package_name']
    for name, digest in receipt['files_sha256'].items():
        assert sha(destination / name) == digest, ('new package changed', name)
    for name, digest in receipt['latest_alias_sha256'].items():
        assert sha(BASE / name) == digest, ('latest alias changed', name)
    assert sha(destination / 'evidence' / RECEIPT.name) == sha(RECEIPT)
    assert sha(OLD_RECEIPT) == receipt['previous_sync_receipt_sha256']
    previous_unchanged()
    print(json.dumps(dict(package=NAME, manifest_files=len(receipt['files_sha256']),
        previous_manifest_files_unchanged=receipt['previous_package_all_manifest_files_reverified'],
        owned_latest_aliases_verified=True, receipt_verified_separately=True,
        local_hashes_verified=True, cloud_propagation_verified=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'artifacts/publication_delivery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify() if args.verify else deliver()
