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
NAME = 'ILM_NatureStyle_20261114_CompleteContinuation'
OLD_RECEIPT = HERE / 'evidence/realized_writer_local_sync.json'
RECEIPT = HERE / 'evidence/complete_continuation_local_sync.json'
STAGE = ROOT / 'artifacts/complete_continuation_local_delivery_v1'
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


def authenticate_model():
    handoff = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == handoff['active_model_worker_boot_id']
    for key, start in [('active_controller', 'active_controller_start_ticks'),
                       ('active_model_worker_pid', 'active_model_worker_start_ticks')]:
        raw = Path(f"/proc/{handoff[key]}/stat").read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        assert fields[19] == handoff[start] and fields[0] != 'Z'
    registry = load(ROOT / 'references/evidence/joint_context_learning_registration_20261114.json')
    assert len(registry['source_sha256']) == 59
    assert all(sha(ROOT / name) == digest for name, digest in registry['source_sha256'].items())


def previous_unchanged():
    old = load(OLD_RECEIPT)
    for name, digest in old['files_sha256'].items():
        assert sha(BASE / old['package_name'] / name) == digest, ('old package changed', name)
    return old


def payload(old, build):
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    names = set(old['files_sha256']) | set(build['source_sha256'])
    names |= set(build['immediate_baseline']['figures_sha256'])
    names |= {name + '.pdf' for name in build['documents']}
    names |= {'redlined_manuscript.tex', 'build_complete_continuation_article.py',
              'sync_complete_continuation_publication.py',
              'evidence/complete_continuation_article_build.json',
              'evidence/realized_writer_local_sync.json',
              'docs/first-imagized-language-model-goal.md'}
    publication_prefixes = ('complete_continuation', 'continuation_history', 'joint_writer',
        'prepare_complete_continuation', 'prepare_continuation_history', 'prepare_joint_writer',
        'verify_complete_continuation', 'verify_continuation_history', 'verify_joint_writer',
        'format_joint_writer')
    for path in HERE.iterdir():
        if path.is_file() and path.name.startswith(publication_prefixes) and path.suffix in ['.py', '.mjs', '.tex']:
            names.add(path.name)
    for path in (HERE / 'evidence').glob('*.json'):
        if path.name.startswith(('complete_continuation', 'continuation_history', 'joint_writer', 'joint_context_staged')):
            names.add('evidence/' + path.name)
    reference_prefixes = ('visual_continuation_', 'continuation_', 'joint_writer_', 'joint_context_')
    for name in tracked:
        path = Path(name)
        if (path.parent in [Path('references'), Path('references/evidence')]
                and path.name.startswith(reference_prefixes)
                and path.suffix in ['.md', '.json', '.png', '.svg', '.pdf', '.tex', '.csv']):
            names.add(name)
    names.discard('evidence/' + RECEIPT.name)
    files = {}
    for name in sorted(names):
        path = source(name)
        relative = str(path.relative_to(ROOT))
        assert relative in tracked or path == Path(__file__).resolve(), ('untracked payload', relative)
        files[name] = sha(path)
    return files


def deliver():
    authenticate_model()
    old = previous_unchanged()
    destination = BASE / NAME
    assert not destination.exists(), 'Preserve an existing destination; verify it instead of overwriting'
    assert not RECEIPT.exists(), 'Preserve the existing receipt'
    build = load(HERE / 'evidence/complete_continuation_article_build.json')
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
    reserve = load(ROOT / 'references/evidence/joint_context_learning_registration_20261114.json')['minimum_free_disk_gib']
    assert available > size + max(reserve, 1) * 1024 ** 3, 'Preserve the registered model disk reserve'
    STAGE.mkdir(parents=True, exist_ok=False)
    journal = dict(package_name=NAME, phase='copying', files_sha256=files,
                   began_unix=time.time(), previous_receipt_sha256=sha(OLD_RECEIPT))
    (STAGE / 'journal.json').write_text(json.dumps(journal, indent=2) + '\n')
    destination.mkdir()
    for name, digest in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source(name), target)
        assert sha(target) == digest, ('copied file differs', name)
    previous_unchanged()
    authenticate_model()
    aliases = {}
    for name, filename in ALIASES.items():
        assert sha(BASE / name) == old['latest_alias_sha256'][name]
        temporary = BASE / (name + '.complete-continuation.tmp')
        assert not temporary.exists(), 'Preserve an unexpected staged alias'
        shutil.copy2(HERE / filename, temporary)
        assert sha(temporary) == sha(HERE / filename)
        temporary.replace(BASE / name)
        aliases[name] = sha(BASE / name)
    receipt = dict(schema='complete-continuation-local-delivery-v1', package_name=NAME,
        files_sha256=files, previous_package_name=old['package_name'],
        previous_sync_receipt_sha256=sha(OLD_RECEIPT), previous_package_retained=True,
        previous_package_all_manifest_files_reverified=len(old['files_sha256']),
        previous_owned_alias_sha256=old['latest_alias_sha256'], latest_alias_sha256=aliases,
        selected_publication_and_public_evidence_only=True, receipt_self_excluded_from_manifest=True,
        publication_build_receipt_sha256=sha(HERE / 'evidence/complete_continuation_article_build.json'),
        PDF_pages={name: row['pages'] for name, row in build['inspection']['documents'].items()},
        local_hashes_verified=True, cloud_propagation_verified=False,
        current_model_ownership_and59sources_preserved=True, new_model_calls=0,
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
