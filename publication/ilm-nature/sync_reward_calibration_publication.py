#!/usr/bin/env python3
"""Copy the reviewed calibration article to a new, verified local package."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HELPER = HERE / 'sync_glyph_score_gap_publication.py'
import hashlib
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == '95e34f67e07a825123fe0b1e6fcebc0fc649650400ba8764af7bed040b41c596'
spec = importlib.util.spec_from_file_location('previous_delivery', HELPER)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sha, load, source = shared.sha, shared.load, shared.source
BASE = shared.BASE
NAME = 'ILM_NatureStyle_20261116_RewardCalibration'
OLD = HERE / 'evidence/glyph_score_gap_local_sync.json'
RECEIPT = HERE / 'evidence/reward_calibration_local_sync.json'
BUILD = HERE / 'evidence/reward_calibration_article_build.json'
STAGE = ROOT / 'artifacts/reward_calibration_local_delivery_v1'
SCIENTIFIC_COMMIT = '43086a61fe07a0ce06aadcb318fe6c0fff62a293'


def authenticate_model():
    shared.authenticate_model()
    owner = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']['active_bootstrap_supervisor']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == owner['boot_id']
    raw = Path(f"/proc/{owner['pid']}/stat").read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    assert fields[0] != 'Z' and fields[19] == owner['start_ticks']


def previous_unchanged():
    assert sha(OLD) == '242a876117832ccab2d3ae4feb6a9f1366592865aee2308c977d668dd155744d'
    old = load(OLD)
    for name, digest in old['files_sha256'].items():
        assert sha(BASE / old['package_name'] / name) == digest, ('old package changed', name)
    assert sha(BASE / old['package_name'] / 'evidence' / OLD.name) == sha(OLD)
    return old


def payload(old, build):
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    names = set(old['files_sha256']) | set(build['source_sha256'])
    names |= set(build['immediate_baseline']['figures_sha256'])
    names |= {name + '.pdf' for name in build['documents']}
    names |= {'redlined_manuscript.tex', 'evidence/' + BUILD.name,
              'evidence/' + OLD.name, Path(__file__).name,
              'references/reward_calibration_local_delivery_plan.md',
              'references/reward_calibration_publication_revision_plan.md',
              'docs/first-imagized-language-model-goal.md'}
    for path in HERE.iterdir():
        if path.is_file() and 'reward_calibration' in path.name and path.suffix in ['.py', '.tex', '.mjs']:
            names.add(path.name)
    for path in (HERE / 'figures').rglob('*'):
        if path.is_file() and ('reward_calibration' in str(path.relative_to(HERE)).lower()):
            names.add(str(path.relative_to(HERE)))
    for path in (HERE / 'evidence').glob('reward_calibration*.json'):
        names.add(str(path.relative_to(HERE)))
    prefixes = ('whole_answer_', 'selective_visual_verifier_')
    for name in tracked:
        path = Path(name)
        if path.parent in [Path('references'), Path('references/evidence')] and path.name.startswith(prefixes) and path.suffix in ['.json', '.md']:
            names.add(name)
    names.discard('evidence/' + RECEIPT.name)
    files = {}
    for name in sorted(names):
        path = source(name)
        relative = str(path.relative_to(ROOT))
        assert relative in tracked or path == Path(__file__).resolve() or name == 'references/reward_calibration_local_delivery_plan.md', ('untracked payload', relative)
        files[name] = sha(path)
    return files


def deliver():
    began = time.monotonic()
    assert shared.available_host_gib() >= 48, 'host reserve'
    authenticate_model()
    old = previous_unchanged()
    assert sha(BUILD) == '6be4f4363fec650519297a994352b74f93ec2355e2d590893cafa7e36a1d77c1'
    build = load(BUILD)
    assert build['promoted'] and build['visual_review_complete'] and build['active_article_updated']
    for name, digest in build['source_sha256'].items():
        assert sha(HERE / name) == digest, ('compiled source changed', name)
    for name, row in build['documents'].items():
        assert sha(HERE / (name + '.pdf')) == row['pdf_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SCIENTIFIC_COMMIT
    for name, digest in old['latest_alias_sha256'].items():
        assert sha(BASE / name) == digest, ('owned alias changed', name)
    destination = BASE / NAME
    assert not destination.exists() and not RECEIPT.exists() and not STAGE.exists(), 'Preserve an existing delivery; verify instead'
    files = payload(old, build)
    size = sum(source(name).stat().st_size for name in files)
    reserve = load(ROOT / 'references/evidence/white_target_ranking_registration_20261115.json')['minimum_free_disk_gib']
    assert shutil.disk_usage(BASE).free > size + max(reserve, 1) * 1024 ** 3
    STAGE.mkdir(parents=True)
    journal = dict(package_name=NAME, phase='copying', began_unix=time.time(), files_sha256=files)
    (STAGE / 'journal.json').write_text(json.dumps(journal, indent=2) + '\n')
    destination.mkdir()
    for name, digest in files.items():
        assert shared.available_host_gib() >= 48, 'host reserve'
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source(name), target)
        assert sha(target) == digest, ('copied file differs', name)
    previous_unchanged()
    authenticate_model()
    aliases = {}
    for name, filename in shared.ALIASES.items():
        assert sha(BASE / name) == old['latest_alias_sha256'][name]
        temporary = BASE / (name + '.reward-calibration.tmp')
        assert not temporary.exists(), 'Preserve unexpected staged alias'
        shutil.copy2(HERE / filename, temporary)
        assert sha(temporary) == sha(HERE / filename)
        temporary.replace(BASE / name)
        aliases[name] = sha(BASE / name)
    receipt = dict(schema='reward-calibration-local-delivery-v1', package_name=NAME,
        publication_source_commit=SCIENTIFIC_COMMIT, files_sha256=files,
        previous_package_name=old['package_name'], previous_sync_receipt_sha256=sha(OLD),
        previous_package_retained=True, previous_package_all_manifest_files_reverified=len(old['files_sha256']),
        previous_owned_alias_sha256=old['latest_alias_sha256'], latest_alias_sha256=aliases,
        selected_publication_and_public_evidence_only=True, receipt_self_excluded_from_manifest=True,
        publication_build_receipt_sha256=sha(BUILD), frozen_source_snapshot=True,
        PDF_pages={name: row['pages'] for name, row in build['inspection']['documents'].items()},
        local_hashes_verified=True, cloud_propagation_verified=False,
        Office_application_export_verified=False, current_model_ownership_and115sources_preserved=True,
        new_model_calls=0, new_optimizer_updates=0, bytes_copied=size, elapsed_seconds=time.monotonic()-began)
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
    assert sha(OLD) == receipt['previous_sync_receipt_sha256']
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
