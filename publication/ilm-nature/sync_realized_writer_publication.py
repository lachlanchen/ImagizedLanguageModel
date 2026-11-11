#!/usr/bin/env python3
"""Deliver the verified realized-writer package while preserving prior copies."""
import json
from pathlib import Path
import shutil

from sync_motor_diagnostics_publication import HERE, ROOT, BASE, load, sha

NAME = 'ILM_NatureStyle_20261112_RealizedWriting'


def main():
    old = load(HERE / 'evidence/motor_diagnostics_local_sync.json')
    previous, destination = BASE / old['package_name'], BASE / NAME
    if destination.exists():
        raise FileExistsError('Preserve existing delivery: ' + str(destination))
    build = load(HERE / 'evidence/realized_writer_publication_build.json')
    if not build['visual_QA']['visual_QA_complete'] or not build['all160_article_page_rasters_reproduced']:
        raise ValueError('Finish actual visual review and independent reproduction')
    for name, digest in build['files_sha256'].items():
        if sha(HERE / name) != digest:
            raise ValueError('Verified current source changed: ' + name)
    for name, digest in old['files_sha256'].items():
        if sha(previous / name) != digest:
            raise ValueError('Previous package changed: ' + name)
    for name, digest in old['latest_alias_sha256'].items():
        if sha(BASE / name) != digest:
            raise ValueError('Owned latest alias changed outside this update: ' + name)
    names = set(old['files_sha256']) | set(build['files_sha256']) | {
        'evidence/realized_writer_publication_build.json', 'sync_realized_writer_publication.py'}
    names |= {'evidence/' + p.name for p in (HERE / 'evidence').glob('realized_writer_*.json')
              if p.name != 'realized_writer_local_sync.json'}
    for pattern in ['rollout_endpoint_motor*', 'balanced_writer_matched*', 'manuscript_realized_writer_revision_plan.md']:
        for path in (ROOT / 'references').glob(pattern):
            if path.is_file():
                names.add(str(path.relative_to(ROOT)))
        for path in (ROOT / 'references/evidence').glob(pattern):
            if path.is_file():
                names.add(str(path.relative_to(ROOT)))
            elif path.is_dir():
                names |= {str(p.relative_to(ROOT)) for p in path.iterdir() if p.is_file()}
    files = {}
    for name in sorted(names):
        if 'baseline_sources/' in name or 'node_modules/' in name or 'office_profile/' in name:
            raise ValueError('Generated cache cannot enter delivery')
        source = ROOT / name if name.startswith('references/') else HERE / name
        if not source.is_file():
            raise FileNotFoundError(source)
        files[name] = sha(source)
    destination.mkdir()
    for name, digest in files.items():
        source = ROOT / name if name.startswith('references/') else HERE / name
        target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError('Copied file differs: ' + name)
    aliases = {}
    for name, source in [('ILM_Manuscript_LATEST.pdf', HERE / 'manuscript.pdf'),
                         ('ILM_Manuscript_Review_LATEST.pdf', HERE / 'manuscript_review.pdf')]:
        if sha(BASE / name) != old['latest_alias_sha256'][name]:
            raise ValueError('Alias changed during copy: ' + name)
        temporary = BASE / (name + '.realized-writing.tmp')
        shutil.copy2(source, temporary)
        if sha(temporary) != sha(source):
            raise ValueError('Staged latest PDF differs')
        temporary.replace(BASE / name); aliases[name] = sha(BASE / name)
    receipt = dict(package_name=NAME, files_sha256=files, previous_package_name=old['package_name'],
        previous_owned_alias_sha256=old['latest_alias_sha256'], latest_alias_sha256=aliases,
        previous_package_retained=True, previous_package_all_manifest_files_reverified=len(old['files_sha256']),
        local_hashes_verified=True, cloud_propagation_verified=False,
        receipt_self_excluded_from_manifest=True, selected_publication_and_public_evidence_only=True)
    target = HERE / 'evidence/realized_writer_local_sync.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    copied = destination / 'evidence/realized_writer_local_sync.json'
    shutil.copy2(target, copied)
    if sha(copied) != sha(target):
        raise ValueError('Copied sync receipt differs')
    print(json.dumps(dict(package=NAME, manifest_files=len(files), receipt_verified_separately=True,
        previous_manifest_files_unchanged=len(old['files_sha256']),
        owned_latest_aliases_verified=True, cloud_propagation_verified=False)))


if __name__ == '__main__':
    main()
