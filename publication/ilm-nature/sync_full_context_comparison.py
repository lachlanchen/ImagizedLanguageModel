"""Deliver the verified article, editable figure and sources to the local ILM folder."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from sync_glyph_score_gap_publication import ALIASES, BASE, HERE, ROOT, load, sha, source
from scripts.diagnose_visual_context_gradients import require, write_new

PREVIOUS = HERE/'evidence/visual_context_evidence_local_sync.json'
BUILD = HERE/'evidence/full_context_comparison_article_build.json'
RECEIPT = HERE/'evidence/full_context_comparison_local_sync.json'
DEST = BASE/'ILM_NatureStyle_20261118_FullContextComparison'


def sync():
    old, build = load(PREVIOUS), load(BUILD)
    require(build['promoted'], 'Require inspected, promoted editions')
    previous = BASE / old['package_name']
    for name, digest in old['files_sha256'].items():
        require(sha(previous/name) == digest, 'Previous package changed: '+name)
    if RECEIPT.exists():
        record = load(RECEIPT)
        require(record['build_receipt_sha256'] == sha(BUILD), 'Article build receipt differs')
        for name, digest in record['files_sha256'].items():
            require(sha(DEST/name) == digest, 'Delivered snapshot changed: '+name)
        for name, digest in record['latest_alias_sha256'].items():
            require(sha(BASE/name) == digest, 'Latest alias changed: '+name)
        copy = DEST/'evidence'/RECEIPT.name
        if not copy.exists():
            shutil.copyfile(RECEIPT, copy)
        require(sha(copy) == sha(RECEIPT), 'Delivered receipt differs')
        print(f"Verified {len(record['files_sha256'])} files against their saved delivery hashes.")
        return
    names = set(old['files_sha256']) | set(build['all_root_tex_sha256'])
    additions = ['evidence/'+PREVIOUS.name,'evidence/'+BUILD.name,'full_context_comparison_figure.mjs',
                 'build_full_context_comparison_article.py','figures/Full_context_comparison.pptx',
                 'figures/Figure_32_full_context_comparison.png','references/full_context_comparison_manuscript_revision1.md',
                 'references/trainable_visual_context_result_20261117.md',
                 'references/evidence/trainable_visual_context_result_20261117.json']
    names.update(additions)
    hashes = {n:sha(source(n)) for n in sorted(names)}
    doc_hashes = {n+'.pdf':r['pdf_sha256'] for n,r in build['inspection']['documents'].items()}
    require(all(hashes[n] == d for n,d in doc_hashes.items()),'Compiled PDF changed')
    require(all(hashes[n] == h for n,h in build['all_root_tex_sha256'].items()),'Published TeX changed')
    latest = {n:hashes[doc] for n,doc in ALIASES.items()}
    for name in latest:
        require(sha(BASE/name) in (old['latest_alias_sha256'][name],latest[name]), 'Latest alias changed: '+name)
    if not DEST.exists():
        size = sum(source(n).stat().st_size for n in names)
        require(shutil.disk_usage(BASE).free > size + 48*1024**3,'Disk reserve')
        with TemporaryDirectory(dir=BASE) as scratch:
            stage=Path(scratch)/DEST.name
            for name,digest in hashes.items():
                target=stage/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source(name),target)
                require(sha(target) == digest,'Copy changed')
            stage.rename(DEST)
    require(all(sha(DEST/n)==h for n,h in hashes.items()),'Versioned delivery differs')
    for name,doc in ALIASES.items():
        if sha(BASE/name) != latest[name]:
            temp=BASE/(name+'.full_context_tmp')
            with (DEST/doc).open('rb') as src, temp.open('xb') as dst:
                shutil.copyfileobj(src,dst)
            os.replace(temp,BASE/name)
        require(sha(BASE/name)==latest[name],'Alias differs')
    if not RECEIPT.exists():
        write_new(RECEIPT,dict(package_name=DEST.name,previous_package_name=previous.name,
                  previous_package_files_verified=len(old['files_sha256']),files_sha256=hashes,
                  latest_alias_sha256=latest,publication_source_commit=subprocess.check_output(
                  ['git','log','-1','--format=%H','--',BUILD],cwd=ROOT,text=True).strip(),
                  build_receipt_sha256=sha(BUILD),local_hashes_verified=True,cloud_propagation_verified=False))
    record=load(RECEIPT)
    require(record['files_sha256']==hashes and record['latest_alias_sha256']==latest,'Different delivery receipt')
    record_copy=DEST/'evidence'/RECEIPT.name
    if not record_copy.exists():
        shutil.copyfile(RECEIPT,record_copy)
    require(sha(record_copy)==sha(RECEIPT),'Receipt differs')
    print(f'Verified {len(hashes)} delivered files and {len(latest)} latest PDF aliases.')


if __name__ == '__main__':
    with (ROOT/'artifacts/publication_delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sync()
