#!/usr/bin/env python3
"""Deliver reviewed public evidence while preserving the active model and old package."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import sync_glyph_score_gap_publication as shared
from prepare_visual_context_evidence import protected_sources

ROOT,HERE,BASE=shared.ROOT,shared.HERE,shared.BASE
sha,load,source=shared.sha,shared.load,shared.source
NAME='ILM_NatureStyle_20261117_VisualContextEvidence'
COMMIT='27b090fb7d4a292834af2a13456e8d16a1f7241c'
OLD=HERE/'evidence/reward_calibration_local_sync.json'
OLD_SHA='74a59c24e78077b3957cd7cbb5184f5acb1ea358140365e498667750e73543ec'
BUILD=HERE/'evidence/visual_context_evidence_article_build.json'
RECEIPT=HERE/'evidence/visual_context_evidence_local_sync.json'
STAGE=ROOT/'artifacts/visual_context_evidence_local_delivery_v1'
PLAN='references/visual_context_evidence_local_delivery_plan.md'


def authenticate():shared.authenticate_model();return protected_sources()


def previous():
    assert sha(OLD)==OLD_SHA;old=load(OLD)
    for name,digest in old['files_sha256'].items():assert sha(BASE/old['package_name']/name)==digest,name
    assert sha(BASE/old['package_name']/'evidence'/OLD.name)==sha(OLD)
    return old


def payload(old,build):
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    names=set(old['files_sha256'])|set(build['source_sha256'])|set(build['immediate_baseline']['figures_sha256'])
    names|={name+'.pdf' for name in build['documents']}
    names|={'redlined_manuscript.tex','evidence/'+BUILD.name,'evidence/'+OLD.name,Path(__file__).name,PLAN,
        'references/visual_context_evidence_manuscript_revision1.md','docs/trainable-visual-context.md',
        'docs/first-imagized-language-model-goal.md'}
    for name in tracked:
        p=Path(name)
        if name.startswith('publication/ilm-nature/'):
            rel=str(p.relative_to('publication/ilm-nature'))
            if any(word in rel for word in ('visual_context','white_target_completed')) and p.suffix in ('.py','.mjs','.tex','.json','.png','.pptx'):
                names.add(rel)
        elif name.startswith('references/') and p.suffix in ('.md','.json','.png') and p.name.startswith(
            ('white_target_physical','isolated_glyph','isolated_reader','training_glyph_label','training_glyph_supervision',
             'frozen_encoder_verifier','trainable_visual_context')):names.add(name)
    names.discard('evidence/'+RECEIPT.name);files={}
    for name in sorted(names):
        p=source(name);relative=str(p.relative_to(ROOT))
        assert relative in tracked or p==Path(__file__).resolve() or name==PLAN,relative
        files[name]=sha(p)
    return files


def deliver():
    began=time.monotonic();assert shared.available_host_gib()>=48;bindings=authenticate();old=previous();build=load(BUILD)
    assert build['promoted'] and build['visual_review_complete'] and build['active_article_updated']
    for n,h in build['source_sha256'].items():assert sha(HERE/n)==h,n
    for n,row in build['documents'].items():assert sha(HERE/(n+'.pdf'))==row['pdf_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==COMMIT
    for name,h in old['latest_alias_sha256'].items():assert sha(BASE/name)==h,('owned alias changed',name)
    destination=BASE/NAME;assert not destination.exists() and not RECEIPT.exists() and not STAGE.exists()
    files=payload(old,build);size=sum(source(n).stat().st_size for n in files)
    assert shutil.disk_usage(BASE).free>size+48*1024**3
    STAGE.mkdir(parents=True);destination.mkdir()
    journal=dict(phase='copying',package_name=NAME,files_sha256=files)
    (STAGE/'journal.json').write_text(json.dumps(journal,indent=2)+'\n')
    for name,h in files.items():
        assert shared.available_host_gib()>=48
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source(name),target)
        assert sha(target)==h,name
    previous();authenticate();aliases={}
    for name,filename in shared.ALIASES.items():
        assert sha(BASE/name)==old['latest_alias_sha256'][name]
        temporary=BASE/(name+'.visual-context.tmp');assert not temporary.exists()
        shutil.copy2(HERE/filename,temporary);assert sha(temporary)==sha(HERE/filename);temporary.replace(BASE/name)
        aliases[name]=sha(BASE/name)
    value=dict(schema='visual-context-evidence-local-delivery-v1',package_name=NAME,publication_source_commit=COMMIT,
        files_sha256=files,previous_package_name=old['package_name'],previous_sync_receipt_sha256=sha(OLD),
        previous_package_retained=True,previous_package_all_manifest_files_reverified=len(old['files_sha256']),
        previous_owned_alias_sha256=old['latest_alias_sha256'],latest_alias_sha256=aliases,
        publication_build_receipt_sha256=sha(BUILD),protected_model_bindings=bindings,
        receipt_self_excluded_from_manifest=True,selected_publication_and_public_evidence_only=True,
        PDF_pages={n:r['pages'] for n,r in build['inspection']['documents'].items()},
        local_hashes_verified=True,cloud_propagation_verified=False,Office_application_export_verified=False,
        publication_new_model_calls=0,publication_new_optimizer_updates=0,bytes_copied=size,
        elapsed_seconds=time.monotonic()-began)
    with RECEIPT.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    target=destination/'evidence'/RECEIPT.name;shutil.copy2(RECEIPT,target);assert sha(target)==sha(RECEIPT)
    journal.update(phase='complete',receipt_sha256=sha(RECEIPT));(STAGE/'journal.json').write_text(json.dumps(journal,indent=2)+'\n')
    verify()


def verify():
    r=load(RECEIPT);destination=BASE/r['package_name']
    for n,h in r['files_sha256'].items():assert sha(destination/n)==h,n
    for n,h in r['latest_alias_sha256'].items():assert sha(BASE/n)==h,n
    assert sha(destination/'evidence'/RECEIPT.name)==sha(RECEIPT)
    previous();print(json.dumps(dict(package=NAME,manifest_files=len(r['files_sha256']),
        prior_files_unchanged=983,owned_aliases_verified=True,receipt_verified_separately=True,cloud_propagation_verified=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    with (ROOT/'artifacts/publication_delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.verify:verify()
        else:deliver()
