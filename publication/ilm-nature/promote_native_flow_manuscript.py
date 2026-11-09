#!/usr/bin/env python3
"""Promote only reviewed source-bound PDFs, then verify local Nutstore copies."""
import argparse
import json
import os
from pathlib import Path
import shutil

from native_flow_manuscript import BASE, DOCS, FIGURES, HERE, ROOT, baseline, build_slot, load, require, save, sha, verify

NAMES = ('article_content.tex','supplementary_information.tex','native_qa_results.tex','native_flow_results.tex',
    'native_qa_note.tex','native_flow_note.tex','redlined_manuscript.tex','redlined_supplementary_information.tex',
    *(name+'.pdf' for name in DOCS),'ILM_Figures_editable.pptx','ILM_Figures_review.pdf',
    *('figures/'+name+'.'+suffix for name in FIGURES for suffix in ('pdf','png','svg')))
DEST = Path.home()/'Nutstore Files/Projects/ILM'
ALIASES = {'manuscript.pdf':'ILM_Manuscript_LATEST.pdf','manuscript_review.pdf':'ILM_Manuscript_Review_LATEST.pdf',
    'supplementary_information.pdf':'ILM_Supplementary_LATEST.pdf',
    'ILM_Figures_editable.pptx':'ILM_Figures_editable_LATEST.pptx','ILM_Figures_review.pdf':'ILM_Figures_review_LATEST.pdf'}


def current_review(stage):
    built,review=load(stage/'build.json'),load(stage/'review.json')
    require(built['preparation_sha256']==sha(stage/'preparation.json')
        and review['build_receipt_sha256']==sha(stage/'build.json')
        and review['raw_PDF_page_review_completed']
        and review['PDFs_sha256']==built['PDFs_sha256'],'actual review of complete matching PDFs required')
    for name,digest in {**built['PDFs_sha256'],**built['document_sources_sha256']}.items():
        require(sha(stage/'paper'/name)==digest,'compiled/reviewed document changed: '+name)
    for name,digest in review['staged_files_sha256'].items():
        require(sha(stage/'paper'/name)==digest,'reviewed artifact changed: '+name)
    require(set(review['staged_files_sha256'])==set(NAMES),'complete fixed promotion set required')
    return built,review


def atomic_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name('.'+target.name+'.ilm-transfer')
    with temporary.open('xb') as destination,Path(source).open('rb') as origin:
        shutil.copyfileobj(origin,destination)
    require(sha(temporary)==sha(source),'temporary copy differs')
    os.replace(temporary,target)


def promote(stage):
    with build_slot():
        meta=verify(stage);built,review=current_review(stage)
        preserved={}
        for name in NAMES:
            path=HERE/name
            if path.exists():
                require(path.read_bytes()==baseline(name),'active replacement is not the declared baseline: '+name)
                preserved[name]=sha(path)
        folder=HERE/'baselines/pre-native-flow-daef4bf';folder.mkdir(exist_ok=False)
        save(folder/'baseline_manifest.json',dict(source_commit=BASE,files_sha256=meta['active_publication_sha256'],
            replacement_baselines_sha256=preserved,bytes_recoverable_from_Git=True))
        for name in NAMES:atomic_copy(stage/'paper'/name,HERE/name)
        promoted={name:sha(HERE/name) for name in NAMES}
        require(promoted==review['staged_files_sha256'],'active promotion differs from reviewed package')
        for name,digest in meta['active_publication_sha256'].items():
            if name not in NAMES:require(sha(HERE/name)==digest,'unrelated publication asset changed')
        evidence=HERE/'evidence'
        atomic_copy(stage/'build.json',evidence/'native_flow_manuscript_build.json')
        atomic_copy(stage/'review.json',evidence/'native_flow_manuscript_review.json')
        receipt=dict(baseline_commit=BASE,source_preparation_sha256=sha(stage/'preparation.json'),
            build_receipt_sha256=sha(stage/'build.json'),review_receipt_sha256=sha(stage/'review.json'),
            promoted_files_sha256=promoted,all_prior_figure_bytes_preserved=True,
            current_native_slides=46,current_native_charts=52,title_abstract_introduction_unchanged=True,
            active_publication_updated=True,Nutstore_updated=False,general_ilm_goal_complete=False)
        save(stage/'promotion.json',receipt);save(evidence/'native_flow_manuscript_promotion.json',receipt)
        print('Promoted the reviewed46-slide / five-PDF publication package; unrelated assets preserved.')


def sync(stage):
    _,review=current_review(stage);promotion=load(stage/'promotion.json');meta=load(stage/'preparation.json')
    for name,digest in promotion['promoted_files_sha256'].items():
        require(sha(HERE/name)==digest,'active reviewed artifact changed before sync')
    require(DEST.is_dir(),'existing authorized Nutstore ILM folder required')
    for name,alias in ALIASES.items():
        target=DEST/alias
        if target.exists():require(sha(target) in (meta['active_publication_sha256'][name],sha(HERE/name)),
            'Nutstore alias contains a different revision; preserve it: '+alias)
    package=DEST/'ILM_NatureStyle_20261109_CompleteFlow';package.mkdir(exist_ok=False)
    files={}
    # Include all TeX/style inputs and figures required to reproduce these PDFs.
    for source in sorted((stage/'paper').rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(stage/'paper')
        target=package/relative;atomic_copy(source,target);files[str(relative)]=sha(target)
    for name in ('build.json','review.json','promotion.json'):
        atomic_copy(stage/name,package/'evidence'/name);files['evidence/'+name]=sha(stage/name)
    aliases={}
    for name,alias in ALIASES.items():
        target=DEST/alias
        if target.exists() and sha(target)!=sha(HERE/name):
            old=sha(target);backup=target.with_name(target.stem+'_before_complete_flow_'+old[:12]+target.suffix)
            if backup.exists():require(sha(backup)==old,'different Nutstore alias backup exists')
            else:atomic_copy(target,backup)
        atomic_copy(HERE/name,target);aliases[alias]=sha(target)
        require(aliases[alias]==sha(HERE/name),'local alias copy differs')
    receipt=dict(package_folder=package.name,package_files_sha256=files,aliases_sha256=aliases,
        actual_review_receipt_sha256=sha(stage/'review.json'),verified_local_copy=True,
        cloud_propagation_verified=False,PowerPoint_application_inspection=False,general_ilm_goal_complete=False)
    save(stage/'sync.json',receipt);save(HERE/'evidence/native_flow_manuscript_sync.json',receipt)
    print('Verified local Nutstore source package and five latest aliases; cloud propagation is not asserted.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('promote','sync'),required=True)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();{'promote':promote,'sync':sync}[args.stage](args.input.resolve())
