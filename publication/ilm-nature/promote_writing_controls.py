#!/usr/bin/env python3
"""Promote the reviewed writing-control package and verify local Nutstore copies."""
import argparse
from pathlib import Path

from writing_controls_publication import (BASE,DOCS,FIGURES,HERE,ROOT,baseline,build_slot,
    load,require,save,sha,verify)
from promote_native_flow_manuscript import ALIASES,DEST,atomic_copy

NAMES=('frontmatter.tex','article_content.tex','supplementary_information.tex',
    'writing_controls_results.tex','writing_controls_notes.tex',
    'redlined_manuscript.tex','redlined_supplementary_information.tex',
    *(stem+'.pdf' for stem in DOCS),'ILM_Figures_editable.pptx','ILM_Figures_review.pdf',
    *('figures/'+name+'.'+suffix for name in FIGURES for suffix in ('pdf','png','svg')))


def reviewed(stage):
    built,review=load(stage/'build.json'),load(stage/'review.json')
    require(review['build_receipt_sha256']==sha(stage/'build.json')
        and review['all_changed_PDF_pages_actually_viewed']
        and review['PDFs_sha256']==built['PDFs_sha256'],'actual matching full PDF review required')
    require(set(review['staged_files_sha256'])==set(NAMES),'complete fixed promotion set required')
    for name,digest in review['staged_files_sha256'].items():
        require(sha(stage/'paper'/name)==digest,'reviewed file changed: '+name)
    return built,review


def promote(stage):
    with build_slot():
        meta=verify(stage);built,review=reviewed(stage)
        addition=('\\subsection*{Precise image fitting does not establish language generalization}\n'
            '\\input{writing_controls_results.tex}\n\n')
        require((stage/'paper/article_content.tex').read_text().replace(addition,'')==baseline('article_content.tex').decode(),
            'prior article text changed outside the accepted insertion')
        require((stage/'paper/supplementary_information.tex').read_text().replace('\\input{writing_controls_notes.tex}\n\n','')
            ==baseline('supplementary_information.tex').decode(),'prior supplement text changed')
        require((stage/'paper/frontmatter.tex').read_text().split(r'\newcommand{\PaperSummary}')[0]
            ==baseline('frontmatter.tex').decode().split(r'\newcommand{\PaperSummary}')[0],
            'title or prior value inputs changed')
        preserved={name:sha(HERE/name) for name in NAMES if (HERE/name).exists()}
        folder=HERE/'baselines/pre-writing-controls-3011de6';folder.mkdir(exist_ok=False)
        save(folder/'baseline_manifest.json',dict(source_commit=BASE,files_sha256=meta['active_before_sha256'],
            replacement_baselines_sha256=preserved,bytes_recoverable_from_Git=True))
        for name in NAMES:atomic_copy(stage/'paper'/name,HERE/name)
        promoted={name:sha(HERE/name) for name in NAMES}
        require(promoted==review['staged_files_sha256'],'promotion differs from reviewed files')
        for name,digest in meta['active_before_sha256'].items():
            if name not in NAMES:require(sha(HERE/name)==digest,'unrelated publication asset changed')
        for source,target in [('preparation.json','writing_controls_preparation.json'),
                ('render.json','writing_controls_render.json'),('figure_review.json','writing_controls_figure_review.json'),
                ('build.json','writing_controls_build.json'),('review.json','writing_controls_review.json')]:
            atomic_copy(stage/source,HERE/'evidence'/target)
        receipt=dict(baseline_commit=BASE,promoted_files_sha256=promoted,
            preparation_sha256=sha(stage/'preparation.json'),build_receipt_sha256=sha(stage/'build.json'),
            review_receipt_sha256=sha(stage/'review.json'),all_prior_figure_bytes_preserved=True,
            prior_article_and_supplement_body_preserved=True,title_and_Introduction_preserved=True,
            abstract_updated=True,current_native_slides=48,current_native_charts=53,
            active_publication_updated=True,Nutstore_updated=False,general_ilm_goal_complete=False)
        save(stage/'promotion.json',receipt);save(HERE/'evidence/writing_controls_promotion.json',receipt)
        print('Promoted the reviewed48-slide / five-document writing-control package.')


def sync(stage):
    _,review=reviewed(stage);promotion=load(stage/'promotion.json');meta=load(stage/'preparation.json')
    for name,digest in promotion['promoted_files_sha256'].items():require(sha(HERE/name)==digest,'active file changed before sync')
    require(DEST.is_dir(),'existing authorized Nutstore ILM destination required')
    for name,alias in ALIASES.items():
        target=DEST/alias
        if target.exists():require(sha(target) in (meta['active_before_sha256'][name],sha(HERE/name)),
            'a different Nutstore revision exists; preserve it: '+alias)
    package=DEST/'ILM_NatureStyle_20261109_WritingControls';package.mkdir(exist_ok=False)
    files={}
    for path in sorted((stage/'paper').rglob('*')):
        if path.is_file():
            relative=path.relative_to(stage/'paper');atomic_copy(path,package/relative);files[str(relative)]=sha(package/relative)
    for name in ('build.json','review.json','promotion.json','render.json','figure_review.json'):
        atomic_copy(stage/name,package/'evidence'/name);files['evidence/'+name]=sha(stage/name)
    aliases={}
    for name,alias in ALIASES.items():
        target=DEST/alias
        if target.exists() and sha(target)!=sha(HERE/name):
            old=sha(target);backup=target.with_name(target.stem+'_before_writing_controls_'+old[:12]+target.suffix)
            if backup.exists():require(sha(backup)==old,'different alias backup exists')
            else:atomic_copy(target,backup)
        atomic_copy(HERE/name,target);aliases[alias]=sha(target)
        require(aliases[alias]==sha(HERE/name),'local alias hash differs')
    receipt=dict(package_folder=package.name,files_sha256=files,aliases_sha256=aliases,verified_local_copy=True,
        cloud_propagation_verified=False,PowerPoint_application_inspected=False,general_ilm_goal_complete=False)
    save(stage/'sync.json',receipt);save(HERE/'evidence/writing_controls_sync.json',receipt)
    print('Verified the local Nutstore package and five latest aliases; cloud propagation unverified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('promote','sync'),required=True)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.input.resolve())
