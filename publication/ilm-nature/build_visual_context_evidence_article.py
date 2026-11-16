#!/usr/bin/env python3
"""Build completed evidence beside the current, separately owned context fit."""
import argparse
import fcntl
import json
import shutil
import build_ranking_completed_article as base
from prepare_visual_context_evidence import ROOT,HERE,STAGE,protected_sources
from prepare_glyph_score_gap_figure import owners

RECEIPT=HERE/'evidence/visual_context_evidence_article_build.json'
base.STAGE=STAGE
base.FIGURES=[*base.FIGURES,
    ('Figure_28_glyph_score_gaps','glyph_score_gap_caption.tex','fig:glyph-score-gaps'),
    ('Figure_29_reward_calibration','reward_calibration_caption.tex','fig:reward-calibration'),
    ('Figure_30_white_target_completed','white_target_completed_caption.tex','fig:white-target-completed'),
    ('Figure_31_visual_context_evidence','visual_context_completed_caption.tex','fig:visual-context-evidence')]
original_literal=base.literal_figures
def literal_figures():return original_literal().replace('build_ranking_completed_article.py','build_visual_context_evidence_article.py')
base.literal_figures=literal_figures


def promote():
    build=base.load(STAGE/'build_receipt.json');inspection=base.load(STAGE/'inspection.json');review=base.load(STAGE/'visual_review.json')
    assert review['complete'] and not review['Office_application_export_verified']
    for name,info in inspection['documents'].items():
        seen=review['documents'][name]
        assert seen['pdf_sha256']==info['pdf_sha256'] and seen['overview_pages']==list(range(1,info['pages']+1)) and seen['detailed_pages']
    for name,digest in build['source_sha256'].items():assert base.sha(HERE/name)==digest
    protected=owners();bindings=protected_sources();base.preserved()
    for folder,name in base.DOCUMENTS:
        source=STAGE/folder/(name+'.pdf');assert base.sha(source)==inspection['documents'][name]['pdf_sha256']
        target=HERE/(name+'.pdf');temporary=target.with_suffix('.promoting.pdf');shutil.copy2(source,temporary);temporary.replace(target)
    shutil.copy2(STAGE/'redlines/redlined_manuscript/redlined_manuscript.tex',HERE/'redlined_manuscript.tex')
    protected_sources();assert owners()==protected
    build.update(promoted=True,visual_review_complete=True,inspection=inspection,visual_review=review,
        immediate_baseline=base.load(STAGE/'before_manifest.json'),active_article_updated=True,
        publication_new_optimizer_updates=0,publication_new_model_calls=0,
        white_target_physical_gate_passed=False,white_target_language_unmeasured=True,
        current_context_language_unmeasured=True,protected_model_bindings=bindings,
        current_model_ownership_verified_before_and_after_build=True,builder_with_inspection_sha256=base.sha(__file__))
    build.pop('protected_model_processes')
    for child in build['costs']['children']:
        child['command']=[('python' if x==base.os.sys.executable else x.replace(str(ROOT)+'/', '')) for x in child['command']]
    build['compiled_previous_redline_source_sha256']=build['source_sha256'].pop('redlined_manuscript.tex')
    build['new_redline_source_sha256']=base.sha(HERE/'redlined_manuscript.tex')
    with RECEIPT.open('x') as f:json.dump(build,f,indent=2);f.write('\n')
    print('Promoted completed evidence with reviewed native Figures30--31.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-literal',action='store_true');p.add_argument('--inspect',action='store_true');p.add_argument('--promote',action='store_true');p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    if a.attempt>1:
        previous=STAGE;STAGE=STAGE.parent/f'article_attempt{a.attempt:02}';base.STAGE=STAGE
        if not any((a.write_literal,a.inspect,a.promote)):
            STAGE.mkdir(exist_ok=False);shutil.copytree(previous/'before',STAGE/'before')
            shutil.copy2(previous/'before_manifest.json',STAGE/'before_manifest.json')
            cost_source=STAGE.parent/f'article_attempt{a.attempt-1:02}'
            costs=base.load(cost_source/'child_costs.json')
            if (cost_source/'inspection.json').exists():
                costs['prior_inspection_seconds']=costs.get('prior_inspection_seconds',0.)+base.load(cost_source/'inspection.json')['inspection_seconds']
            (STAGE/'child_costs.json').write_text(json.dumps(costs,indent=2)+'\n')
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);protected=owners();protected_sources()
        if a.write_literal:(HERE/'complete_continuation_figures.tex').write_text(literal_figures())
        elif a.inspect:base.inspect()
        elif a.promote:promote()
        else:
            base.build();value=base.load(STAGE/'build_receipt.json')
            value['revision_entrypoint_sha256']=base.sha(__file__)
            (STAGE/'build_receipt.json').write_text(json.dumps(value,indent=2)+'\n')
        protected_sources();assert owners()==protected
