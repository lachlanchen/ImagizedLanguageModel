#!/usr/bin/env python3
"""Seal actual visual review and describe common-prefix state divergence on all64 cases."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.diagnose_reference_image_history import RUN,load
from scripts.summarize_reference_image_history import PUBLIC,LEARNING
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/seal_reference_image_history.py'
REVIEW=ROOT/'references/evidence/reference_image_history_review_20261112.json'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only seal')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),'Commit independent seal before analysis')
    require(not REVIEW.exists(),'Preserve prior completed review')
    torch.set_num_threads(2)
    evidence=load(LEARNING);manifest=load(PUBLIC/'display_manifest.json');export=load(PUBLIC/'export.json')
    registration=load(RUN/'registration.json')
    require(sum(len(b['cells']) for b in manifest['boards'])==evidence['displayed_original_tiles']==8561
        and manifest['all_displayed_pixels_verified'], 'Displayed-tile coverage is not verified')
    visual=load(RUN/'visual_review.json')
    require(visual['learning_sha256']==sha(LEARNING) and visual['display_manifest_sha256']==sha(PUBLIC/'display_manifest.json'),
            'Actual review input binding differs')
    require(visual['viewed_boards']==[x['file'] for x in manifest['boards']] and len(visual['viewed_boards'])==16,
            'Every complete board must actually have been viewed')
    require(visual['model_reviewer_prior_cohort_familiarity'] and not visual['independent_human_adjudication'],
            'Disclose reviewer limits')
    require(export['learning_sha256']==sha(LEARNING),'Export analysis binding differs')
    for name,digest in export['files_sha256'].items():require(sha(PUBLIC/name)==digest,'Export file differs: '+name)
    rows=[]
    for record in evidence['records']:
        job=record['job'];raw=dict(np.load(PUBLIC/record['pixels_file']))
        trace_path=RUN/'execution'/job['id']/'trace.pt'
        require(sha(trace_path)==record['trace_sha256'],'Correct-history trace differs')
        trace=torch.load(trace_path,weights_only=True,map_location='cpu')
        require(sha(ROOT/job['original_output'])==registration['bindings_sha256'][job['original_output']],
                'Original autonomous state binding differs')
        original=torch.load(ROOT/job['original_output'],weights_only=True,map_location='cpu')
        n=min(len(raw['gold_ink']),len(raw['autonomous_ink']));prefix=0
        while prefix<n and np.array_equal(raw['gold_ink'][prefix],raw['autonomous_ink'][prefix]):prefix+=1
        require(torch.equal(trace['states'][:prefix+1],original['states'][:prefix+1]),
                'State diverged before the image history differed')
        overlap=min(len(trace['states']),len(original['states']))
        equal=(trace['states'][:overlap]==original['states'][:overlap]).all(1)
        changed=torch.nonzero(~equal).flatten().tolist()
        first_state=changed[0] if changed else None
        first_pixel=prefix if prefix<n else None
        if first_pixel is not None:
            require(first_state is None or first_state>first_pixel,'State changed before mismatching pixel was fed')
        forced=record['canonical_forced_positions'];auto=record['canonical_autonomous_shared_positions']
        rows.append(dict(id=job['id'],layout=job['layout'],reference_glyphs=len(forced),
            common_pixel_prefix=prefix,first_pixel_difference=first_pixel,first_state_difference=first_state,
            state_difference_immediately_after_first_pixel_difference=bool(first_pixel is not None and first_state==first_pixel+1),
            first_f1=forced[0]['foreground_f1'],first_pixel_exact=forced[0]['pixel_exact'],
            forced_case_mean_f1=float(np.mean([p['foreground_f1'] for p in forced])),
            autonomous_shared_case_mean_f1=float(np.mean([p['foreground_f1'] for p in auto])) if auto else None,
            forced_shared_case_mean_f1=float(np.mean([p['foreground_f1'] for p in forced[:len(auto)]])) if auto else None,
            native_complete_exact=job['native_complete_exact'],original_scores=job['original_scores']))
    groups={}
    for layout in ['fixed_cells','proportional']:
        group=[x for x in rows if x['layout']==layout];near=[x for x in group if x['first_f1']>=.95 and not x['first_pixel_exact']]
        groups[layout]=dict(cases=len(group),mean_of_forced_case_means=float(np.mean([x['forced_case_mean_f1'] for x in group])),
            first_pixel_exact=sum(x['first_pixel_exact'] for x in group),
            cases_with_pixel_difference=sum(x['first_pixel_difference'] is not None for x in group),
            immediate_state_divergences=sum(x['state_difference_immediately_after_first_pixel_difference'] for x in group),
            close_but_nonidentical_first_cases=[x['id'] for x in near],
            native_exact_cases=sum(x['native_complete_exact'] for x in group))
    value=dict(schema='reference-image-history-reviewed-v1',learning_sha256=sha(LEARNING),
        display_manifest_sha256=sha(PUBLIC/'display_manifest.json'),export_sha256=sha(PUBLIC/'export.json'),
        review_notes_sha256=sha(RUN/'visual_review.json'),source_sha256=sha(ROOT/OWN),
        all16boards_actually_viewed=True,all8561_displayed_original_tiles_independently_verified=True,
        all64_common_pixel_prefixes_have_bit_exact_context_states=True,groups=groups,per_case=rows,
        common_prefix_analysis_posthoc_descriptive=True,
        reference_conditioned_output_not_language_evidence=True,canonical_mismatch_not_semantic_error=True,
        alternate_wording_and_lengths_preserved=True,original_language_scores_unchanged=True,
        model_reviewer_prior_cohort_familiarity=True,independent_human_adjudication=False,
        new_model_calls=0,new_optimizer_updates=0,goal_achieved=False)
    save_json(REVIEW,value)
    print(json.dumps(dict(groups=groups,review_sha256=sha(REVIEW)),indent=2))


if __name__=='__main__':main()
