#!/usr/bin/env python3
"""Recount complete balanced-writer answers and all216 actual initial states."""
import json
import os
from pathlib import Path
import subprocess

import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.compare_balanced_writer_matched import RUN,validate,load,CHECKPOINT_SHA
from scripts.review_balanced_writer_matched import authenticate_review,CONTROLS,EVALUATIONS
from scripts.summarize_glyph_context_motor import recount
from scripts.summarize_compact_visual_span_learning import metric_rows,qa_values,clustered_interval
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/summarize_balanced_writer_matched.py'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only completed-output analysis')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
        'Commit analysis before inspecting reference scores')
    torch.set_num_threads(2);r=validate(RUN);review=RUN/'review'
    mapping,manual=authenticate_review(RUN,review/'readings.json')
    comparison=load(review/'comparison.json');references={x['id']:x for x in load(EVALUATIONS['fixed_cells']/'references.json')['reference_rows']}
    states=[];summaries={};intervals={};differences={}
    for layout in ['fixed_cells','proportional']:
        scored=load(review/layout/'scores.json');actual=[dict(row,step=18000) for row in recount(scored['records'],references)]
        require(actual==scored['results']==comparison['results'][layout],'Independent aggregate recount differs')
        summaries[layout]=actual;intervals[layout]=metric_rows(scored['records'],references,18000);differences[layout]={}
        for name,path in [('parent',CONTROLS[layout]/'review/scores.json'),
                          ('original_flow',ROOT/'artifacts/glyph_context_motor_v1/review'/layout/'scores.json')]:
            control=load(path);d=[]
            for paired in [False,True]:
                new=qa_values(scored['records'],references,'normal',paired)
                old=qa_values(control['records'],references,'normal',paired)
                d.append(dict(metric='both_correct' if paired else 'complete_exact',
                    comparison='balanced_minus_'+name,**clustered_interval(new-old)))
            differences[layout][name]=d
        prior=load(CONTROLS[layout]/'generation/report.json');by_id={x['id']:x for x in prior['records']}
        for row in [x for x in r['inputs'] if x['layout']==layout]:
            old_dir=CONTROLS[layout]/'generation'/row['original_case'];new_dir=RUN/('generation_'+layout)/row['id']
            original=by_id[row['original_case']]
            require(original['input_sha256']==row['sha256'] and
                original['legacy_output_sha256']==sha(old_dir/'legacy_output.pt'),'Original parent binding changed')
            old=torch.load(old_dir/'legacy_output.pt',map_location='cpu',weights_only=True)
            new=torch.load(new_dir/'legacy_output.pt',map_location='cpu',weights_only=True)
            states.append(dict(case=row['id'],old_tensor_sha256=sha(old_dir/'legacy_output.pt'),
                new_tensor_sha256=sha(new_dir/'legacy_output.pt'),
                initial_context_state_bit_exact=torch.equal(old['states'][0],new['states'][0]),
                initial_state_max_absolute_difference=float((old['states'][0]-new['states'][0]).abs().max())))
    require(len(states)==216,'Missing initial-state comparison')
    save_json(RUN/'initial_state_comparison.json',dict(records=states))
    generations={layout:load(RUN/('generation_'+layout)/'report.json') for layout in summaries}
    value=dict(schema='balanced-writer-matched-language-v1',source_sha256=sha(ROOT/OWN),
        registration_sha256=sha(RUN/'registration.json'),checkpoint_sha256=CHECKPOINT_SHA,
        results=summaries,pointwise_passage_group_intervals=intervals,candidate_minus_controls_intervals=differences,
        original_exact_retention=comparison['original_exact_retention'],flow_exact_retention=comparison['flow_exact_retention'],
        union_exact_retention=comparison['union_exact_retention'],prospective_development_pilot_passed=comparison['prospective_development_pilot_passed'],
        all216_initial_context_states_bit_exact_to_parent=all(x['initial_context_state_bit_exact'] for x in states),
        initial_state_comparison_sha256=sha(RUN/'initial_state_comparison.json'),
        independent_per_answer_and_aggregate_recount_passed=True,actual_outputs=216,
        clear_whole_readings=sum(not x['uncertain'] for x in manual['readings'].values()),
        uncertain_whole_readings=sum(x['uncertain'] for x in manual['readings'].values()),
        opaque_boards_viewed=54,book_prefixes_viewed=12,readings_sha256=sha(review/'readings.json'),
        reading_audit_sha256=sha(review/'reading_audit.json'),comparison_sha256=sha(review/'comparison.json'),
        generation_report_sha256={layout:sha(RUN/('generation_'+layout)/'report.json') for layout in generations},
        generation_resources={layout:{k:v[k] for k in ['elapsed_seconds','actor_loading_seconds','total_generation_seconds',
            'resource_wait_seconds','peak_allocated_vram_gib','minimum_available_gib']} for layout,v in generations.items()},
        inherited_visual_contextual_parameters=7522463984,trained_readout_and_stop_bundle=True,new_optimizer_updates=0,
        reused_development_cohort=True,untouched_confirmation_complete=False,independent_human_adjudication=False,
        model_reviewer_prior_cohort_familiarity=True,strict_canonical_mismatch_not_semantic_falsity=True,
        new128glyph_book_cold_replay_not_run=True,earlier_different_cohort64glyph_replay_not_substituted=True,
        uncertainty='Pointwise95% percentile bootstrap,10000draws/seed202611154,16paired passages; reused development, no independent seed or simultaneous coverage. Twelve book prefixes from two works have no generalization interval.',
        generalized_language_learning_proven=False,task_efficiency_advantage_proven=False,goal_achieved=False)
    save_json(RUN/'learning_summary.json',value)
    save_json(ROOT/'references/evidence/balanced_writer_matched_learning_20261111.json',value)
    print(json.dumps(dict(results=summaries,all216_initial_states_exact=value['all216_initial_context_states_bit_exact_to_parent'],
        pilot_passed=value['prospective_development_pilot_passed'],goal_achieved=False)))


if __name__=='__main__':main()
