#!/usr/bin/env python3
"""Bind completed history/preflight measurements without new model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import prepare_realized_writer_evidence as previous

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASELINE='d1f1d190cff5b151441e417ee0bd953052c49e55'
SOURCES=[
    'references/evidence/reference_image_history_learning_20261112.json',
    'references/evidence/reference_image_history_review_20261112.json',
    'references/evidence/visual_continuation_preflight_20261112/execution_report.json',
    'references/evidence/visual_continuation_preflight_20261112/audit.json',
    'references/evidence/visual_continuation_preflight_registration_20261112.json',
    'references/evidence/visual_continuation_full_logger_recovery_registration_20261112.json',
    'references/evidence/visual_continuation_full_launch_20261112.json']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def summarize(positions):
    values=[]
    for p in positions:
        tp,fp,fn=p['tp'],p['fp'],p['fn'];den=2*tp+fp+fn
        f1=2*tp/den if den else 1.
        require(f1==p['foreground_f1'] and bool(not(fp or fn))==p['pixel_exact'],
                'Literal foreground counts disagree with stored metrics')
        require(p['target_ink']==tp+fn and p['predicted_ink']==tp+fp,'Ink counts differ')
        values.append(f1)
    return dict(positions=len(values),exact=sum(p['pixel_exact'] for p in positions),
        mean_f1=float(np.mean(values)),median_f1=float(np.median(values)),
        at_least_095=sum(v>=.95 for v in values))


def main(stage):
    require(not stage.exists(),'Preserve previous publication preparation')
    evidence=[load(ROOT/n) for n in SOURCES];history,review,report,audit,registration,full,launch=evidence
    require(review['learning_sha256']==sha(ROOT/SOURCES[0]) and review['all16boards_actually_viewed']
        and review['all8561_displayed_original_tiles_independently_verified'],'Completed original review required')
    require(history['all64_initial_states_bit_exact'] and history['all64_autonomous_pixel_and_stop_replays_exact']
        and history['original_complete_language_scores_unchanged'],'Preserve state/replay and language boundaries')
    require(len(history['records'])==64 and history['supplied_target_length'] and history['reference_conditioned'],
        'This is a complete supplied-reference diagnostic')
    groups={};missing={}
    for layout in ['fixed_cells','proportional']:
        records=[r for r in history['records'] if r['job']['layout']==layout]
        require(len(records)==32,'All32 cases required')
        pools={k:[] for k in ['first','later','all','autonomous_shared','forced_shared']}
        for record in records:
            forced=record['canonical_forced_positions'];auto=record['canonical_autonomous_shared_positions']
            require(len(auto)<=len(forced),'Shared indices exceed supplied target')
            for key,values in [('first',forced[:1]),('later',forced[1:]),('all',forced),
                               ('autonomous_shared',auto),('forced_shared',forced[:len(auto)])]:
                pools[key].extend(values)
        groups[layout]={key:summarize(values) for key,values in pools.items()}
        require(groups[layout]==history['groups'][layout],'Complete position aggregation differs')
        require(groups[layout]['all']['positions']==745 and groups[layout]['later']['positions']==713,
                'Retain complete reference boundaries')
        missing[layout]=groups[layout]['all']['positions']-groups[layout]['autonomous_shared']['positions']
    require(not any(history['later_physical_gates'].values()),'Both recorded gates fail')
    require(report['all_checks_passed'] and report['actual_optimizer_updates']==4
        and report['all36_answer_layers_have_real_gradients'],'Actual full-body preflight required')
    require(audit['complete_target_positions']==772 and audit['optimizer_parameters']==85
        and audit['actual_Adam_steps_per_parameter']==4,'Actual optimizer audit differs')
    require(audit['execution_report_sha256']==sha(ROOT/SOURCES[2]) and
        audit['registration_sha256']==sha(ROOT/SOURCES[4]),'Preflight source identity differs')
    require(not report['language_quality_evaluated'] and not report['carry_preflight_updates_into_full_fit'],
        'Preflight is not autonomous language evidence')
    require(full['training_cases']==13766 and full['total_steps']==41298 and
        full['scheduled_target_exposure']==6108741 and not launch['language_evaluated'],
        'Retain registered full fit and unmeasured language')
    pdfs={n:sha(HERE/n) for n in ['manuscript.pdf','manuscript_review.pdf','redlined_manuscript.pdf']}
    old=load(HERE/'evidence/realized_writer_publication_build.json')['files_sha256']
    require(all(d==old[n] for n,d in pdfs.items()),'Previous article PDF changed before staging')
    previous.BASELINE=BASELINE;preserved=previous.preserve();stage.mkdir(parents=True)
    measured=dict(schema='continuation-history-publication-evidence-v1',baseline_commit=BASELINE,
        baseline_PDF_sha256=pdfs,source_sha256={n:sha(ROOT/n) for n in SOURCES},
        extractor_sha256=sha(Path(__file__)),history_groups=groups,
        missing_autonomous_target_positions=missing,stopping=history['stopping'],review_groups=review['groups'],
        later_physical_gates=history['later_physical_gates'],all_literal_position_metrics_recounted=True,
        positions_recounted_from_public_integer_records=True,new_raw_pixel_recount=False,
        preflight_cases=audit['cases'],preflight_resources=audit['resources'],preflight_checks=report['checks'],
        zero_adapter_native_split_max_abs=report['zero_adapter_native_split_max_abs'],
        trainable_parameters=report['trainable_parameters'],new_residual_parameters=report['new_residual_parameters'],
        query_parameters=report['query_parameters'],inherited_body_parameters=report['inherited_visual_contextual_parameters'],
        preflight_optimizer_updates=4,preflight_optimizer_tensors=85,
        residual_down_initial_tensors_not_archived=audit['residual_down_initial_tensors_not_archived'],
        full_fit=dict(training_cases=13766,total_steps=41298,snapshot_steps=full['snapshot_steps'],
            target_positions_per_epoch=full['target_positions_per_epoch'],target_exposure=6108741,
            prescribed_output_streams=732,language_quality_unmeasured=True),
        semantic_alignment_claimed=False,new_model_calls=0,new_optimizer_updates=0,
        original_complete_language_scores_unchanged=True,model_reviewer_prior_cohort_familiarity=True,
        independent_human_adjudication=False,goal_achieved=False)
    save(stage/'measured_data.json',measured);save(stage/'preserved_assets.json',preserved)
    print(json.dumps(dict(stage=str(stage.relative_to(ROOT)),preserved_assets=len(preserved['files_sha256']),
        reference_positions=1490,shared_indices={k:v['forced_shared']['positions'] for k,v in groups.items()},
        preflight_positions=772,new_model_calls=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,default=ROOT/'artifacts/continuation_history_publication_v1')
    main(parser.parse_args().stage)
