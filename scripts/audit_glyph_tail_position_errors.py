#!/usr/bin/env python3
"""Training-only first-decision diagnosis; no model updates or held-out references."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from PIL import Image

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/audit_glyph_tail_position_errors.py'


def load(path): return json.loads(Path(path).read_text())


def summarize(rows):
    return dict(cases=len(rows),mean_f1=statistics.mean(r['f1'] for r in rows) if rows else None,
        canonical_exact=sum(r['exact'] for r in rows),
        fraction_f1_at_least095=sum(r['f1']>=.95 for r in rows)/len(rows) if rows else None)


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','training-record audit requires hidden CUDA')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),'commit diagnosis first')
    require(not output.exists(),'preserve the previous diagnosis')
    require(load(run/'qualify/status.json')['phase']=='completed','finish qualification first')
    m=load(run/'preparation.json');q=load(run/'qualify/report.json')
    rows=load(run/'qualify/records.json')['training'];jobs=m['training_jobs']
    require(len(rows)==sum(len(j['labels']) for j in jobs),'incomplete training-prediction cohort')
    require(abs(statistics.mean(r['f1'] for r in rows)-q['training_mean_f1'])<1e-12,'aggregate differs from qualification')
    by_case=defaultdict(list)
    for r in rows:
        require(jobs[r['case']]['labels'][r['position']]==r['label'],'case/position/label mapping differs')
        by_case[r['case']].append(r)
    require(len(by_case)==len(jobs),'missing training question')
    first=[r for r in rows if r['position']==0];later=[r for r in rows if r['position']>0]
    frequency=Counter(j['labels'][0] for j in jobs)
    frequency_bins={name:summarize([r for r in first if predicate(frequency[r['label']])]) for name,predicate in [
        ('one',lambda n:n==1),('two_to_four',lambda n:2<=n<=4),
        ('five_to_nineteen',lambda n:5<=n<=19),('twenty_or_more',lambda n:n>=20)]}
    positions={str(i):summarize([r for r in rows if r['position']==i]) for i in sorted({r['position'] for r in rows})}
    glyphs=defaultdict(lambda:dict(first=[],later=[]))
    for r in rows:glyphs[r['label']]['first' if r['position']==0 else 'later'].append(r)
    matched=[]
    for label,groups in sorted(glyphs.items()):
        if groups['first'] and groups['later']:
            a,b=summarize(groups['first']),summarize(groups['later'])
            matched.append(dict(label=label,first=a,later=b,later_minus_first_f1=b['mean_f1']-a['mean_f1']))
    images=defaultdict(list)
    for index,job in enumerate(jobs):
        path=ROOT/job['prompt']
        require(sha(path)==m['bindings_sha256'][job['prompt']],'training prompt changed')
        with Image.open(path) as im:
            physical=im.convert('L');h=hashlib.sha256()
            h.update(str(physical.size).encode());h.update(physical.tobytes())
        images[h.hexdigest()].append(index)
    conflicts=[];first_max=sequence_max=0
    for digest,indexes in images.items():
        initial=Counter(jobs[i]['labels'][0] for i in indexes)
        sequences=Counter(tuple(jobs[i]['labels']) for i in indexes)
        first_max+=max(initial.values());sequence_max+=max(sequences.values())
        if len(initial)>1 or len(sequences)>1:
            conflicts.append(dict(input_pixel_sha256=digest,training_indices=indexes,
                distinct_first_target_forms=len(initial),distinct_target_sequences=len(sequences)))
    save_json(output,dict(source_sha256=sha(ROOT/OWN),preparation_sha256=sha(run/'preparation.json'),
        qualification_report_sha256=sha(run/'qualify/report.json'),training_records_sha256=sha(run/'qualify/records.json'),
        first=summarize(first),later=summarize(later),positions=positions,first_target_frequency_bins=frequency_bins,
        distinct_initial_forms=len(frequency),initial_forms_with_one_training_example=sum(n==1 for n in frequency.values()),
        same_form_first_and_later=matched,
        same_form_macro_later_minus_first_f1=statistics.mean(r['later_minus_first_f1'] for r in matched),
        unique_physical_training_inputs=len(images),training_questions=len(jobs),conflicting_input_groups=conflicts,
        empirical_deterministic_first_form_exact_ceiling=first_max/len(jobs),
        empirical_deterministic_full_sequence_exact_ceiling=sequence_max/len(jobs),
        question_optimizer_updates=13500,mean_question_visits=13500/len(jobs),
        optimizer_source_counter_audit_required_for_update_count=True,
        later_states_receive_reference_images_not_autonomous_outputs=True,
        frequency_strata_and_same_form_differences_are_descriptive_not_causal=True,
        first_decision_occurs_before_generated_image_feedback=True,
        frozen_state_information_limit_or_learning_algorithm_impossibility_proven=False,
        new_reference_strings_read=False,neural_forward_calls=0,optimizer_updates=0,candidate_changed=False))
    print(json.dumps(dict(first=summarize(first),later=summarize(later),initial_forms=len(frequency),
        single_example_initial_forms=sum(n==1 for n in frequency.values()),input_conflicts=len(conflicts),
        same_form_macro_later_minus_first=statistics.mean(r['later_minus_first_f1'] for r in matched)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run.resolve(),a.output.resolve())
