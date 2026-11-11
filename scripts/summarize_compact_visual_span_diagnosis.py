#!/usr/bin/env python3
"""Authenticate raw diagnostic pixels and separate first-span/history errors."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.diagnose_compact_visual_span import ARMS, DATA, digest_tensor, image_metrics
from scripts.infer_compact_visual_span import read_image


def load(path):return json.loads(Path(path).read_text())


def totals(rows):
    keys=('glyphs','exact_glyphs','tp','fp','fn')
    value={k:sum(row[k] for row in rows) for k in keys}
    value['foreground_f1']=2*value['tp']/max(2*value['tp']+value['fp']+value['fn'],1)
    value['exact_glyph_rate']=value['exact_glyphs']/max(value['glyphs'],1)
    value['records']=len(rows)
    value['mean_record_foreground_f1']=float(np.mean([r['foreground_f1'] for r in rows])) if rows else None
    return value


def analyze(run,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():
        raise ValueError('Analyze completed diagnostics on CPU into a new directory')
    torch.set_num_threads(2)
    registration=load(run/'registration.json');report=load(run/'execution/report.json')
    geometry=load(run/'feedback_geometry.json')
    if report['registration_sha256']!=sha(run/'registration.json') or report['optimizer_updates']!=0:
        raise ValueError('Execution does not match the frozen-weight request')
    if load(run/'execution/status.json')['phase']!='complete':raise ValueError('Diagnostic execution incomplete')
    if sha(DATA/'report.json')!=registration['corpus_report_sha256']:raise ValueError('Corpus changed')
    corpora={split:VisualSequenceCorpus(DATA,split=split) for split in ('train','development')}
    expected={row['id']:row for row in registration['cases']}
    groups=defaultdict(list);reader_groups=defaultdict(list);sensitivity_groups=defaultdict(list)
    case_results=[];bindings={};exemplars=[]
    for model in ('v1','wider'):
        records=load(run/'execution'/f'{model}_records.json')
        if len(records)!=32 or {r['id'] for r in records}!=expected.keys():raise ValueError('Missing cases')
        floor={r['case']:r for r in geometry[model]['cases']}
        bindings[f'{model}_records.json']=sha(run/'execution'/f'{model}_records.json')
        fixed_cases=set()
        for row in records:
            key=row['id'];case=expected[key]
            if any(row[k]!=v for k,v in case.items()):raise ValueError('Case metadata changed')
            original=corpora[case['split']][case['record_index']]
            if digest_tensor(original['ink'])!=case['ink_sha256']:raise ValueError('Original diagnostic pixels changed')
            prefix,stop=case['prefix_spans'],case['stop_span']
            ends=original['span_glyph_ends'].tolist();start=ends[prefix-1];end=ends[stop-1]
            gold=original['ink'][start:end]
            first=ends[prefix]-start
            directory=run/'execution'/model/key
            if not torch.equal(read_image(directory/'reference.png',len(gold)),gold):raise ValueError('Reference PNG differs')
            if not torch.equal(read_image(directory/'input.png',start),original['ink'][:start]):raise ValueError('Prefix differs')
            codec=read_image(directory/'codec.png',len(gold))
            cm=image_metrics(codec,gold)
            if cm!=row['codec']:raise ValueError('Codec metrics differ from actual pixels')
            groups[model,case['split'],case['kind'],'codec','all'].append(cm)
            rr=row['reader']
            if rr['feedback_relative_mse']+1e-5<floor[key]['minimum_relative_error_mean']:
                raise ValueError('Actual reader error violates the computed affine bound')
            reader_groups[model,case['split'],case['kind']].append({**rr,**floor[key]})
            arrays={};case_arms={}
            for arm in ARMS:
                saved=row['arms'][arm]
                if sha(directory/(arm+'.png'))!=saved['png_sha256'] or sha(directory/(arm+'.npz'))!=saved['raw_sha256']:
                    raise ValueError('Original generated diagnostic output changed')
                with np.load(directory/(arm+'.npz'),allow_pickle=False) as data:
                    ink=torch.from_numpy(data['ink'].copy());latent=data['latent'].copy()
                if ink.shape!=gold.shape or latent.shape!=(len(gold),768) or not np.isfinite(latent).all():
                    raise ValueError('Invalid raw diagnostic endpoint')
                if not torch.equal(read_image(directory/(arm+'.png'),len(gold)),ink):raise ValueError('PNG differs from raw ink')
                measured=image_metrics(ink,gold)
                if any(saved[k]!=v for k,v in measured.items()):raise ValueError('Pixel metrics differ')
                offsets=np.cumsum([0]+[t['glyphs'] for t in saved['trace']])
                if offsets[-1]!=len(gold) or len(saved['trace'])!=case['target_spans']:raise ValueError('Trace does not cover targets')
                for j,event in enumerate(saved['trace']):
                    if event['span']!=prefix+j or digest_tensor(ink[offsets[j]:offsets[j+1]])!=event['ink_sha256']:
                        raise ValueError('Trace differs from emitted pixels')
                case_arms[arm]={'all':measured,'first':image_metrics(ink[:first],gold[:first]),
                                'later':image_metrics(ink[first:],gold[first:]) if first<len(gold) else None,
                                'reported_relative_latent_mse':saved['relative_latent_mse'],
                                'predicted_length_correct':saved['predicted_length_correct'],'target_spans':case['target_spans']}
                arrays[arm]=(ink.numpy(),latent)
                for part in ('all','first','later'):
                    value=case_arms[arm][part]
                    if value is not None:groups[model,case['split'],case['kind'],arm,part].append(value)
            for j in range(case['target_spans']):
                if len({row['arms'][arm]['trace'][j]['noise_sha256'] for arm in ARMS})!=1:
                    raise ValueError('Initial noise differs across arms')
            base_ink,base_z=arrays['visual_reference_8'];wrong_ink,wrong_z=arrays['shuffled_reference_8']
            sensitivity=dict(case=key,changed_pixel_fraction=float(np.mean(base_ink!=wrong_ink)),
                             relative_endpoint_difference=float(np.mean((base_z-wrong_z)**2)/max(np.mean(base_z**2),1e-8)),
                             reference_minus_shuffled_f1=case_arms['visual_reference_8']['all']['foreground_f1']-case_arms['shuffled_reference_8']['all']['foreground_f1'],
                             fine_minus_coarse_f1=case_arms['visual_reference_64']['all']['foreground_f1']-case_arms['visual_reference_8']['all']['foreground_f1'],
                             reference_minus_emitted_f1=case_arms['visual_reference_8']['all']['foreground_f1']-case_arms['emitted_fixed_counts_8']['all']['foreground_f1'])
            sensitivity_groups[model,case['split'],case['kind']].append(sensitivity)
            case_results.append(dict(model=model,case=key,split=case['split'],kind=case['kind'],
                                     glyphs=len(gold),first_span_glyphs=first,arms=case_arms,sensitivity=sensitivity))
            # Display the first case of each split/kind in registered order.
            # This display rule was added during execution, not preregistered.
            bucket=(case['split'],case['kind'])
            if bucket not in fixed_cases:
                fixed_cases.add(bucket)
                exemplars.append((model,key,directory))
    metrics=[dict(model=k[0],split=k[1],kind=k[2],arm=k[3],part=k[4],**totals(rows)) for k,rows in sorted(groups.items())]
    readers=[];sensitivity=[]
    for key,rows in sorted(reader_groups.items()):
        positions=sum(r['update_rows'] for r in rows)
        readers.append(dict(model=key[0],split=key[1],kind=key[2],records=len(rows),native_positions=positions,
                            achieved_feedback_relative_mse=sum(r['feedback_relative_mse']*r['update_rows'] for r in rows)/positions,
                            affine_relative_mse_floor=sum(r['minimum_relative_error_sum'] for r in rows)/positions,
                            update_count_accuracy=sum(r['correct_counts'] for r in rows)/sum(r['spans'] for r in rows),
                            boundary_foreground_f1=2*sum(r['boundary_tp'] for r in rows)/max(sum(2*r['boundary_tp']+r['boundary_fp']+r['boundary_fn'] for r in rows),1),
                            exact_prompt_groupings=sum(r['grouping_exact'] for r in rows),
                            mean_image_vs_native_context_cosine=float(np.mean([r['image_vs_native_context_cosine'] for r in rows])),
                            mean_learned_vs_reference_context_cosine=float(np.mean([r['learned_vs_reference_context_cosine'] for r in rows]))))
    for key,rows in sorted(sensitivity_groups.items()):
        sensitivity.append(dict(model=key[0],split=key[1],kind=key[2],records=len(rows),
                                **{metric:float(np.mean([r[metric] for r in rows])) for metric in rows[0] if metric!='case'}))
    output.mkdir(parents=True)
    raw=[]
    for model,key,source in exemplars:
        dest=output/'raw_examples'/model/key;dest.mkdir(parents=True)
        names=['reference.png','input.png','codec.png',*[arm+'.png' for arm in ARMS]]
        for name in names:
            shutil.copyfile(source/name,dest/name)
            raw.append(dict(model=model,case=key,file=str((dest/name).relative_to(output)),sha256=sha(dest/name)))
    summary=dict(schema='compact-visual-span-failure-localization-analysis-v1',registration_sha256=sha(run/'registration.json'),
                 execution_report_sha256=sha(run/'execution/report.json'),record_files_sha256=bindings,
                 source_sha256=sha(Path(__file__)),geometry_sha256=sha(run/'feedback_geometry.json'),
                 records=case_results,metrics=metrics,reader_metrics=readers,sensitivity=sensitivity,raw_examples=raw,
                 example_selection='First registered case of each split/kind; deterministic analysis-time display rule, not preregistered examples',
                 original_pixel_metrics_independently_recomputed=True,original_noise_and_feedback_hashes_verified=True,
                 reported_latent_errors_independently_recomputed=False,
                 latent_error_limit='Target codec latents were not persisted separately; these values come from the bound neural execution, while pixel measurements are recomputed from original targets.',
                 all_reference_supplied_diagnostics=True,autonomous_language_scores_added=False,goal_achieved=False,
                 uncertainty='Descriptive, selected development diagnostics; no confidence interval or generalization claim.',
                 resources=report)
    save_json(output/'summary.json',summary)
    with (output/'pixel_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (output/'reader_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(readers[0]));w.writeheader();w.writerows(readers)
    print(json.dumps({'reader_metrics':readers,'whole_stream_metrics':[r for r in metrics if r['part']=='all'],
                      'sensitivity':sensitivity},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/compact_visual_span_diagnosis_v2')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/compact_visual_span_diagnosis_analysis_v2')
    args=parser.parse_args();analyze(args.run,args.output)
