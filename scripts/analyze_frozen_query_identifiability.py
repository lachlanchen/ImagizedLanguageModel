#!/usr/bin/env python3
"""CPU exact-query target floor on the completed full-support cache."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/nonlinear_feedback_alignment_v1'
OUT=ROOT/'references/evidence/frozen_query_identifiability_20261118.json'
PROTOCOL=ROOT/'references/frozen_query_identifiability_protocol_20261118.md'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only analysis')
    began=time.perf_counter()
    import torch
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from scripts.train_full_rank_visual_span import source_targets
    torch.set_num_threads(2)
    def guard():
        if time.perf_counter()-began>=60 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=2*1024**3:
            raise RuntimeError('Query analysis resource ceiling')
    reg_path=ROOT/'references/evidence/nonlinear_feedback_alignment_registration_20261118.json'
    reg=json.loads(reg_path.read_text());run_result=json.loads((SOURCE/'execution/result.json').read_text())
    for path,digest in [(SOURCE/'prepared.pt',reg['input_sha256'][str((SOURCE/'prepared.pt').relative_to(ROOT))]),
                        (SOURCE/'execution/frozen_queries.pt',run_result['features_sha256'])]:
        if sha(path)!=digest:raise ValueError('Registered cache changed')
    data=torch.load(SOURCE/'prepared.pt',weights_only=True,map_location='cpu')
    query=torch.load(SOURCE/'execution/frozen_queries.pt',weights_only=True,map_location='cpu')
    if query.dtype!=torch.bfloat16 or query.shape!=(29740,3,384):raise ValueError('Query cache shape changed')
    groups=defaultdict(list)
    for row,(image,slot) in enumerate(zip(data['row_image'].tolist(),data['row_slot'].tolist())):
        key=query[image,slot].view(torch.uint8).numpy().tobytes();groups[key].append(row)
    plan_path=ROOT/reg['teacher']/'preparation.json';plan=json.loads(plan_path.read_text())
    if sha(plan_path)!=reg['input_sha256'][str(plan_path.relative_to(ROOT))]:raise ValueError('Teacher preparation changed')
    archive=open_compact_archive(ROOT.parent/'LocalLLM');lookup,references=source_targets(archive,plan['training_native_units']);del archive
    total=duplicate_rows=conflicting_rows=0;mass=0.;max_disagreement=0.;records=[]
    for key,rows in groups.items():
        if len(rows)<2:continue
        guard();indices=torch.tensor(rows);units=data['row_native_id'][indices]
        value=references[lookup[units]].double();weights=data['row_weight'][indices]
        coefficient=weights/value.square().sum(-1).clamp_min(1e-8)
        center=(coefficient[:,None]*value).sum(0)/coefficient.sum()
        direct=(coefficient*(value-center).square().sum(-1)).sum()
        moment=(coefficient*value.square().sum(-1)).sum()-(coefficient[:,None]*value).sum(0).square().sum()/coefficient.sum()
        disagreement=float(abs(direct-moment));max_disagreement=max(max_disagreement,disagreement)
        if disagreement>1e-12:raise ValueError('Moment/direct floor mismatch')
        total+=float(direct);duplicate_rows+=len(rows);mass+=float(weights.sum())
        target_conflict=not bool((value==value[0]).all())
        if target_conflict:conflicting_rows+=len(rows)
        image_ids=data['row_image'][indices].tolist();slots=data['row_slot'][indices].tolist()
        records.append(dict(query_sha256=hashlib.sha256(key).hexdigest(),rows=rows,
            image_indices=image_ids,slots=slots,native_units=units.tolist(),mass=float(weights.sum()),
            loss_floor=float(direct),target_conflict=target_conflict,
            same_image_cross_slot=len(set(image_ids))==1 and len(set(slots))>1,
            cross_image=len(set(image_ids))>1,
            encoding_only=len(set(image_ids))==1 and len(set(slots))==1))
    records.sort(key=lambda r:(-r['loss_floor'],r['query_sha256']))
    private=SOURCE/'exact_query_groups_20261118.json'
    with private.open('x') as stream:json.dump(records,stream,indent=2);stream.write('\n')
    guard()
    bindings={str(p.relative_to(ROOT)):sha(p) for p in [reg_path,SOURCE/'prepared.pt',SOURCE/'execution/result.json',
        SOURCE/'execution/frozen_queries.pt',plan_path,private,Path(__file__),PROTOCOL]}
    result=dict(schema='ilm-frozen-query-identifiability-v1',rows=len(data['row_image']),unique_query_inputs=len(groups),
        duplicate_groups=len(records),duplicate_rows=duplicate_rows,duplicate_mass=mass,
        conflicting_groups=sum(r['target_conflict'] for r in records),conflicting_rows=conflicting_rows,
        same_image_cross_slot_groups=sum(r['same_image_cross_slot'] for r in records),
        cross_image_groups=sum(r['cross_image'] for r in records),encoding_only_groups=sum(r['encoding_only'] for r in records),
        exact_query_relative_error_floor=total,gate_010_ruled_out_by_exact_query_identity=total>=.10,
        maximum_direct_moment_disagreement=max_disagreement,largest_ten_groups=records[:10],bindings=bindings,
        elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        neural_calls=0,language_measured=False,originals_changed=False,full_language_fit_authorized=False,goal_achieved=False)
    with OUT.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['bindings','largest_ten_groups']}))


if __name__=='__main__':main()
