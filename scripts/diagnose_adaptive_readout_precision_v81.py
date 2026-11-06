#!/usr/bin/env python3
"""Replay a failed execution check and retain its actual floating-point errors."""
import argparse
import fcntl
import json
from pathlib import Path
import re

import torch

from ilm.visual_lm.adaptive_image_continuation import AdaptiveImageLanguageModel
from ilm.visual_lm.native_image_continuation import NativeImagePrefix
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, sha, committed_identity
from scripts.train_adaptive_readout_v81 import (
    SEED, SOURCES, build, experiment_identity, select_arm, pool_for, update,
    batch, new_optimizer, restore_values, cases_and_data,
)


def compare(actual, reference):
    error = (actual-reference).abs().double()
    scale = torch.maximum(actual.abs(),reference.abs()).double()
    bound = 1e-4+1e-5*scale
    index = int(error.flatten().argmax())
    return dict(max_abs=float(error.max()),rms=float(error.square().mean().sqrt()),
        elements=error.numel(),elements_above_original_atol=int((error>1e-4).sum()),
        scale_at_max_abs_error=float(scale.flatten()[index]),
        max_scaled_error=float((error/bound).max()),
        elements_above_atol1e4_rtol1e5=int((error>bound).sum()))


@torch.no_grad()
def measure(model,pool):
    prefix,codes,_ = batch(pool,list(range(len(pool))),model.device)
    full = model.conditions(prefix,codes,checkpoint_layers=False)
    altered = codes.clone();altered[:,-1] = -altered[:,-1]
    future = model.conditions(prefix,altered,checkpoint_layers=False)
    states = dict(full=full.cpu(),future=future.cpu(),single=[],incremental=[])
    rows = []
    for i,(original,target) in enumerate(pool):
        own = original.to(model.device);target = target.to(model.device)
        single = model.conditions(own,target[None],checkpoint_layers=False)
        current = model.start_prefix(own) if isinstance(model,AdaptiveImageLanguageModel) else own
        sequence = [current.initial_state[:,0].float()]
        for block in target:
            hidden,values = model.continuation(model.embed_images(block[None,None]),current,
                checkpoint_layers=False,collect_cache=True)
            current = NativeImagePrefix(values,hidden[:,-1:],current.rotary_offsets+2,current.blocked_before)
            sequence.append(hidden[:,-1])
        incremental = torch.stack(sequence,dim=1)
        states['single'].append(single.cpu());states['incremental'].append(incremental.cpu())
        rows.append(dict(case=i,padding=compare(full[i,:len(single[0])],single[0]),
            incremental=compare(incremental,single)))
    return states,dict(future_max_abs=float((full[:,:-1]-future[:,:-1]).abs().max()),rows=rows,
        max_padding_abs=max(r['padding']['max_abs'] for r in rows),
        max_incremental_abs=max(r['incremental']['max_abs'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-run',type=Path,default=ROOT/'artifacts/adaptive_readout_v81_20261106')
    args = parser.parse_args();failed = args.failed_run.resolve();out = failed/'precision_diagnosis'
    original = json.loads((failed/'identity.json').read_text())
    failure = json.loads((failed/'failure.json').read_text())
    if failure['stage'] != 'preflight' or (failed/'preflight/report.json').exists() or out.exists():
        raise ValueError('one preserved failed preflight and a new diagnostic directory required')
    for name,digest in original['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('original failing implementation changed')
    source = ROOT/'artifacts/native_chinese_v80_20261106'
    identity,cases = experiment_identity(source)
    diagnostic = dict(**committed_identity((str(Path(__file__).relative_to(ROOT)),*SOURCES)),
        original_identity_sha256=sha(failed/'identity.json'),failure_sha256=sha(failed/'failure.json'))
    out.mkdir();write(out/'identity.json',diagnostic)
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from scripts.train_clean_corpus_v75 import setup
        device = setup()
        _,data,indices = cases_and_data()
        actor,_,image_state,adapters = build(source,device)
        pool = pool_for(actor,cases,data,indices,[0,2,3,17],out)
        results = {}
        for arm in ('control','adapted'):
            model = select_arm(actor,arm,image_state,adapters)
            optimizer = new_optimizer(model);sampler = ExactEpochSampler(4,SEED+1)
            generator = torch.Generator(device=device).manual_seed(SEED+2)
            snapshot = failed/'preflight'/arm/'resume_check.pt'
            value = torch.load(snapshot,map_location='cpu',weights_only=True)
            if value['update'] != 2 or value['arm'] != arm:
                raise ValueError('original second-update checkpoint required')
            restore_values(value,actor,optimizer,sampler,generator)
            replayed = update(model,optimizer,pool,sampler,generator,2,schedule=False)
            states,errors = measure(model,pool)
            torch.save(states,out/(arm+'_states.pt'))
            results[arm] = dict(errors,replayed_preflight_update=replayed,
                source_checkpoint_sha256=sha(snapshot),states_sha256=sha(out/(arm+'_states.pt')))
            del optimizer,value
        match = re.search(r'mismatch: \(([^)]*)\)',failure['error'])
        expected = [float(v.strip()) for v in match[1].split(',')]
        got = results['adapted']
        if [got['future_max_abs'],got['max_padding_abs'],got['max_incremental_abs']] != expected:
            raise AssertionError('original preflight mismatch did not replay exactly')
        write(out/'report.json',dict(identity=diagnostic,results=results,
            original_mismatch_exactly_reproduced=True,original_preflight_still_failed=True,
            diagnostic_combined_tolerance_not_a_language_quality_gate=True,
            production_training_updates=0,language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps({a:{k:v for k,v in r.items() if k in ('max_padding_abs','max_incremental_abs','future_max_abs','rows')} for a,r in results.items()},indent=2))


if __name__ == '__main__':
    main()
