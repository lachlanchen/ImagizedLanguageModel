#!/usr/bin/env python3
"""Prepare training-only physical equivalence labels, without a model call."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import time

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
REG=ROOT/'references/evidence/global_form_preparation_registration_20261118.json'
RUN=ROOT/'artifacts/global_form_preparation_v1'
PREPARED=ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt'
CACHE=ROOT/'artifacts/layer_conditioned_writer_preflight_v1/conditional_training_cache.pt'
FILES=('scripts/prepare_global_form_supervision.py','ilm/visual_lm/global_form_supervision.py',
       'tests/test_global_form_supervision.py','references/global_visual_choice_direction_20261118.md',
       'references/global_form_preparation_protocol_20261118.md',
       'references/evidence/global_form_supervision_cpu_20261118.json',
       'artifacts/nonlinear_feedback_alignment_v1/prepared.pt',
       'artifacts/nonlinear_feedback_alignment_v1/preparation.json',
       'artifacts/layer_conditioned_writer_preflight_v1/conditional_training_cache.pt',
       'artifacts/layer_conditioned_writer_preflight_v1/result.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only registration')
    old=original.load(ROOT/'artifacts/layer_conditioned_writer_preflight_v1/result.json')
    original.require(original.sha(CACHE)==old['conditional_cache_sha256'],'Immutable target cache differs')
    original.write_new(REG,dict(schema='ilm-global-form-preparation-v1',input_sha256={n:original.sha(ROOT/n) for n in FILES},
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy']},source_cells=68948,
        distinct_source_span_images=29740,derived_native_rows=33941,target_positions=[512,5,35],
        label_rule='lexicographically sorted exact 128-byte masks; preserve all physical aliases including empty',
        no_target_omissions=True,maximum_seconds=180,maximum_rss_gib=2,cpu_threads=2,
        minimum_entry_host_gib=48,minimum_runtime_host_gib=24,neural_calls=0,optimizer_updates=0,
        full_language_fit_authorized=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','No CUDA context during preparation')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in [str(REG.relative_to(ROOT)),*[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git','show','HEAD:'+name])==(ROOT/name).read_bytes(),'Commit bound sources first')
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before source preparation')
    import torch
    from ilm.visual_lm.global_form_supervision import build_form_index,form_labels
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2);RUN.mkdir();began=time.perf_counter();samples=[]
    try:
        for name,digest in reg['input_sha256'].items():original.require(original.sha(ROOT/name)==digest,'Bound source changed: '+name)
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        entry=host_resources();original.require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,'Preparation RAM/pressure entry')
        def guard():
            sample=host_resources();samples.append(sample)
            original.require(time.perf_counter()-began<180 and sample['available_gib']>=24,'Preparation time/RAM reserve')
            original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Preparation RSS ceiling')
        with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
            data=torch.load(PREPARED,weights_only=True,map_location='cpu')
            original.require(len(data['packed_ink'])==68948 and len(data['image_glyph_ends'])==29740 and len(data['row_image'])==33941,'Full prepared source changed')
            index=build_form_index(data['packed_ink']);guard()
            original.require(bytes(128) in index and len(index)>1,'Empty/nonempty source support required')
            class_hashes=[hashlib.sha256(bits).hexdigest() for bits in index]
            records=torch.load(CACHE,weights_only=True,map_location='cpu');labels=[];rows=[]
            original.require(len(records)==3 and [len(r['ink']) for r in records]==reg['target_positions'],'Whole target records changed')
            for row in records:
                guard();target=form_labels(row['ink'],index);labels.append(target)
                rows.append(dict(case=row['case'],positions=len(target),distinct_target_forms=int(target.unique().numel()),
                    blank_positions=int((target==index[bytes(128)]).sum()),label_sha256=hashlib.sha256(target.numpy().tobytes()).hexdigest()))
            path=RUN/'labels.pt';torch.save(dict(labels=labels,class_mask_sha256=class_hashes,empty_class=index[bytes(128)]),path)
            for name in [str(PREPARED.relative_to(ROOT)),str(CACHE.relative_to(ROOT))]:
                original.require(original.sha(ROOT/name)==reg['input_sha256'][name],'Source changed during preparation')
            guard();result=dict(schema='ilm-global-form-preparation-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                source_cells=68948,classes=len(index),empty_class=index[bytes(128)],cases=rows,complete_target_positions=sum(len(v) for v in labels),
                labels_file=str(path.relative_to(ROOT)),labels_sha256=original.sha(path),all_targets_exactly_supported=True,
                no_unicode_or_teacher_labels=True,no_source_mutation=True,neural_calls=0,optimizer_updates=0,gpu_used=False,
                resources=dict(elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    minimum_available_host_gib=min(v['available_gib'] for v in samples),cpu_threads=2),
                full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
