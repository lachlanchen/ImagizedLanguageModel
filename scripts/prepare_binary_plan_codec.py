#!/usr/bin/env python3
"""Bind a physical-only codec split without a neural or teacher call."""
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
REG=ROOT/'references/evidence/binary_codec_preparation_registration_20261118.json'
RUN=ROOT/'artifacts/binary_codec_preparation_v1'
SOURCE=ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt'
FILES=('scripts/prepare_binary_plan_codec.py','ilm/visual_lm/binary_plan_codec_learning.py',
       'ilm/visual_lm/binary_plan_glyph_density.py','tests/test_binary_plan_codec_learning.py',
       'references/binary_visual_plan_direction_20261118.md','references/binary_codec_preparation_protocol_20261118.md',
       'references/evidence/binary_codec_learning_cpu_20261118.json',
       'references/evidence/occupancy_density_structure_result_20261118.json',
       'artifacts/nonlinear_feedback_alignment_v1/prepared.pt','artifacts/nonlinear_feedback_alignment_v1/preparation.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only registration')
    old=original.load(ROOT/'references/evidence/global_form_preparation_registration_20261118.json')
    original.require(original.sha(SOURCE)==old['input_sha256'][str(SOURCE.relative_to(ROOT))],'Reuse exact original training images')
    original.write_new(REG,dict(schema='ilm-binary-codec-preparation-v1',input_sha256={n:original.sha(ROOT/n) for n in FILES},
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy']},source_cells=68948,
        distinct_source_span_images=29740,derived_native_rows=33941,
        split_rule='held when SHA256(mask) first eight bytes big-endian modulo ten is zero, except empty always trains',
        no_source_omissions=True,no_language_confirmation_access=True,maximum_seconds=180,maximum_rss_gib=2,
        cpu_threads=2,minimum_entry_host_gib=48,minimum_runtime_host_gib=24,neural_calls=0,optimizer_updates=0,
        full_language_fit_authorized=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','No CUDA during preparation')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in [str(REG.relative_to(ROOT)),*[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git','show','HEAD:'+name])==(ROOT/name).read_bytes(),'Commit bound sources first')
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before preparation')
    import torch
    from ilm.visual_lm.binary_plan_codec_learning import physical_partition
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2);RUN.mkdir();began=time.perf_counter();samples=[]
    try:
        for name,digest in reg['input_sha256'].items():original.require(original.sha(ROOT/name)==digest,'Bound source changed: '+name)
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        entry=host_resources();original.require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,'CPU preparation RAM/pressure entry')
        def guard():
            sample=host_resources();samples.append(sample)
            original.require(time.perf_counter()-began<180 and sample['available_gib']>=24,'Preparation deadline/RAM')
            original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Preparation RSS ceiling')
        with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
            data=torch.load(SOURCE,map_location='cpu',weights_only=True);packed=data['packed_ink']
            original.require(len(packed)==68948 and len(data['image_glyph_ends'])==29740 and len(data['row_image'])==33941,'Complete source support changed')
            held,hashes=physical_partition(packed);guard();groups=sorted(set(hashes));lookup={h:i for i,h in enumerate(groups)}
            group=torch.tensor([lookup[h] for h in hashes],dtype=torch.long)
            train_index=(~held).nonzero().flatten();held_index=held.nonzero().flatten()
            original.require(len(train_index)>0 and len(held_index)>0 and len(train_index)+len(held_index)==68948,'Complete partition coverage')
            train_groups=set(group[~held].tolist());held_groups=set(group[held].tolist())
            original.require(not train_groups&held_groups,'A repeated physical image crosses partitions')
            blank=~packed.bool().any(1);original.require(not bool(held[blank].any()),'Empty images must remain in training')
            path=RUN/'partition.pt';torch.save(dict(held=held,group=group,group_mask_sha256=groups,train_index=train_index,held_index=held_index),path)
            original.require(original.sha(SOURCE)==reg['input_sha256'][str(SOURCE.relative_to(ROOT))],'Source changed during split')
            guard();result=dict(schema='ilm-binary-codec-preparation-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                source_cells=68948,train_cells=len(train_index),held_cells=len(held_index),train_masks=len(train_groups),held_masks=len(held_groups),
                blank_training_cells=int(blank.sum()),all_source_cells_retained=True,exact_physical_aliases_in_one_partition=True,
                partition_file=str(path.relative_to(ROOT)),partition_sha256=original.sha(path),source_sha256=original.sha(SOURCE),
                scope='Physical reconstruction partition within original training-only support; not held-out language or source groups',
                neural_calls=0,optimizer_updates=0,gpu_used=False,full_language_fit_authorized=False,goal_achieved=False,
                resources=dict(elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    minimum_available_host_gib=min(s['available_gib'] for s in samples),cpu_threads=2))
            original.write_new(RUN/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
