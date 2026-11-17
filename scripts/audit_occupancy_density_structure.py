#!/usr/bin/env python3
"""Validate both physical factorizations on every saved training-support cell."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/occupancy_density_structure_v1'
REG=ROOT/'references/evidence/occupancy_density_structure_registration_20261118.json'
FILES=('ilm/visual_lm/occupancy_glyph_density.py','ilm/visual_lm/conditional_pixel_density.py',
       'tests/test_occupancy_glyph_density.py','scripts/audit_occupancy_density_structure.py',
       'references/occupancy_visual_density_direction_20261118.md',
       'references/evidence/occupancy_density_cpu_20261118.json',
       'artifacts/nonlinear_feedback_alignment_v1/prepared.pt',
       'artifacts/nonlinear_feedback_alignment_v1/preparation.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only structure registration')
    original.write_new(REG,dict(schema='ilm-occupancy-density-structure-v1',input_sha256={n:original.sha(ROOT/n) for n in FILES},
        complete_saved_cells=68948,distinct_source_span_images=29740,derived_native_rows=33941,
        modes=['flat','tree'],chunk_cells=256,maximum_seconds=180,cpu_threads=2,maximum_rss_gib=2,
        minimum_entry_host_gib=48,minimum_runtime_host_gib=24,neural_forward_calls=0,optimizer_updates=0,
        no_language_or_compression_claim=True,full_language_fit_authorized=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Never admit a CUDA context in this audit')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for n in [str(REG.relative_to(ROOT)),*[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Commit sources before audit')
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before audit')
    import fcntl
    import numpy as np
    import torch
    from ilm.visual_lm.occupancy_glyph_density import OccupancyGlyphDensity,encode_occupancy,decode_occupancy
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2);RUN.mkdir();began=time.perf_counter();samples=[]
    try:
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound structure source changed: '+n)
        entry=host_resources();original.require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,'Structure admission reserve/pressure')
        def guard():
            sample=host_resources();samples.append(sample)
            original.require(time.perf_counter()-began<180 and sample['available_gib']>=24,'Structure time/RAM reserve')
            original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Structure RSS ceiling')
        with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
            data=torch.load(ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt',weights_only=True,map_location='cpu')
            packed=data['packed_ink'];original.require(len(packed)==68948 and len(data['image_glyph_ends'])==29740 and len(data['row_image'])==33941,'Complete saved support differs')
            counters={m:dict(cells=0,empty_cells=0,total_events=0,length_histogram=torch.zeros(343,dtype=torch.long),
                            depth_counts=torch.zeros(6,dtype=torch.long),mask_counts_by_depth=torch.zeros(6,16,dtype=torch.long),decoded_sha256=hashlib.sha256()) for m in reg['modes']}
            for start in range(0,len(packed),reg['chunk_cells']):
                guard();end=start+reg['chunk_cells']
                bits=torch.from_numpy(np.unpackbits(packed[start:end].numpy(),axis=1,bitorder='big').reshape(-1,1,32,32).copy())
                for mode in reg['modes']:
                    sequence=encode_occupancy(bits,mode);restored=decode_occupancy(sequence,32,mode)
                    original.require(torch.equal(bits,restored),'Physical support roundtrip failed')
                    original.require(np.array_equal(np.packbits(restored.numpy().reshape(len(bits),1024),axis=1,bitorder='big'),packed[start:end].numpy()),'Original packed bytes changed')
                    c=counters[mode];lengths=sequence.valid.sum(1);depth=sequence.depths[sequence.valid];actions=sequence.actions[sequence.valid]
                    c['cells']+=len(bits);c['empty_cells']+=int((~bits.bool().flatten(1).any(1)).sum());c['total_events']+=int(lengths.sum())
                    c['length_histogram']+=torch.bincount(lengths,minlength=343);c['depth_counts']+=torch.bincount(depth,minlength=6)
                    c['mask_counts_by_depth']+=torch.bincount(depth*16+actions,minlength=96).reshape(6,16)
                    c['decoded_sha256'].update(restored.numpy().tobytes())
            reports={}
            for mode,c in counters.items():
                original.require(0<=c['empty_cells']<=c['cells'] and c['empty_cells']==int(c['mask_counts_by_depth'][0,0]),'Image/root whitespace counts differ')
                report={k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in c.items() if k!='decoded_sha256'}
                report['decoded_sha256']=c['decoded_sha256'].hexdigest();report['mean_events_per_cell']=c['total_events']/c['cells']
                report['maximum_events']=int(c['length_histogram'].nonzero()[-1]);report['minimum_events']=int(c['length_histogram'].nonzero()[0])
                # Construction counts tensors; no neural forward or fit runs.
                torch.manual_seed(202611189);model=OccupancyGlyphDensity(mode=mode)
                report['component_parameters']=sum(p.numel() for p in model.parameters());reports[mode]=report
            original.require(reports['flat']['decoded_sha256']==reports['tree']['decoded_sha256'],'Physical modes cover different pixels')
            original.require(reports['flat']['component_parameters']==reports['tree']['component_parameters'],'Control tensor count differs')
            guard();result=dict(schema='ilm-occupancy-density-structure-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                modes=reports,all_68948_saved_cells_roundtrip=True,all_original_packed_bytes_preserved=True,
                actual_tree_to_flat_event_ratio=reports['tree']['total_events']/reports['flat']['total_events'],
                scope='68948 cells in 29740 distinct saved training-span images; not corpus exposure counts, target-only counts or held-out language',
                resources=dict(elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    minimum_available_host_gib=min(v['available_gib'] for v in samples),cpu_threads=2),
                neural_forward_calls=0,optimizer_updates=0,gpu_used=False,full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result)
            print(json.dumps(dict(events={k:(v['mean_events_per_cell'],v['maximum_events']) for k,v in reports.items()},ratio=result['actual_tree_to_flat_event_ratio'],seconds=result['resources']['elapsed_seconds'])),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
