#!/usr/bin/env python3
"""Register/run one fixed learned-metric replacement for the rejected bit metric."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts import calibrate_selective_visual_verifier as base

ROOT = base.ROOT
base.REG = ROOT/'references/evidence/frozen_encoder_verifier_registration_20261116.json'
base.OUT = ROOT/'references/evidence/frozen_encoder_verifier_calibration_20261116.json'
base.RUN = ROOT/'artifacts/frozen_encoder_verifier_v1'
CHECKPOINT = ROOT/'artifacts/balanced_glyph_writer_v1/fit/checkpoint_latest.pt'
CHECKPOINT_SHA = 'dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d'
PROTOCOL = ROOT/'references/frozen_encoder_verifier_protocol_20261116.md'


def contract():
    rows,sources,training = base.contract()
    if base.sha(CHECKPOINT) != CHECKPOINT_SHA: raise ValueError('Frozen encoder checkpoint changed')
    paths = [Path(__file__),PROTOCOL,CHECKPOINT,
        ROOT/'ilm/training_only/frozen_encoder_verifier.py',ROOT/'tests/test_frozen_encoder_verifier.py',
        ROOT/'ilm/visual_lm/semantic_glyph_loop.py',
        ROOT/'references/evidence/training_glyph_label_result_20261116.json']
    for p in paths: sources[str(p.relative_to(ROOT))]=base.sha(p)
    old = base.load(ROOT/'references/evidence/whole_answer_reward_calibration_registration_20261116.json')
    for name,digest in old['input_sha256'].items():
        if name.startswith(('data/','artifacts/')) and sources.get(name)!=digest:
            raise ValueError('Preserve original frozen output/reference data')
    return rows,sources,training


def main(action):
    action_began=time.monotonic()
    rows,sources,training = contract()
    if action=='register':
        base.write_new(base.REG,dict(schema='frozen-encoder-content-verifier-registration-v1',rows=rows,
            input_sha256=sources,training_pixel_sources=training,checkpoint_sha256=CHECKPOINT_SHA,
            feature='Frozen bridge encoder, 4096 raw normalized-semantic coordinates; +1 white, -1 ink; FP32 forward, FP64 direct Euclidean distances.',
            acceptance='Strict 2*distance < nearest other prototype separation; no fitted radius or alternative metric.',
            reward='Unchanged stopped ordered physical-identity edit score and complete-image-content bonus from the rejected Hamming verifier; caps/allwhite zero.',
            checks=['all clean references covered','all9correct full content accepted','no9readablewrong or46uncertain full content accepted',
                    'correct/readablewrong AUC>=.95','mincorrect>maxuncredited','capszero','distinct finite source features'],
            resources=dict(cpu_threads=2,maximum_rss_gib=2,maximum_elapsed_seconds=300,maximum_allocated_gpu_gib=2,
                           entry_host_available_gib=48,runtime_host_available_gib=24,entry_gpu0_free_gib=20),
            action_on_failure='Reject this exact fixed feature map/metric. No metric/checkpoint/radius tuning, student fit or old-reading repair.',
            action_on_pass='Limited reused-output qualification only; register fresh-clone actor own-image sampling/replay/gradient/resource preflight before optimization.',
            neural_execution=True,fixed_weights=True,optimizer_updates=0,policy_fit_authorized=False,
            deployed_catalogue=False,raw_Unicode_recovery_claim=False))
        print(json.dumps(dict(registered=str(base.REG.relative_to(ROOT)),cases=len(rows),training_shards=len(training))))
        return
    reg = base.load(base.REG)
    if reg['rows']!=rows or reg['input_sha256']!=sources or reg['training_pixel_sources']!=training:
        raise ValueError('Registered fixed feature calibration changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True,timeout=30).split()[0]
    if head!=remote or subprocess.check_output(['git','show','HEAD:'+str(base.REG.relative_to(ROOT))],cwd=ROOT)!=base.REG.read_bytes():
        raise ValueError('Commit and push fixed calibration before model execution')
    mem=next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
    if int(mem.split()[1])/1024**2<48:raise RuntimeError('Host entry reserve not met')
    info=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if int(info[1])<20*1024:raise RuntimeError('GPU0 entry reserve not met')
    os.environ['CUDA_VISIBLE_DEVICES']=info[0].strip()
    import torch
    from ilm.training_only import visual_content_verifier
    from ilm.training_only.frozen_encoder_verifier import FrozenEncoderVerifier
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    FrozenEncoderVerifier.checkpoint=CHECKPOINT
    visual_content_verifier.SelectiveGlyphVerifier=FrozenEncoderVerifier
    base.write_new(ROOT/'artifacts/frozen_encoder_verifier_v1_owner.json',dict(pid=os.getpid(),
        start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),physical_gpu_uuid=info[0].strip(),
        registration_sha256=base.sha(base.REG),started_unix=time.time(),automatic_retry=False))
    began=time.monotonic()
    try:
        result=base.calibrate(reg)
        verifier=FrozenEncoderVerifier.last_instance
        result['schema']='frozen-encoder-visual-content-calibration-v1'
        result['neural_calls']=verifier.forward_calls
        result['encoded_glyph_positions']=verifier.encoded_images
        result['encoder_parameters']=verifier.encoder_parameters
        result['feature_sha256']=verifier.feature_sha256
        import numpy as np
        np.savez_compressed(base.RUN/'feature_radii.npz',minimum_distance=verifier.minimum_distance)
        base.write_new(base.RUN/'decode_traces.json',verifier.decode_traces)
        result['radius_file_sha256']=base.sha(base.RUN/'feature_radii.npz')
        result['decode_traces_sha256']=base.sha(base.RUN/'decode_traces.json')
        result['minimum_feature_separation']=float(verifier.minimum_distance.min())
        result['maximum_feature_separation']=float(verifier.minimum_distance.max())
        result['checks']['distinct_finite_source_features']=bool((verifier.minimum_distance>0).all())
        result['calibration_passed']=all(result['checks'].values())
        for key in ('minimum_separation','maximum_separation'):result['inventory'].pop(key)
        result['peak_allocated_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
        result['torch_version']=torch.__version__
        result['cuda_runtime']=torch.version.cuda
        result['raw_Unicode_recovery_claim']=False
        result['metric_fitted_on_development']=False
        result['new_student_optimizer_updates']=0
        result['whole_action_seconds']=time.monotonic()-action_began
        if result['whole_action_seconds']>300:raise RuntimeError('Whole action wall ceiling exceeded')
        if result['peak_allocated_gpu_gib']>2:raise RuntimeError('GPU allocation ceiling exceeded')
        base.write_new(base.OUT,result)
        print(json.dumps({k:result[k] for k in ['calibration_passed','accepted_correct','accepted_readable_wrong',
            'accepted_uncertain','checks','neural_calls','encoder_parameters','peak_allocated_gpu_gib','elapsed_seconds']}),flush=True)
    except BaseException as error:
        if base.RUN.exists():base.write_new(base.RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-began,automatic_retry=False,student_training_authorized=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));args=parser.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS')!='2' or os.environ.get('OMP_NUM_THREADS')!='2':
        raise ValueError('Two-thread execution required')
    if args.action=='run':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Registered deterministic CUDA workspace required')
        with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args.action)
    else:main(args.action)
