#!/usr/bin/env python3
"""Test a numerical repair on exact captured native writing-query states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT,gpu_zero,save_json,sha
from scripts.train_conditional_pixel_density import atomic_tensor,require

CAPTURE=ROOT/'artifacts/native_pixel_numerics_v1'
FILES=('scripts/validate_accurate_pixel_density.py','ilm/visual_lm/accurate_pixel_density.py',
    'ilm/visual_lm/conditional_pixel_density.py','ilm/visual_lm/pixel_density_numerics.py',
    'references/accurate_pixel_density_protocol.md')


def load(path):return json.loads(Path(path).read_text())


def identity():
    code={name:sha(ROOT/name) for name in FILES}
    for name,digest in code.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()==digest,
            'commit numerical repair source first: '+name)
    return code


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for preparation')
    code=identity();report=load(CAPTURE/'execution/report.json');analysis=load(CAPTURE/'analysis.json')
    require(load(CAPTURE/'execution/status.json')['phase']=='completed','complete numerical capture required')
    require(not analysis['old_gate_passed'] and analysis['greedy_GPU_paths_bit_exact'],'the declared captured discrepancy is required')
    bound={}
    for name in ('preparation.json','execution/report.json','execution/status.json','execution/decoder_snapshot.pt',
            'execution/GPU_trace.pt','FP64_reference.pt','analysis.json'):
        path=CAPTURE/name;bound[str(path.relative_to(ROOT))]=sha(path)
    require(sha(CAPTURE/'execution/decoder_snapshot.pt')==report['snapshot_sha256']
        and sha(CAPTURE/'execution/GPU_trace.pt')==report['GPU_trace_sha256']
        and sha(CAPTURE/'FP64_reference.pt')==analysis['reference_sha256'],'original tensors changed')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'preparation.json',dict(code_sha256=code,bound_sha256=bound,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        maximum_seconds=120,maximum_allocated_vram_gib=2.,unchanged_absolute_tolerance=1e-4,
        condition_examples=2,physical_positions_per_example=256,optimizer_updates=0,
        language_quality_claim=False))


def run(output):
    require(os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 only')
    meta=load(output/'preparation.json');require(identity()==meta['code_sha256'],'repair source changed')
    for name,digest in meta['bound_sha256'].items():require(sha(ROOT/name)==digest,'captured tensor changed: '+name)
    execution=output/'execution';execution.mkdir(exist_ok=False);started=time.perf_counter()
    def status(phase,**extra):save_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**extra))
    def deadline(signum,frame):raise TimeoutError('registered120-second decoder numerical test deadline')
    old=signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
    try:
        with gpu_zero() as device:
            import torch
            from ilm.visual_lm.conditional_pixel_density import ConditionalPixelDensity,PixelDensityConfig
            from ilm.visual_lm.accurate_pixel_density import AccurateConditionalPixelDensity
            from ilm.visual_lm.pixel_density_numerics import trace_density,differences
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            payload=torch.load(CAPTURE/'execution/decoder_snapshot.pt',map_location='cpu',weights_only=True)
            reference=torch.load(CAPTURE/'FP64_reference.pt',map_location='cpu',weights_only=True)['full']
            captured=torch.load(CAPTURE/'execution/GPU_trace.pt',map_location='cpu',weights_only=True)
            config=PixelDensityConfig(**payload['config'])
            baseline=ConditionalPixelDensity(torch.zeros(config.condition_width),torch.ones(config.condition_width),config).to(device)
            candidate=AccurateConditionalPixelDensity(torch.zeros(config.condition_width),torch.ones(config.condition_width),config).to(device)
            baseline.load_state_dict(payload['model']);candidate.load_state_dict(payload['model'])
            require(all(torch.equal(baseline.state_dict()[n],v) for n,v in candidate.state_dict().items()),'parameter parity failed')
            x,y=payload['states'].to(device),payload['targets'].to(device)
            status('matched_original_and_accurate_normalization_traces')
            original=trace_density(baseline,x,y);accurate=trace_density(candidate,x,y)
            atomic_tensor(execution/'baseline_trace.pt',original);atomic_tensor(execution/'candidate_trace.pt',accurate)
            replay=all(torch.equal(captured[k],original[k]) for k in ('teacher_full','teacher_cached','greedy_full','greedy_cached'))
            comparisons=dict(original=differences(original['teacher_full'],original['teacher_cached']),
                accurate=differences(accurate['teacher_full'],accurate['teacher_cached']),
                original_full_reference=differences(reference,original['teacher_full']),
                original_cached_reference=differences(reference,original['teacher_cached']),
                accurate_full_reference=differences(reference,accurate['teacher_full']),
                accurate_cached_reference=differences(reference,accurate['teacher_cached']))
            checks=dict(original_capture_replayed_bit_exact=replay,
                parameter_keys_shapes_values_identical=True,
                unchanged_absolute_gate_passed=comparisons['accurate']['max_absolute']<=1e-4,
                both_autonomous_greedy_paths_exact=bool(torch.equal(accurate['greedy_full'],accurate['greedy_cached'])),
                autonomous_greedy_matches_original=bool(torch.equal(accurate['greedy_full'],original['greedy_full'])),
                future_target_exclusion_exact=accurate['suffix_causality_max_error']==0. and accurate['altered_future_effect']>0.,
                both_paths_closer_to_independent_FP64=(comparisons['accurate_full_reference']['max_absolute']<comparisons['original_full_reference']['max_absolute']
                    and comparisons['accurate_cached_reference']['max_absolute']<comparisons['original_cached_reference']['max_absolute']))
            def benchmark(model):
                with torch.no_grad():
                    for _ in range(5):model(x,y)
                    torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):model(x,y)
                    end.record();torch.cuda.synchronize()
                return start.elapsed_time(end)/20
            timing=dict(original_two_block_forward_ms=benchmark(baseline),accurate_two_block_forward_ms=benchmark(candidate))
            candidate.zero_grad(set_to_none=True)
            logits=candidate(x,y)
            loss=-logits.log_softmax(-1).gather(-1,y[:,:,None]).mean();loss.backward()
            checks['all_parameter_gradients_finite_and_present']=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
            require(all(p.dtype==torch.float32 for p in candidate.parameters()),'parameter storage changed')
            peak=torch.cuda.max_memory_allocated()/1024**3
            result=dict(checks=checks,qualified_numerical_repair=all(checks.values()),comparisons=comparisons,
                timings=timing,peak_allocated_vram_gib=peak,elapsed_seconds=time.perf_counter()-started,
                candidate_trace_sha256=sha(execution/'candidate_trace.pt'),baseline_trace_sha256=sha(execution/'baseline_trace.pt'),
                parameters=sum(p.numel() for p in candidate.parameters()),normalization_accumulation='float64',
                parameter_storage='float32',maximum_absolute_tolerance=1e-4,optimizer_updates=0,
                complete_language_answers=0,language_quality_claim=False)
            save_json(execution/'report.json',result)
            require(peak<=2.,'registered component allocation exceeded')
            require(result['qualified_numerical_repair'],'numerical repair failed the registered comparison')
            status('completed')
    except BaseException as error:
        status('failed',reason=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        s=load(execution/'status.json') if (execution/'status.json').exists() else {}
        save_json(execution/'status.json',dict(s,pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','run'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.output.resolve())
