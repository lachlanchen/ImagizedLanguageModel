#!/usr/bin/env python3
"""Capture the actual failed-interface decoder tensors; analyze them on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from scripts.preflight_native_pixel_actor import (
    NATIVE, RASTER, CONTROLS, CODEC, SEED, unchanged, read_png, prefix_identity,
)
from scripts.train_conditional_pixel_density import atomic_tensor, require

PRIOR=ROOT/'artifacts/native_pixel_actor_preflight_v1'
FILES=('scripts/capture_native_pixel_numerics.py','ilm/visual_lm/pixel_density_numerics.py',
    'references/native_pixel_numerics_protocol.md')


def load(path):return json.loads(Path(path).read_text())


def identity():
    result={name:sha(ROOT/name) for name in FILES}
    for name,digest in result.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()==digest,
            'commit numerical diagnosis first: '+name)
    return result


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for preparation')
    source=unchanged(PRIOR);code=identity()
    status=load(PRIOR/'execution/status.json')
    require(status['phase']=='failed' and status['logical_updates']==0 and status['pid'] is None,
        'the predecessor must be terminal before diagnosis')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'preparation.json',dict(code_sha256=code,prior_code_sha256=source['code_sha256'],
        prior_preparation_sha256=sha(PRIOR/'preparation.json'),prior_status_sha256=sha(PRIOR/'execution/status.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=SEED,maximum_seconds=600,maximum_allocated_vram_gib=22.,
        retained_optimizer_updates=0,generation_quality_claim=False,
        old_max_absolute_tolerance=1e-4,old_gate_remains_required=True,
        diagnostic_policy='retain actual states and weights; compare FP32 paths with independent FP64 reference; no automatic tolerance change'))


def capture(output):
    require(os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0',
        'physical GPU0 only')
    meta=load(output/'preparation.json');require(identity()==meta['code_sha256'],'diagnosis source changed')
    unchanged(PRIOR)
    require(sha(PRIOR/'preparation.json')==meta['prior_preparation_sha256']
        and sha(PRIOR/'execution/status.json')==meta['prior_status_sha256'],'predecessor changed')
    execution=output/'execution';execution.mkdir(exist_ok=False)
    started=time.perf_counter()
    def status(phase,**extra):
        save_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**extra))
    def deadline(signum,frame):raise TimeoutError('registered600-second numerical capture deadline')
    old=signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    try:
        with gpu_zero() as device:
            import gc
            import torch
            from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
            from ilm.visual_lm.native_pixel_language import NativePixelLanguageModel
            from ilm.visual_lm.conditional_pixel_density import pack_pixels
            from ilm.visual_lm.pixel_density_numerics import trace_density,differences
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            rows=torch.load(PRIOR/'data.pt',map_location='cpu',weights_only=True)
            local=ROOT.parent/'LocalLLM/.local'
            status('loading_same_complete_native_actor')
            actor,provenance=load_trained_native_image_actor(checkpoint=NATIVE/'training/checkpoint_latest.pt',
                controls=CONTROLS,codec=CODEC,ollama_root=local/'models/ollama',
                gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
            previous=actor.writing
            warm={k:v.detach().cpu().clone() for k,v in previous.state_dict().items()
                if k.startswith(('image_input.','stop.')) or k=='input_scale'}
            raster=torch.load(RASTER/'execution/final.pt',map_location='cpu',weights_only=True)['model']
            writer=NativePixelLanguageModel(previous.continuation.context,previous.writer,
                input_scale=float(previous.input_scale),query_seed=SEED+1,raster_state=raster).to(device)
            writer.warm_image_interfaces(warm);actor.writing=writer
            del previous,warm,raster;gc.collect();torch.cuda.empty_cache();actor.eval()
            status('capturing_same_native_PNG_prefills')
            prefixes=[];identities=[]
            for i,row in enumerate(rows):
                prefix,geometry=actor.prefill(read_png(PRIOR/f'inputs/{i:02}.png'))
                require(geometry['all_prompt_keys_retained'] and prefix.blocked_before==0,'full native prefix required')
                identities.append(prefix_identity(prefix));prefixes.append(prefix.to('cpu'));del prefix
            atomic_tensor(execution/'prefixes.pt',[dict(key_values=p.key_values,initial_state=p.initial_state,
                rotary_offsets=p.rotary_offsets,blocked_before=p.blocked_before,valid_lengths=p.valid_lengths) for p in prefixes])
            encoded=[writer.image_codes(row['images']).cpu() for row in rows[:2]]
            atomic_tensor(execution/'initial_actor.pt',writer.trainable_state_dict())
            atomic_tensor(execution/'encoded_training_images.pt',encoded)
            with torch.no_grad():
                full=writer.conditions(prefixes[0].to(device),encoded[0][None].to(device),checkpoint_layers=False)
                states=writer.normalized_conditions(full[0,:2])
                targets=pack_pixels(rows[0]['images'][:2,0].to(device)<0)
                # This snapshot is written before a numerical assertion can fail.
                atomic_tensor(execution/'decoder_snapshot.pt',dict(model={k:v.detach().cpu().clone() for k,v in writer.density.state_dict().items()},
                    config=writer.density.configuration(),states=states.cpu(),targets=targets.cpu()))
                status('tracing_all256_physical_positions')
                traced=trace_density(writer.density,states,targets)
                atomic_tensor(execution/'GPU_trace.pt',traced)
                measured=differences(traced['teacher_full'],traced['teacher_cached'])
                measured.update(greedy_bit_exact=bool(torch.equal(traced['greedy_cached'],traced['greedy_full'])),
                    suffix_causality_max_error=traced['suffix_causality_max_error'],altered_future_effect=traced['altered_future_effect'])
            peak=torch.cuda.max_memory_allocated()/1024**3
            require(peak<=22.,'registered capture allocation exceeded')
            save_json(execution/'report.json',dict(measurements=measured,
                snapshot_sha256=sha(execution/'decoder_snapshot.pt'),GPU_trace_sha256=sha(execution/'GPU_trace.pt'),
                initial_actor_sha256=sha(execution/'initial_actor.pt'),prefixes_sha256=sha(execution/'prefixes.pt'),
                source=provenance,prefix_identities=identities,elapsed_seconds=time.perf_counter()-started,
                peak_allocated_vram_gib=peak,old_absolute_gate_passed=measured['max_absolute']<=1e-4,
                optimizer_updates=0,autonomous_complete_answers_generated=0,language_quality_claim=False))
            status('completed')
    except BaseException as error:
        status('failed',reason=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        state=load(execution/'status.json') if (execution/'status.json').exists() else {}
        save_json(execution/'status.json',dict(state,pid=None))


def analyze(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for FP64 CPU reference')
    meta=load(output/'preparation.json');require(identity()==meta['code_sha256'],'diagnosis source changed')
    require(load(output/'execution/status.json')['phase']=='completed','capture must finish before CPU audit')
    import torch
    from ilm.visual_lm.pixel_density_numerics import differences,reference_teacher
    torch.set_num_threads(4)
    report=load(output/'execution/report.json')
    for name,key in (('decoder_snapshot.pt','snapshot_sha256'),('GPU_trace.pt','GPU_trace_sha256')):
        require(sha(output/'execution'/name)==report[key],'captured numerical tensors changed')
    payload=torch.load(output/'execution/decoder_snapshot.pt',map_location='cpu',weights_only=True)
    trace=torch.load(output/'execution/GPU_trace.pt',map_location='cpu',weights_only=True)
    require(torch.equal(payload['states'],trace['states']) and torch.equal(payload['targets'],trace['targets']),
        'traced conditions and true prefix must match captured tensors')
    start=time.perf_counter()
    full=reference_teacher(payload['model'],payload['config'],payload['states'],payload['targets'])
    cached=reference_teacher(payload['model'],payload['config'],payload['states'],payload['targets'],incremental=True)
    atomic_tensor(output/'FP64_reference.pt',dict(full=full,cached=cached))
    result=dict(GPU_paths=differences(trace['teacher_full'],trace['teacher_cached']),
        FP64_paths=differences(full,cached),GPU_full_vs_FP64=differences(full,trace['teacher_full']),
        GPU_cached_vs_FP64=differences(full,trace['teacher_cached']),
        greedy_GPU_paths_bit_exact=bool(torch.equal(trace['greedy_cached'],trace['greedy_full'])),
        reference_sha256=sha(output/'FP64_reference.pt'),elapsed_seconds=time.perf_counter()-start,
        old_max_absolute_tolerance=1e-4,old_gate_passed=report['old_absolute_gate_passed'],
        automatic_tolerance_change=False,training_qualified=False,language_quality_claim=False)
    require(not (output/'analysis.json').exists(),'completed analysis is immutable')
    save_json(output/'analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','capture','analyze'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.output.resolve())
