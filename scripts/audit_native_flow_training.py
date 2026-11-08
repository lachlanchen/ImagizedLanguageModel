#!/usr/bin/env python3
"""Independently audit the final native-flow checkpoint and actual previews.

CPU tensor inspection only. This authenticates optimization and generated
pixels, not language correctness, generalization or a new neural replay.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.native_flow_run import CONFIG, digest, identity, schedule, validate_checkpoint
from ilm.image_edit.native_flow_assessment import authenticate_native_png


def terminal_run(run):
    status = json.loads((run/'status.json').read_text())
    if status['phase'] != 'completed' or status['step'] != 1536 or status['pid'] is not None:
        raise ValueError('wait for completed final1536-update training')
    # Do not depend on an operator-created runtime handoff or a reusable PID.
    # A completed status can be written just before Python itself exits.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            words = [os.fsdecode(w) for w in (proc/'cmdline').read_bytes().split(b'\0') if w]
            if stat[0] == 'Z' or not any(Path(w).name == 'train_native_visual_flow.py' for w in words):
                continue
            if '--output' not in words:
                continue
            cwd = (proc/'cwd').resolve(strict=True)
            path = Path(words[words.index('--output')+1])
            if ((cwd/path).resolve() if not path.is_absolute() else path.resolve()) == run:
                raise ValueError('the actual training process has not exited yet')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return json.loads((run/'report.json').read_text())


def audit(run, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for CPU checkpoint inspection')
    trained = terminal_run(run)
    meta = json.loads((run/'preparation.json').read_text())
    if identity() != meta['code_sha256'] or trained['metadata'] != meta or trained['stage'] != 'completed':
        raise ValueError('original implementation or completed report differs')
    if sha(run/'training_pairs.jsonl') != meta['training_manifest_sha256']:
        raise ValueError('training image manifest changed')
    rows = [json.loads(line) for line in (run/'training_pairs.jsonl').read_text().splitlines()]
    if len(rows) != 384 or Counter(r['task'] for r in rows) != Counter(qa=256,copy=128) or any(r['split'] != 'train' for r in rows):
        raise ValueError('complete original train-only384-pair stream required')
    for row in rows:
        for key in ('input','target'):
            if sha(run/row[key]) != row[key+'_sha256']:
                raise ValueError('training input/target image changed')
    plan = schedule(rows)
    if plan != json.loads((run/'schedule.json').read_text()) or digest(plan) != meta['schedule_sha256']:
        raise ValueError('fixed four-pass schedule does not reproduce')
    import torch
    torch.set_num_threads(4)
    # Read once, then deserialize/hash that one byte snapshot.
    raw = (run/'checkpoint.pt').read_bytes()
    checkpoint_sha = hashlib.sha256(raw).hexdigest()
    if checkpoint_sha != trained['checkpoint_sha256']:
        raise ValueError('completed final checkpoint bytes differ')
    state = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    del raw
    if validate_checkpoint(state,meta,plan) != 1536 or trained['updates'] != 1536:
        raise ValueError('all1536 actual updates required')
    adapters = state['adapters']; optimizer = state['optimizer']
    names = {part+'.'+name+'.'+which for part,layers in meta['adapter_layer_names'].items()
        for name in layers for which in ('down','up')}
    ids = optimizer['param_groups'][0]['params']
    if set(adapters) != names or len(ids) != len(adapters) or set(ids) != set(optimizer['state']):
        raise ValueError('optimizer must cover the complete declared adapter graph')
    finite_moments = 0
    for (name,value), index in zip(adapters.items(),ids):
        actual = optimizer['state'][index]
        if value.dtype != torch.float32 or not torch.isfinite(value).all() or int(actual['step']) != 1536:
            raise ValueError('invalid final adapter or actual optimizer step: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            moment = actual[key]
            if moment.shape != value.shape or moment.dtype != value.dtype or not torch.isfinite(moment).all():
                raise ValueError('invalid actual AdamW moment: '+name)
        if (actual['exp_avg_sq'] < 0).any():
            raise ValueError('negative AdamW squared moment')
        finite_moments += 1
    parameters = sum(p.numel() for p in adapters.values())
    if parameters != meta['adapter_trainable_parameters'] or parameters != trained['actual_trainable_parameters']:
        raise ValueError('actual trainable count differs')
    history = state['history']
    if Counter(r['id'] for r in history) != Counter({r['id']:4 for r in rows}):
        raise ValueError('actual history is not four complete data permutations')
    if any(not math.isfinite(row[key]) for row in history for key in ('loss','gradient_norm','elapsed_seconds')):
        raise ValueError('nonfinite actual training observation')
    rng = {key:bool(torch.is_tensor(state[key]) and state[key].dtype == torch.uint8 and state[key].numel())
        for key in ('torch_cpu_rng','torch_cuda_rng','time_rng','noise_rng')}
    if not all(rng.values()):
        raise ValueError('checkpoint lacks actual CPU/CUDA/time/noise generator states')
    sum_update_seconds = sum(row['elapsed_seconds'] for row in history)
    del state, adapters, optimizer
    selected = [r for r in rows if r['task'] == 'qa'][:2]
    expected = {(step,row['id']):(20269490+i,row['input_sha256']) for step in range(128,1537,128)
        for i,row in enumerate(selected)}
    previews = {}
    for path in sorted(run.glob('attempt_*/previews/*/record.json')):
        record = json.loads(path.read_text());key = (record['after_update'],record['id'])
        if key not in expected or (record['seed'],record['input_sha256']) != expected[key] or not record['monitoring_only'] or record['checkpoint_selected_by_quality']:
            raise ValueError('unregistered preview or appearance selection')
        signature = authenticate_native_png(path.parent/'answer.png',record['result']['sha256'],record['result']['tensor_sha256'])
        if key in previews and previews[key]['actual_pixels'] != signature:
            raise ValueError('repeated fixed draw differs')
        previews[key] = dict(step=key[0],id=key[1],seed=record['seed'],
            image=str((path.parent/'answer.png').relative_to(run)),PNG_sha256=record['result']['sha256'],
            actual_tensor_file_sha256=record['result']['tensor_sha256'],actual_pixels=signature)
    if set(previews) != set(expected):
        raise ValueError('complete24 actual fixed monitoring draws required')
    if torch.cuda.is_initialized():
        raise ValueError('CPU audit initialized CUDA')
    output.mkdir(parents=True,exist_ok=False)
    result = dict(implementation_sha256=sha(Path(__file__)),training_preparation_sha256=sha(run/'preparation.json'),
        completed_report_sha256=sha(run/'report.json'),final_checkpoint_sha256=checkpoint_sha,
        code_sha256=meta['code_sha256'],actual_optimizer_updates=1536,registered_optimizer_updates=CONFIG['updates'],
        actual_train_only_image_pairs=len(rows),actual_training_QA_examples=256,actual_training_copy_examples=128,
        actual_training_QA_articles=meta['training_QA_articles'],
        exact_four_pass_schedule_reproduced=True,actual_adapter_tensors=len(names),actual_adapter_parameters=parameters,
        actual_AdamW_states_verified=finite_moments,all_actual_optimizer_steps_1536=True,
        all_actual_adapter_and_AdamW_tensors_finite=True,checkpoint_read_as_one_atomic_snapshot=True,
        actual_RNG_states_present=rng,new_full_neural_replay_performed=False,
        actual_peak_allocated_vram_GiB=trained['peak_allocated_vram_gib_this_attempt'],
        actual_peak_reserved_vram_GiB=trained['peak_reserved_vram_gib_this_attempt'],
        actual_attempt_elapsed_seconds=trained['elapsed_seconds_this_attempt'],
        sum_actual_update_elapsed_seconds=sum_update_seconds,whole_attempt_includes_loading_saving_and24_previews=True,
        actual_runtime_parameters=trained['runtime_parameters'],previews=list(previews.values()),
        verified_fixed_monitoring_previews=len(previews),all_monitoring_PNGs_equal_actual_decoder_tensors=True,
        monitored_examples_seen_during_training=True,heldout_assessment_completed=False,
        neural_models_loaded=0,new_neural_forward_calls=0,optimizer_updates_by_this_audit=0,
        language_quality_qualified=False,semantic_learning_validated=False,general_ilm_goal_complete=False)
    save_json(output/'report.json',result)
    print(json.dumps({k:result[k] for k in ('actual_optimizer_updates','actual_adapter_tensors','actual_adapter_parameters',
        'actual_AdamW_states_verified','verified_fixed_monitoring_previews','actual_peak_allocated_vram_GiB')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    audit(args.training_run.resolve(),args.output.resolve())
