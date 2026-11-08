"""Reproducible experiment state for mixed image-answer flow learning.

Imports stay framework-free until a model stage is admitted. Deployment
exports contain adapters and source bindings, not optimizer or answer data.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from .baseline import ROOT, sha

DATA = ROOT / "artifacts/visual_flow_pages_v1"
CONDITION = ROOT / "artifacts/visual_task_condition_1024_v1/runtime"
PREFLIGHT = ROOT / "artifacts/visual_flow_step_preflight_v2/execution"
FILES = ("ilm/image_edit/native_flow_run.py", "scripts/train_native_visual_flow.py",
    "scripts/infer_native_visual_flow.py", "ilm/image_edit/native_flow_step.py",
    "ilm/image_edit/visual_flow_learning.py", "ilm/image_edit/visual_semantic.py",
    "ilm/image_edit/compiled_visual_condition.py", "ilm/image_edit/baseline.py",
    "references/native_visual_flow_learning_protocol.md")
CONFIG = dict(epochs=4, updates=1536, batch_size=1, rank=16, learning_rate=5e-5,
    betas=[.9,.95], weight_decay=.05, clip_norm=1., save_every=32, preview_every=128,
    initialization_seed=20269401, order_seed=20269402, time_seed=20269403,
    noise_seed=20269404, codec_seed=20269405, maximum_allocated_vram_gib=23.,
    training_branch="positive_fixed_task", objective="native_unweighted_velocity_MSE",
    input_canvas=[1024,1024], output_canvas=[1024,1024], inference_steps=50, guidance=4.)


def identity():
    if subprocess.check_output(["git","status","--porcelain","--",*FILES],cwd=ROOT,text=True):
        raise ValueError("commit the complete experiment implementation and protocol first")
    return {name:sha(ROOT/name) for name in FILES}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def schedule(rows):
    if len(rows)!=384 or len({r['id'] for r in rows})!=384:
        raise ValueError("all384 unique training image pairs required")
    rng=random.Random(CONFIG['order_seed']);result=[]
    for epoch in range(1,CONFIG['epochs']+1):
        ids=[r['id'] for r in rows];rng.shuffle(ids)
        start=len(result)
        result.extend(dict(update=start+i+1,epoch=epoch,id=id) for i,id in enumerate(ids))
    if len(result)!=CONFIG['updates']:raise ValueError("complete four-pass schedule required")
    return result


def restore_adapters(pipe, values):
    from .native_flow_step import trainable
    copy_adapter_values(trainable(pipe),values)


def copy_adapter_values(selected,values):
    import torch
    if set(selected)!=set(values):raise ValueError("adapter names differ from actual model")
    with torch.no_grad():
        for name,p in selected.items():
            value=values[name]
            if value.shape!=p.shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise ValueError("invalid native FP32 adapter state: "+name)
            p.copy_(value.to(p.device))


def atomic_checkpoint(path, value, attempt):
    """Keep current readable throughout replacement and retain one predecessor."""
    import torch
    path=Path(path);temporary=path.with_name(path.name+f'.{attempt}.{os.getpid()}.tmp')
    with temporary.open('xb') as handle:
        torch.save(value,handle);handle.flush();os.fsync(handle.fileno())
    if path.exists():
        previous=path.with_name('checkpoint.previous.pt')
        previous_tmp=path.with_name(f'checkpoint.previous.{attempt}.{os.getpid()}.tmp')
        try:os.link(path,previous_tmp)
        except OSError:shutil.copyfile(path,previous_tmp)
        os.replace(previous_tmp,previous)
    os.replace(temporary,path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def checkpoint_state(*, adapters, optimizer, step, history, metadata, time_rng, noise_rng, cuda_rng):
    import torch
    return dict(adapters=adapters,optimizer=optimizer.state_dict(),step=step,history=list(history),metadata=metadata,
        torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=cuda_rng,
        time_rng=time_rng.get_state(),noise_rng=noise_rng.get_state())


def validate_checkpoint(value, metadata, plan):
    import torch
    step=value['step'];history=value['history']
    if value['metadata']!=metadata or not 0<=step<=CONFIG['updates'] or len(history)!=step:
        raise ValueError("checkpoint metadata or completed history differs")
    for actual,wanted in zip(history,plan):
        if any(actual[k]!=v for k,v in wanted.items()):raise ValueError("checkpoint schedule prefix differs")
    state=value['optimizer']
    if step and (not state['state'] or any(int(s['step'])!=step for s in state['state'].values())):
        raise ValueError("actual adapter optimizer step counters differ")
    if not all(torch.isfinite(v).all() for v in value['adapters'].values()):
        raise ValueError("checkpoint contains nonfinite adapter weights")
    if (len(state['param_groups'])!=1 or state['param_groups'][0]['lr']!=CONFIG['learning_rate'] or
        state['param_groups'][0]['weight_decay']!=CONFIG['weight_decay'] or
        tuple(state['param_groups'][0]['betas'])!=tuple(CONFIG['betas'])):
        raise ValueError("actual optimizer settings differ")
    return step
