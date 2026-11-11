"""Bounded full-sequence continuation training and checkpoint primitives."""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from .visual_sequence_corpus import VisualSequenceCorpus
from .semantic_glyph_loop import reconstruction_loss,tensor_hash
from .visual_continuation import VisualContinuationActor,prefill_visual_prompt
from scripts.train_semantic_glyph_language import reader,bridge_model
from scripts.compare_balanced_writer_matched import CHECKPOINT,CHECKPOINT_SHA
from scripts.score_pixel_language_first_blocks import require,host_resources

RUN=ROOT/'artifacts/visual_continuation_full_v1'
DATA=ROOT/'data/visual_sequence_corpus_v1'
SEED=2026111231


def load(path):return json.loads(Path(path).read_text())
def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def layout_schedule(rows,epochs,seed):
    schedule=[]
    for epoch in range(epochs):
        generator=torch.Generator().manual_seed(seed+epoch)
        for index in torch.randperm(len(rows),generator=generator).tolist():
            schedule.append((index,0 if (index+epoch)%3==0 else 1))
    return np.asarray(schedule,dtype=np.int64)


def learning_rate(step,total,peak=.0001,warmup=500):
    if not 1<=step<=total:raise ValueError('optimizer position outside schedule')
    if step<=min(warmup,total):return peak*step/min(warmup,total)
    progress=(step-warmup)/max(1,total-warmup)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*progress)))


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
                python=random.getstate(),numpy=(n[0],torch.tensor(n[1].astype(np.int64)),n[2],n[3],n[4]))


def set_rng_state(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],n[1].numpy().astype(np.uint32),n[2],n[3],n[4]))


def trainable_state(actor):
    return {n:p.detach().cpu().clone() for n,p in actor.named_parameters() if p.requires_grad}


def restore_trainable(actor,state):
    parameters={n:p for n,p in actor.named_parameters() if p.requires_grad}
    require(parameters.keys()==state.keys(),'Trainable parameter map differs')
    with torch.no_grad():
        for n,p in parameters.items():
            require(p.shape==state[n].shape and state[n].dtype==torch.float32,'Trainable shape/precision differs')
            p.copy_(state[n].to(p.device))


def build_actor(device,seed=SEED):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
    interface,provenance=reader(device)
    require(all(not bool(p.any()) for n,p in interface.actor.context.named_parameters() if n.endswith('.up')),
            'Source prompt adapters are not zero')
    bridge=bridge_model({},device,CHECKPOINT)
    actor=VisualContinuationActor(interface,bridge,rank=8,alpha=16.).train()
    bridge.readout.requires_grad_(True);bridge.stop.requires_grad_(True)
    require(sum(p.numel() for p in actor.parameters() if p.requires_grad)==20212225,'Wrong training model')
    return actor,provenance


class TrainingImages:
    def __init__(self,run):
        self.run=run;self.prepared=load(run/'preparation.json');self.rows=load(run/'training_manifest.json')['rows']
        self.corpus=VisualSequenceCorpus(DATA,split='train',expected_report_sha256=self.prepared['corpus_report_sha256'])

    def get(self,index,layout):
        row=self.rows[index];key='fixed_cells' if layout==0 else 'proportional'
        path=ROOT/row['images'][key]['file']
        require(sha(path)==row['images'][key]['sha256'],'Training prompt changed')
        with Image.open(path) as image:
            prompt=torch.from_numpy(np.asarray(image.convert('L')).copy()).float()[None,None]/127.5-1
        gold=self.corpus[row['corpus_index']]['ink'][row['cut']:].clone()
        require(len(gold)==row['target_glyphs'] and tensor_hash(gold)==row['target_sha256'],'Complete target changed')
        return prompt,gold,row,key


def update(actor,optimizer,images,position,total,monitor):
    index,layout=map(int,images['schedule'][position-1]);prompt,gold,row,key=images['data'].get(index,layout)
    monitor.position=dict(step=position,total=total,case=row['id'],layout=key,target_glyphs=len(gold));monitor.guard()
    began=time.perf_counter();memory,states,_=prefill_visual_prompt(actor.interface,prompt);del states
    signed=(1-2*gold.float()).to(actor.dynamics.query.device)
    with torch.no_grad():feedback=actor.bridge.feedback_embedding(signed)[None]
    optimizer.zero_grad(set_to_none=True)
    h=actor.dynamics(actor.dynamics.embeddings(feedback),memory,checkpoint_layers=True)
    logits,_,stop=actor.bridge.predict(h[0].float())
    image_loss=reconstruction_loss(logits[:-1],signed)
    stop_loss=.5*(F.softplus(stop[:-1]).mean()+F.softplus(-stop[-1]));loss=image_loss+.1*stop_loss
    require(bool(torch.isfinite(loss)),'Nonfinite training loss');loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    lr=learning_rate(position,total)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step();monitor.guard();torch.cuda.synchronize()
    return dict(step=position,row_index=index,case=row['id'],kind=row['kind'],layout=key,
        target_glyphs=len(gold),image_loss=float(image_loss.detach()),stop_loss=float(stop_loss.detach()),loss=float(loss.detach()),
        gradient_norm_before_clip=float(norm),learning_rate=lr,elapsed_seconds=time.perf_counter()-began)


class TrainingMonitor:
    """Keep resource observation bounded during multi-hour source-body training."""
    def __init__(self,path,seconds,pressure_wait=7200):
        self.path=path;self.seconds=seconds;self.began=time.perf_counter();self.waited=0.;self.allowance=pressure_wait
        self.position={};self.admission=None;self.samples=deque(maxlen=64);self.minimum=float('inf');self.next_poll=0.;self.count=0
    def status(self,phase,**extra):
        save_json(self.path/'status.json',dict(phase=phase,pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            elapsed_seconds=time.perf_counter()-self.began,position=self.position,**extra))
    def guard(self):
        now=time.perf_counter()
        require(now-self.began<self.seconds,'Registered stage wall budget exhausted')
        require(torch.cuda.max_memory_allocated()/1024**3<=22.,'Allocated GPU limit exceeded')
        if now<self.next_poll:return
        while True:
            value=host_resources();self.samples.append(value);self.count+=1;self.minimum=min(self.minimum,value['available_gib'])
            require(value['available_gib']>=24.,'Host runtime reserve exhausted')
            if value['some_pressure_avg10']<=.5 and value['full_pressure_avg10']<=.1:
                self.next_poll=time.perf_counter()+.5;return
            remaining=min(self.allowance-self.waited,self.seconds-(time.perf_counter()-self.began))
            require(remaining>0,'Pressure wait budget exhausted');self.status('waiting_for_memory_pressure',resources=value)
            began=time.perf_counter();time.sleep(min(10.,remaining));self.waited+=time.perf_counter()-began
    def receipt(self):
        return dict(elapsed_seconds=time.perf_counter()-self.began,resource_wait_seconds=self.waited,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,minimum_available_gib=self.minimum,
            resource_samples=self.count,resource_sample_retention=64,host_poll_interval_seconds=.5,admission=self.admission)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def same_nested(a,b):
    if isinstance(a,torch.Tensor):require(isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu()),'Tensor replay differs')
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),'Map replay differs')
        for k in a:same_nested(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        require(type(a)==type(b) and len(a)==len(b),'Sequence replay differs')
        for x,y in zip(a,b):same_nested(x,y)
    else:require(a==b,'Scalar replay differs')
