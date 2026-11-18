"""Train from pictured titles, then score autonomous physical outputs by article."""
import argparse
from collections import Counter
import fcntl
import io
import json
import os
from pathlib import Path
import resource
import shutil
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.title_transducer import TitleTransducer, article_memory, visual_features
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_visual_sequence_corpus import ROOT
from scripts.train_visual_extractive_reader import save_page

DATA = ROOT/'data/visual_title_composition_v1'
OUT = ROOT/'artifacts/visual_title_transducer_v1'
PROTOCOL = ROOT/'references/visual_title_transducer_protocol.md'


class Examples:
    def __init__(self, split):
        self.rows = load(DATA/f'{split}_roster.json')
        self.p = np.memmap(DATA/f'{split}_passages.bin', dtype=np.uint8, mode='r').reshape(-1, 128)
        self.y = np.memmap(DATA/f'{split}_titles.bin', dtype=np.uint8, mode='r').reshape(-1, 128)
        self.question = np.fromfile(DATA/'question.bin', dtype=np.uint8).reshape(-1, 128)

    def pair(self, i):
        r = self.rows[i]
        p, t = r['passage_byte_offset']//128, r['title_byte_offset']//128
        return self.p[p:p+r['passage_cells']], self.y[t:t+r['title_cells']]


def batch_loss(model, examples, indices, dev):
    items = []
    for i in indices:
        passage, title = examples.pair(i)
        _, features, groups, lookup = article_memory(passage, examples.question)
        target_groups = [lookup.get(x.tobytes(), -1) for x in title]
        items.append((features, groups, title, target_groups))
    b = len(items)
    n, t = max(len(x[0]) for x in items), max(len(x[2]) for x in items) + 1
    x = np.zeros((b, n, 67), np.float32)
    gid = np.full((b, n), -2, np.int64)
    target = np.full((b, t), -3, np.int64)
    route = np.full((b, t), -100, np.int64)
    ink = np.zeros((b, t, 1024), np.float32)
    visual = np.zeros((b, t, 64), np.float32)
    for j, (f, g, y, ids) in enumerate(items):
        x[j, :len(f)] = f
        gid[j, :len(g)] = g
        target[j, :len(y)] = ids
        route[j, :len(y)] = np.asarray(ids) == -1
        route[j, len(y)] = 2
        ink[j, :len(y)] = np.unpackbits(y, axis=1)
        visual[j, :len(y)] = visual_features(y)
    x, gid, target, route, ink, visual = [torch.as_tensor(z, device=dev) for z in (x,gid,target,route,ink,visual)]
    valid = gid >= 0
    mem, hidden = model.encode(x, valid)
    focus, prev = torch.zeros_like(gid, dtype=x.dtype), torch.zeros_like(hidden)
    cost = x.new_zeros(())
    for j in range(t):
        hidden, probability, gate, paint = model.step(mem, valid, hidden, focus, prev)
        matched = probability * ((gid == target[:,j,None]) & valid)
        mass = matched.sum(1)
        select = route[:, j] == 0
        cost = cost + F.cross_entropy(gate, route[:,j], reduction='sum')
        cost = cost - (mass.clamp_min(1e-12).log() * select).sum()
        paint_cost = F.binary_cross_entropy_with_logits(paint, ink[:,j], reduction='none').mean(-1)
        cost = cost + (paint_cost * (route[:,j] == 1)).sum()
        focus = matched / mass[:,None].clamp_min(1e-12)
        prev = visual[:,j]
    return cost / (route >= 0).sum()


def resource_check(start):
    available = next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    require(int(available.split()[1])/1024**2 >= 24 and shutil.disk_usage(OUT.parent).free >= 48*1024**3, 'Host reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 4*1024**2 and time.monotonic()-start < 1500, 'Process ceiling')
    require(torch.cuda.max_memory_allocated() < 8*1024**3, 'Device ceiling')


def evaluate(model, examples, training_inventory, start):
    rows = []
    modes = ['normal','contiguous','exchange']
    example_rows = set()
    with ZipFile(OUT/'outputs.zip','x',compression=ZIP_DEFLATED) as archive:
        for i, r in enumerate(examples.rows):
            if i % 32 == 0:
                resource_check(start)
            p, y = examples.pair(i)
            row = dict(index=i, stratum=r['full_prompt_class'], unknown=any(
                c.tobytes() not in training_inventory for c in y))
            for mode in modes:
                source = examples.pair((i+1)%len(examples.rows))[0] if mode=='exchange' else p
                result, actions, stopped = model.write(source, examples.question, contiguous=mode=='contiguous')
                pixels = np.asarray(result, np.uint8).reshape(-1,128)
                archive.writestr(f'{mode}/{i}.bin', pixels.tobytes())
                exact = stopped and np.array_equal(pixels,y)
                row[mode] = dict(exact=exact,stopped=stopped,draws=sum(actions),cells=len(pixels))
                key = (row['stratum'],exact)
                if mode=='normal' and key not in example_rows:
                    example_rows.add(key)
                    prefix = OUT/f"first_{row['stratum']}_{'exact' if exact else 'miss'}"
                    save_page(Path(str(prefix)+'_ref.png'),y)
                    if len(pixels):
                        save_page(Path(str(prefix)+'_output.png'),pixels)
            rows.append(row)
    write_new(OUT/'rows.json',rows)
    scores={}
    for name in ['all','contiguous','distributed','missing_glyph','unknown']:
        group = [r for r in rows if name=='all' or r['stratum']==name or (name=='unknown' and r['unknown'])]
        scores[name]={'n':len(group)}
        for mode in modes:
            scores[name][mode]={k:sum(r[mode][k] for r in group) for k in ('exact','draws','cells','stopped')}
        diff=np.array([int(r['normal']['exact'])-int(r['contiguous']['exact']) for r in group])
        rng=np.random.default_rng(417)
        interval=np.quantile(rng.choice(diff,(2500,len(diff))).mean(1),[.025,.975])
        scores[name]['normal_minus_contiguous_95']=interval.tolist()
    return scores


def run():
    start=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(417)
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(), 'Expected 4090')
    resource_check(start)
    OUT.mkdir(exist_ok=False)
    data_report=load(DATA/'report.json')
    for key,s in [('training','train'),('development','dev')]:
        for field,suffix in [('passage','passages.bin'),('title','titles.bin'),('roster','roster.json')]:
            require(sha(DATA/f'{s}_{suffix}') == data_report['partitions'][key][field+'_sha256'], 'Changed corpus')
    train=Examples('train')
    require(len(train.rows)==2362 and sha(DATA/'question.bin')==data_report['question_sha256'],'Source roster')
    model=TitleTransducer().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    rng=np.random.default_rng(417); losses=[]
    train_start=time.monotonic()
    for epoch in range(12):
        order=rng.permutation(len(train.rows)); total=0.
        for i in range(0,len(order),16):
            if i%128==0: resource_check(start)
            loss=batch_loss(model,train,order[i:i+16],'cuda')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())
        losses.append(total/((len(order)+15)//16))
        print('completed epoch',epoch+1,'mean minibatch loss',losses[-1],flush=True)
    fit=time.monotonic()-train_start
    torch.save(model.state_dict(),OUT/'weights.pt')
    restored=TitleTransducer().cuda()
    restored.load_state_dict(torch.load(OUT/'weights.pt',weights_only=True))
    restored.eval()
    require(all(torch.equal(x,y) for x,y in zip(model.parameters(),restored.parameters())),'Final state mismatch')
    inventory=set(t.tobytes() for t in train.question)
    for i in range(len(train.rows)):
        for array in train.pair(i):
            inventory.update(x.tobytes() for x in array)
    scores=evaluate(restored,Examples('dev'),inventory,start)
    d=scores['distributed']
    passed=d['normal']['exact']>=5 and d['normal_minus_contiguous_95'][0]>0 and scores['missing_glyph']['normal']['exact']>=1
    result=dict(scores=scores,composition_pass=passed,fit_seconds=fit,total_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parameters=sum(x.numel() for x in model.parameters()),device=torch.cuda.get_device_name(),epoch_loss=losses,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'ilm/visual_lm/title_transducer.py',
                   PROTOCOL,DATA/'report.json',OUT/'weights.pt',OUT/'outputs.zip',OUT/'rows.json']})
    write_new(OUT/'report.json',result)
    print(json.dumps(scores),flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
