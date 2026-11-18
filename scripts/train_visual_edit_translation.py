"""Fit the physical ink renderer, then the recurrent classical Chinese visual editor."""
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import choose_landmarks, nibbles, visual_input
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, START, STOP, VisualEditWriter
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_pictured_idioms import guard, learn_motor
from scripts.prepare_visual_edit_translation import OUT, ROOT


def inputs(split):
    prep = load(OUT/'preparation.json')
    file = OUT/(split+'.jsonl')
    require(sha(file)==prep['partitions'][split]['sha256'],'Prepared records changed')
    return [json.loads(line) for line in file.open()]


def bank():
    prep = load(OUT/'preparation.json')
    path = OUT/'pixels.npy'
    require(sha(path)==prep['pixels_sha256'],'Prepared pixels changed')
    return np.load(path)


def padded(rows,dtype=np.int32):
    lengths = np.array(list(map(len,rows)),np.int32)
    arr = np.zeros((len(rows),int(lengths.max())),dtype)
    for i,row in enumerate(rows):
        arr[i,:len(row)] = row
    return arr,lengths


def examples(rows):
    sources,questions,actions,emissions = [],[],[],[]
    prompts = load(OUT/'preparation.json')['question_tiles']
    for i,r in enumerate(rows):
        for quote in range(1+int(i%4==0)):
            x = r['x']
            sources.append(x)
            questions.append(prompts[quote][(i//4 if quote else i)%2])
            actions.append([COPY]*len(x)+[STOP] if quote else r['a'])
            emissions.append(x+[0] if quote else r['o'])
    x,xlen = padded(sources)
    q,qlen = padded(questions)
    a,steps = padded(actions,np.int8)
    o,_ = padded(emissions)
    pos = np.arange(a.shape[1])[None]
    advance = np.isin(a,[COPY,REPLACE,DROP]) & (pos<steps[:,None])
    cursor = np.pad(advance.cumsum(1,dtype=np.int32)[:,:-1],((0,0),(1,0)))
    last = np.maximum.accumulate(np.where((a<=INSERT)&(pos<steps[:,None]),pos+1,0),axis=1)
    prev_output = np.take_along_axis(np.pad(o,((0,0),(1,0))),np.pad(last[:,:-1],((0,0),(1,0))),axis=1)
    prev_action = np.pad(a[:,:-1],((0,0),(1,0)),constant_values=START)
    assert np.all(cursor.max(1)==xlen)
    columns = [x,xlen,q,qlen,a,steps,o,cursor,prev_action,prev_output.astype(np.int32)]
    return [torch.from_numpy(v).cuda() for v in columns],steps,xlen


def scored(net,features,ink,columns,ix,source_width,steps,*,draw_auxiliary=None):
    x,lens,q,qlens,a,ns,o,cursor,prev_action,prev_output = [v[ix] for v in columns]
    x = x[:,:source_width]
    a,o,cursor,prev_action,prev_output = [v[:,:steps].long() for v in (a,o,cursor,prev_action,prev_output)]
    valid = torch.arange(source_width,device='cuda')[None]<lens[:,None]
    qvalid = torch.arange(q.shape[1],device='cuda')[None]<qlens[:,None]
    use = torch.arange(steps,device='cuda')[None]<ns[:,None]
    mem,g = net.encode(features[x],valid,features[q],qvalid)
    h = net.teacher(mem,g,cursor,features[prev_output],prev_action)
    logits = net.action_logits(h,cursor==lens[:,None])
    action_loss = F.cross_entropy(logits[use],a[use])
    paint = ((a==REPLACE)|(a==INSERT)) & use
    if not paint.any():
        return action_loss,0.
    wanted = ink[o[paint]][:,net.landmarks].float()
    pattern = net.draw.teacher(h[paint],wanted)
    ink_loss = F.cross_entropy(pattern.flatten(0,1),nibbles(wanted).flatten())
    loss = action_loss + 3*ink_loss
    if draw_auxiliary is not None:
        loss = loss + draw_auxiliary(net.draw, h[paint], wanted)
    return loss, ink_loss.detach()


def run():
    began = time.monotonic()
    guard(began)
    require('4090' in torch.cuda.get_device_name(),'Expected a single 4090-class GPU')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(4919)
    rng = np.random.default_rng(4919)
    out = OUT/'fit'
    out.mkdir()
    tiles = bank()
    train_path = OUT/'train_shapes.npy'
    require(sha(train_path)==load(OUT/'preparation.json')['training_shapes_sha256'],'Training shapes changed')
    train_shapes = np.load(train_path)
    masks = np.unpackbits(tiles,axis=1)
    landmarks = choose_landmarks(masks[train_shapes])
    ink = torch.from_numpy(masks).cuda().bool()
    net = VisualEditWriter(landmarks).cuda()
    motor_start = time.monotonic()
    motor = learn_motor(net,ink[train_shapes],rng,began)
    motor['seconds'] = time.monotonic()-motor_start
    write_new(out/'motor.json',motor)
    features = visual_input(ink,net.landmarks)
    data,steps,source_lengths = examples(inputs('train'))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    begin_fit = time.monotonic()
    history = []
    for epoch in range(8):
        order = rng.permutation(len(steps))
        batches = []
        for block in np.array_split(order,32):
            block = block[np.argsort(steps[block])]
            batches.extend(block[a:a+128] for a in range(0,len(block),128))
        rng.shuffle(batches)
        totals = np.zeros(3)
        for it,idx in enumerate(batches):
            w,t = int(source_lengths[idx].max()),int(steps[idx].max())
            loss,paint_loss = scored(net,features,ink,data,torch.as_tensor(idx,device='cuda'),w,t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True)
            opt.step()
            totals += [len(idx),loss.detach().item()*len(idx),float(paint_loss)*len(idx)]
            if it%128==0:
                guard(began)
        row=dict(epoch=epoch+1,examples=int(totals[0]),updates=len(batches),loss=totals[1]/totals[0],
                 physical_group_nll=totals[2]/totals[0],wall_seconds=time.monotonic()-begin_fit)
        history.append(row)
        print(row,flush=True)
    state = {k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()}
    save_file(state,str(out/'editor.safetensors'))
    guard(began)
    write_new(out/'report.json',dict(history=history,motor=motor,parameters=sum(p.numel() for p in net.parameters()),
        trainable_editor=sum(p.numel() for p in net.parameters() if p.requires_grad),device=torch.cuda.get_device_name(),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,seconds=time.monotonic()-began,
        weights_sha256=sha(out/'editor.safetensors'),preparation_sha256=sha(OUT/'preparation.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),
             ROOT/'ilm/visual_lm/visual_edit_writer.py',ROOT/'references/visual_edit_translation_protocol.md')}))


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
