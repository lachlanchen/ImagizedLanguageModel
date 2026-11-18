"""Train matched fixed-path and path-marginal visual language transducers."""
import argparse
import fcntl
import hashlib
import time
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_edit_writer import COPY, STOP
from ilm.visual_lm.visual_span_writer import VisualSpanWriter
from scripts.compare_edit_diversity import PARENT_HASH
from scripts.compare_visual_edit_contrast import PARENT
from scripts.prepare_visual_spans import OUT, ROOT, WORK, frozen
from scripts.train_visual_edit_translation import guard, load, padded, require, sha, write_new

ARMS = ('fixed','marginal')


def restore(checkpoint=None, device='cuda'):
    if checkpoint is None:
        require(sha(PARENT)==PARENT_HASH, 'Original editor changed')
    weights = load_file(str(checkpoint or PARENT))
    net = VisualSpanWriter(weights['landmarks'])
    net.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fertility.')})
    return net.to(device)


def tiles():
    path = WORK/'pixels.npy'
    require(sha(path)==load(WORK/'rosters.json')['pixels_sha256'],'Bitmap bank changed')
    return np.load(path)


def training_examples():
    prompts = load(OUT/'preparation.json')['question_tiles']
    result = []
    for i,r in enumerate(frozen('train')):
        x = r['x']
        result.append((x,r['y'],prompts[0][i%2],r['a']))
        if i%4==0:
            result.append((x,x,prompts[1][(i//4)%2],[COPY]*len(x)+[STOP]))
    return result


def batches(rows, rng):
    buckets = defaultdict(list)
    for i in rng.permutation(len(rows)):
        buckets[len(rows[i][0])].append(i)
    result = []
    for group in buckets.values():
        for start in range(0,len(group),512):
            ordered = sorted(group[start:start+512],key=lambda i:len(rows[i][1]))
            result.extend(ordered[j:j+48] for j in range(0,len(ordered),48))
    rng.shuffle(result)
    return result


def tensor_pad(rows, device, value=0):
    ids, lengths = padded(rows)
    if value:
        ids[np.arange(ids.shape[1])[None]>=lengths[:,None]] = value
    return torch.as_tensor(ids, device=device).long(), torch.as_tensor(lengths, device=device).long()


def fit(arm):
    start = time.monotonic()
    guard(start)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(9091)
    rng = np.random.default_rng(9091)
    net = restore()
    net.motor.requires_grad_(False)
    ink = torch.from_numpy(np.unpackbits(tiles(),axis=1)).cuda().bool()
    features = visual_input(ink,net.landmarks)
    codes = ink[:,net.landmarks]
    rows = training_examples()
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=.0006, weight_decay=.01)
    order_hash = hashlib.sha256()
    hist = []
    folder = WORK/arm
    folder.mkdir()
    for epoch in range(3):
        loss_total = target_total = grid_cells = 0
        order = batches(rows,rng)
        for step,indices in enumerate(order):
            order_hash.update(np.asarray(indices,dtype='<i4').tobytes())
            x,y,q,a = zip(*(rows[i] for i in indices))
            src = torch.tensor(x,device='cuda')
            target,lengths = tensor_pad(y,src.device)
            questions,qlen = tensor_pad(q,src.device)
            trace,alen = tensor_pad(a,src.device,STOP)
            valid = torch.arange(trace.shape[1],device=src.device)[None]<alen[:,None]
            action,glyph = net.score_grid(features[src],codes[target],features[questions],qlen)
            if arm=='fixed':
                scores = net.fixed(action,glyph,trace,valid)
            else:
                scores = net.marginal(action,glyph,codes[src],codes[target],lengths)
                if step%128==0:
                    with torch.no_grad():
                        aligned = net.fixed(action,glyph,trace,valid)
                        require(bool((scores >= aligned-.001).all()),'Path sum below recorded path probability')
            cells = sum(map(len,y))
            loss = -scores.sum()/cells
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,2.,error_if_nonfinite=True)
            opt.step()
            loss_total += float(loss.detach())*cells
            target_total += cells
            grid_cells += len(x)*(src.shape[1]+1)*(target.shape[1]+1)
            if step%128==0:
                guard(start)
        row = dict(epoch=epoch+1,updates=len(order),presentations=len(rows),reference_cells=target_total,
                   padded_grid_cells=grid_cells,nll_per_reference_cell=loss_total/target_total,
                   elapsed_seconds=time.monotonic()-start)
        hist.append(row)
        print(arm,row,flush=True)
    weights = {k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()}
    save_file(weights,str(folder/'editor.safetensors'))
    write_new(folder/'fit.json',dict(history=hist,schedule_sha256=order_hash.hexdigest(),parent=PARENT_HASH,
              parameters=sum(p.numel() for p in net.parameters()),updated_parameters=sum(p.numel() for p in trainable),
              seconds=time.monotonic()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
              device=torch.cuda.get_device_name(),checkpoint_sha256=sha(folder/'editor.safetensors')))
    guard(start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require('4090' in torch.cuda.get_device_name(),'Expected one 4090')
        for name in ([args.arm] if args.arm else ARMS):
            fit(name)
