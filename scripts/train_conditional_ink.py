"""Fit adjacent physical group interactions against the released phrase writer."""
import fcntl
import time

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import LocalInk, signatures
from scripts.infer_visual_phrase import MODELS
from scripts.train_local_ink import (OUT, ROOT, calibration_group, checkpoint, fields, guard, load,
                                     require, sha, sites, tensors, tiles, training_rows, write_new)

WORK = ROOT / 'artifacts/conditional_ink_v1'
DIRECTORY = MODELS / 'local_ink'
GRID = (0., .25, .5, 1., 2.)


@torch.no_grad()
def observations(rows, bank, started):
    """Collect exact frozen teacher logits at every genuine insertion glyph."""
    net = checkpoint('source')
    ink = torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool()
    bits = ink[:, net.landmarks]
    sig = signatures(bits.cpu().numpy())
    features = visual_input(ink, net.landmarks)
    question = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    models = [LocalInk.load(DIRECTORY / (name + '.npz')) for name in ('later', 'onset')]
    columns = [[] for _ in range(5)]
    for start in range(0, len(rows), 64):
        block = [r for r in rows[start:start+64] if r['x']]
        spans = sites(block)
        where = [(b, c, t, k) for b, c, j, k in spans for t in range(j, k)]
        if not where:
            continue
        onset = np.fromiter((t == j for b, c, j, k in spans for t in range(j, k)), dtype=bool)
        context = fields(block, sig, where)
        b, c, t, _ = torch.as_tensor(where, device='cuda').T
        mem, _, hist, y, _ = tensors(block, features, bits, net, question, False)
        target = bits[y[b, t]]
        state = (net.boundary(mem, b, c) + hist[b, t]).tanh()
        logits = net.draw.teacher(state, target)
        for i, model in enumerate(models):
            mask = onset == bool(i)
            if np.any(mask):
                logits[torch.as_tensor(mask, device='cuda')] += torch.from_numpy(model.bias(context[mask])).cuda()
        held = [calibration_group(block[b]['g']) for b, _, _, _ in where]
        values = [torch.from_numpy(context.astype('int64')), torch.from_numpy(onset),
                  nibbles(target).byte().cpu(), logits.cpu(), torch.tensor(held)]
        for col, value in zip(columns, values, strict=True):
            col.append(value)
        if start % 1024 == 0:
            guard(started)
    return [torch.cat(col).cuda() for col in columns]


def loss_at(model, data, indices):
    contexts, onset, codes, logits, _ = data
    target = codes[indices].long()
    delta = model.teacher(contexts[indices], onset[indices], target)
    base = logits[indices, 1:]
    return base, delta, target[:, 1:]


def nll(logits, target):
    return F.cross_entropy(logits.transpose(1, 2), target, reduction='none').sum(1)


def run():
    started = time.monotonic()
    guard(started)
    path = WORK / 'transitions.safetensors'
    if path.exists():
        raise FileExistsError(path)
    torch.manual_seed(4117)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    rows = training_rows()
    data = observations(rows, tiles(), started)
    prep_seconds = time.monotonic() - started
    train = torch.where(~data[-1])[0]
    held = torch.where(data[-1])[0]
    require(len(train) + len(held) == 627438 and int(data[1].sum()) == 296837, 'Insertion coverage differs')
    print('prepared', len(train), len(held), 'in', prep_seconds, flush=True)
    model = ConditionalInk().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    rng = torch.Generator(device='cuda').manual_seed(4117)
    history = []
    for epoch in range(3):
        order = train[torch.randperm(len(train), generator=rng, device='cuda')]
        total = 0.
        for indices in order.split(1024):
            base, delta, target = loss_at(model, data, indices)
            loss = nll(base + delta, target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append(total / len(train))
        guard(started)
        print('epoch', epoch+1, history[-1], flush=True)
    with torch.no_grad():
        totals = np.zeros(len(GRID))
        for indices in held.split(1024):
            base, delta, target = loss_at(model, data, indices)
            for i, weight in enumerate(GRID):
                totals[i] += float(nll(base + weight * delta, target).double().sum())
        model.scale.fill_(GRID[int(totals.argmin())])
    manifest_hash = sha(DIRECTORY / 'manifest.json')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path),
              metadata={'format': 'ilm-conditional-ink-v1', 'counts_manifest': manifest_hash})
    receipt = dict(training_sentences=len(rows), optimizer_cells=len(train), calibration_cells=len(held),
                   epoch_nll=history, calibration_grid=GRID, calibration_nll=(totals/len(held)).tolist(),
                   residual_scale=float(model.scale), parameters=sum(p.numel() for p in model.parameters()),
                   seconds=time.monotonic()-started, preparation_seconds=prep_seconds,
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                   checkpoint_sha256=sha(path), checkpoint_bytes=path.stat().st_size, counts_manifest=manifest_hash)
    write_new(WORK / 'fit.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    WORK.mkdir(exist_ok=True)
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_NB | fcntl.LOCK_EX)
        run()
