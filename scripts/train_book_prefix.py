"""Fit the four matched classical Chinese book writers serially on GPU zero."""
import hashlib
import json
import math
import time

import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.book_prefix import BookWriter, observations
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_book_prefix import corpus, MODEL, WORK, PUB
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_span_choice import gpu_owner

SEED = 194671
STEPS = 3200
BATCH = 24
LENGTH = 64


def train(meta, data, starts, oof, mode, hybrid):
    kind = 'hybrid' if hybrid else 'neural'
    name = f'{mode}-{kind}'
    receipt = PUB / f'{name}.json'
    weights = MODEL / f'{name}.safetensors'
    if receipt.exists():
        if sha(weights) != json.loads(receipt.read_text())['checkpoint']:
            raise ValueError(f'Changed {weights}')
        return
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    energy_start = snapshot()
    start = time.monotonic()
    stream = torch.as_tensor(data['stream'].astype(np.int32), device='cuda')
    features = torch.as_tensor(observations(data, mode)[:meta['train_shapes'] + 1], device='cuda')
    confidence = torch.as_tensor(oof['features'], device='cuda')
    probability = torch.as_tensor(oof['prob'], device='cuda')
    offsets = torch.arange(LENGTH, device='cuda')
    model = BookWriter(meta['train_shapes'], hybrid).cuda()
    adam = torch.optim.AdamW(model.parameters(), lr=.001, fused=True)
    trajectory = []
    seen = np.zeros(len(starts), bool)
    schedule = hashlib.sha256()
    for step in range(1, STEPS + 1):
        selected = rng.integers(len(starts), size=BATCH)
        seen[selected] = True
        selected_starts = starts[selected]
        schedule.update(selected_starts.astype('<i8').tobytes())
        x = torch.as_tensor(selected_starts, device='cuda')[:, None] + offsets
        ids, labels = stream[x].long(), stream[x + 1].long()
        adam.zero_grad(set_to_none=True)
        factor = min(1., step / 200) * (.15 + .85 * (1 + math.cos(math.pi * step / STEPS)) / 2)
        adam.param_groups[0]['lr'] = .001 * factor
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model.loss(features[ids], ids, labels, confidence[x + 1], probability[x + 1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adam.step()
        if step == 1 or step % 200 == 0:
            value = float(loss.detach())
            elapsed = time.monotonic() - start
            peak = torch.cuda.max_memory_allocated()
            print(name, step, round(value, 4), round(elapsed, 2), peak, flush=True)
            if not math.isfinite(value) or elapsed > 1500 or peak > 12 * 2**30:
                raise RuntimeError('Resource or numeric limit exceeded')
            trajectory.append([step, value, elapsed])
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    energy = consumed(energy_start, snapshot())
    tmp = weights.with_suffix('.tmp')
    save_file({k: v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(tmp))
    tmp.replace(weights)
    dump(receipt, dict(mode=mode, hybrid=hybrid, checkpoint=sha(weights), bytes=weights.stat().st_size,
                       params=sum(p.numel() for p in model.parameters()), seconds=seconds, energy=energy,
                       trace=trajectory, updates=STEPS, scored=STEPS*BATCH*(LENGTH-5),
                       unique_windows=int(seen.sum()), schedule_sha=schedule.hexdigest(),
                       peak_allocated=torch.cuda.max_memory_allocated(), corpus=meta['corpus'],
                       torch_version=str(torch.__version__), device=torch.cuda.get_device_name()))
    print('saved', name, energy, flush=True)


def main():
    meta, data = corpus()
    counts = json.loads((PUB/'counts.json').read_text())
    if sha(WORK/'oof.npz') != counts['oof'] or counts['corpus'] != meta['corpus']:
        raise ValueError('Changed training data')
    starts = np.concatenate([np.arange(a, b - LENGTH) for split, _, _, _, a, b in meta['records']
                             if split == 'train' and b-a > LENGTH])
    palette = MODEL/'palette.npy'
    np.save(palette, data['tiles'][:meta['train_shapes'] + 1])
    dump(MODEL/'bundle.json', dict(palette=sha(palette), count=counts['inference_sha'],
                                    vocab=meta['train_shapes'], cell=32))
    with gpu_owner(), np.load(WORK/'oof.npz') as archive:
        oof = dict(archive)
        torch.set_num_threads(2)
        for mode in ('ink', 'byte'):
            for hybrid in (True, False):
                train(meta, data, starts, oof, mode, hybrid)


if __name__ == '__main__':
    main()
