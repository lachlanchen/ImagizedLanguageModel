"""Fit conditioned and source-ablated visual phrase writers on real aligned insertions."""
import argparse
import fcntl
import hashlib
import time
from itertools import chain

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_phrase_writer import VisualPhraseWriter
from scripts.evaluate_visual_spans import insert_runs
from scripts.train_visual_edit_translation import bank, inputs
from scripts.train_visual_spans import OUT, ROOT, WORK, frozen, guard, load, require, sha, tensor_pad, tiles, write_new

RUN = ROOT/'artifacts/visual_phrase_writer_v1'
PARENT = WORK/'marginal/editor.safetensors'
PARENT_HASH = '0083101ef67aeeba6c37b0f73eabec992a0970afbd10f486be2b262ecab76383'
ARMS = ('source', 'prefix')


def initialize():
    require(sha(PARENT) == PARENT_HASH, 'Marginal parent differs')
    torch.manual_seed(3129)
    state = load_file(str(PARENT))
    net = VisualPhraseWriter(state['landmarks'])
    missing, extra = net.load_state_dict(state, strict=False)
    require(set(missing) == {'left.weight', 'end.weight', 'end.bias'} and not extra, 'Incomplete parent')
    return net


def training_rows():
    old = bank()
    require(np.array_equal(old, tiles()[:len(old)]), 'Training bitmap identities differ')
    return list(chain(inputs('train'), frozen('train')))


def batches(rows, rng):
    # Group similar lengths within shuffled pools to reduce padded recurrent work.
    order = rng.permutation(len(rows))
    groups = []
    for block in np.array_split(order, 32):
        ordered = sorted(block.tolist(), key=lambda i: len(rows[i]['y']))
        groups.extend(ordered[i:i+96] for i in range(0, len(ordered), 96))
    rng.shuffle(groups)
    return groups


def tensors(rows, features, bits, net, question, ablate):
    x, n = tensor_pad([r['x'] for r in rows], features.device)
    y, _ = tensor_pad([r['y'] for r in rows], features.device)
    mem, initial, hist = net.contexts(features[x], n, bits[y], question.expand(len(rows), -1, -1),
                                      ablate=ablate)
    return mem, initial, hist, y, n


def fit(arm, rows, spans):
    began = time.monotonic()
    guard(began)
    folder = RUN/arm
    folder.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    net = initialize().cuda()
    net.motor.requires_grad_(False)
    net.action.requires_grad_(False)
    ink = torch.from_numpy(np.unpackbits(tiles(), axis=1)).cuda().bool()
    features = visual_input(ink, net.landmarks)
    bits = ink[:, net.landmarks]
    q = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=.0006, weight_decay=.01)
    rng = np.random.default_rng(3129)
    schedule = hashlib.sha256()
    history = []
    ablate = arm == 'prefix'
    for epoch in range(3):
        nll = 0.
        cells = phrases = updates = 0
        for ids in batches(rows, rng):
            schedule.update(np.array(ids, dtype='<i4').tobytes())
            fields = [(b, c, pos, pos+1 == k) for b, i in enumerate(ids) for c, j, k in spans[i]
                      for pos in range(j, k)]
            if not fields:
                continue
            batch, cursor, pos, end = torch.tensor(fields, device='cuda').T
            mem, _, hist, y, _ = tensors([rows[i] for i in ids], features, bits, net, q, ablate)
            at = net.boundary(mem, batch, cursor, ablate=ablate)
            draw, logits = net.losses(at, hist, batch, pos, bits[y[batch, pos]])
            loss = draw.mean() + F.binary_cross_entropy_with_logits(logits, end.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2., error_if_nonfinite=True)
            opt.step()
            nll += float(loss.detach()) * len(fields)
            cells += len(fields)
            phrases += int(end.sum())
            updates += 1
            if updates % 128 == 0:
                guard(began)
        history.append(dict(epoch=epoch+1, cells=cells, phrases=phrases, updates=updates,
                            pictured_sentences=len(rows), nll_per_cell=nll/cells))
        print(arm, history[-1], flush=True)
    checkpoint = folder/'writer.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k,v in net.state_dict().items()}, str(checkpoint))
    write_new(folder/'fit.json', dict(history=history, parent_sha256=PARENT_HASH, schedule=schedule.hexdigest(),
              checkpoint_sha256=sha(checkpoint), trainable_parameters=sum(p.numel() for p in params),
              parameters=sum(p.numel() for p in net.parameters()), peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
              seconds=time.monotonic()-began, gpu=torch.cuda.get_device_name()))
    guard(began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS)
    options = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require('4090' in torch.cuda.get_device_name(), 'Expected one 4090')
        rows = training_rows()
        groups = {r['g'] for r in rows}
        for split in ('discovery','replication'):
            require(not groups.intersection(r['g'] for r in frozen(split)), 'Test chapter in training')
        spans = [list(insert_runs(r['a'])) for r in rows]
        for arm in ([options.arm] if options.arm else ARMS):
            fit(arm, rows, spans)


if __name__ == '__main__':
    main()
