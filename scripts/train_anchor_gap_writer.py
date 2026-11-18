"""Fit the source copy planner and conditional gap writer on paired bitmap streams."""

import fcntl
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from ilm.visual_lm.idiom_ink import visual_input
from scripts.prepare_ordered_copy_anchors import OUT as DATA
from scripts.train_aligned_visual_translation import ROOT, MODELS, sha, write_new

OUT = ROOT / "artifacts/anchor_gap_writer_v1"
PARENT = MODELS / "marginal_glyph.safetensors"
SEED = 194207


def targets(n, m, pairs):
    keep = np.zeros(n, bool)
    close = np.zeros(m + 1, bool)
    left, right, offset = np.empty((3, m + 1), np.int16)
    previous_i = previous_j = -1
    for i, j in [*pairs, (n, m)]:
        if not (previous_i < i <= n and previous_j < j <= m):
            raise ValueError("Correspondences must be strictly increasing")
        sl = slice(previous_j + 1, j + 1)
        left[sl], right[sl] = previous_i, i
        offset[sl] = np.arange(j - previous_j)
        close[j] = True
        if i < n:
            keep[i] = True
        previous_i, previous_j = i, j
    return keep, left, right, offset, close


def prepare(path):
    with np.load(path) as archive:
        cells, pairs, anchors, source, target, xs, ys = (
            archive[k] for k in ("cells","pairs","offsets","source","target","x_offsets","y_offsets")
        )
    n, m = np.diff(xs), np.diff(ys)
    x = np.zeros((len(n), max(n)), np.int32)
    y = np.zeros((len(m), max(m)), np.int32)
    keep = np.zeros_like(x, dtype=bool)
    shape = (len(m), max(m)+1)
    close = np.zeros(shape, bool)
    left, right, offset = (np.zeros(shape, np.int16) for _ in range(3))
    for b, (sx, sy) in enumerate(zip(n, m)):
        x[b,:sx] = source[xs[b]:xs[b+1]]
        y[b,:sy] = target[ys[b]:ys[b+1]]
        bounds = pairs[anchors[b]:anchors[b+1]].tolist()
        (keep[b,:sx], left[b,:sy+1], right[b,:sy+1],
         offset[b,:sy+1], close[b,:sy+1]) = targets(sx, sy, bounds)
    assert keep.sum() == len(pairs) and close.sum() == len(pairs) + len(n)
    return cells, (x,y,n,m,keep,left,right,offset,close)


def train():
    path = DATA / "training.npz"
    assert sha(path) == "e363898b64668b2c713ce507fd8e9e67b9a4a4840aa65595ccdb0c352aeabb15"
    assert sha(PARENT) == "7ea3d30181f460712d25e2eb84e7cac57fd92b4ed953286db2bdbb84655c0789"
    OUT.mkdir()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    packed, data = prepare(path)
    rng = np.random.default_rng(SEED)
    state = load_file(str(PARENT))
    model = AnchorGapWriter(state["landmarks"], state["exemplars"])
    missing, extra = model.load_state_dict(
        {k:v for k,v in state.items() if not k.startswith(("stride.", "plan."))}, strict=False)
    assert not extra and set(missing) == {k for k in model.state_dict() if k == "edges" or k.startswith(("keep.","interval.","close_gap."))}
    indexes = {bytes(v):i for i,v in enumerate(packed)}
    lookup = torch.full((len(packed),), -1, dtype=torch.long)
    for j, picture in enumerate(np.packbits(state["exemplars"].numpy(), axis=-1)):
        lookup[indexes[bytes(picture)]] = j
    lookup = lookup.cuda()
    model.cuda()
    features = visual_input(torch.from_numpy(np.unpackbits(packed, axis=1)).cuda().bool(), model.landmarks)
    x,y,n,m,keep,left,right,offset,close = (torch.as_tensor(a,device="cuda") for a in data)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0006)
    start = time.monotonic()
    losses, steps = [], 0
    for epoch in range(6):
        batches = []
        for b in np.array_split(rng.permutation(len(x)), 16):
            b = b[np.argsort(data[3][b], kind="stable")]
            batches.extend(b[j:j+96] for j in range(0,len(b),96))
        rng.shuffle(batches)
        total = 0.
        for batch in batches:
            idx = torch.as_tensor(batch,device="cuda")
            nx,ny = int(n[idx].max()),int(m[idx].max())
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss = model.loss(features,x[idx,:nx].long(),y[idx,:ny].long(),n[idx],m[idx],
                                  keep[idx,:nx],left[idx,:ny+1].long(),right[idx,:ny+1].long(),
                                  offset[idx,:ny+1].long(),close[idx,:ny+1],lookup)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1., error_if_nonfinite=True)
            opt.step()
            steps += 1
            total += loss.item()*len(idx)
            if steps % 100 == 0:
                available = next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
                if int(available.split()[1]) < 24*2**20 or torch.cuda.max_memory_allocated() > 10*2**30 or time.monotonic()-start > 3600:
                    raise RuntimeError("Training resource ceiling")
        losses.append(total/len(x))
        print(epoch+1, steps, losses[-1], round(time.monotonic()-start,1),flush=True)
    ckpt = OUT / "writer.safetensors"
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(ckpt))
    write_new(OUT / "fit.json", dict(seed=SEED,epochs=6,steps=steps,loss=losses,rows=len(x),
        target_cells=int(data[3].sum()),source_cells=int(data[2].sum()),anchors=int(data[4].sum()),
        parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        seconds=time.monotonic()-start, peak_mib=torch.cuda.max_memory_allocated()/2**20,
        gpu=torch.cuda.get_device_name(),checkpoint_sha256=sha(ckpt),parent_sha256=sha(PARENT),training_sha256=sha(path)))


if __name__ == "__main__":
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train()
