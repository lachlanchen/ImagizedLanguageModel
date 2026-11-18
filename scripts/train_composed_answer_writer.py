"""Fit and autonomously evaluate a source-span and physical-image writer on real DuReader answers."""
from pathlib import Path
from time import monotonic
import argparse
import fcntl
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_model, save_model

from ilm.visual_lm.composed_answer_writer import COPY, WRITE, STOP, MAX_SPAN, WIDTH, ComposedAnswerWriter, descriptors
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.train_paired_passage import ROOT, unpack_sequences
from scripts.prepare_pictured_entry_answers import load, sha, write_json

DATA = ROOT / "data/composed_chinese_answers_v1"
OUT = ROOT / "artifacts/composed_answer_writer_v1"
PRIOR = ROOT / "models/pictured_passage_zh/mixed_evidence_reader.safetensors"
REG = ROOT / "references/evidence/composed_writer_registration.json"
SEED = 19401
EPOCHS = 18
BATCH = 32
LIMIT = 64
ARMS = ("copy", "finite", "raster", "exchanged")


def padded(seqs, fill=0):
    result = np.full((len(seqs), max(map(len, seqs))), fill, np.int64)
    for i, seq in enumerate(seqs):
        result[i, :len(seq)] = seq
    return result


class Split:
    def __init__(self, split):
        path = DATA / split
        receipt = load(DATA / "preparation.json")[split]
        for name, expected in receipt["files"].items():
            if sha(path / name) != expected:
                raise ValueError(f"Changed corpus file {split}/{name}")
        self.tiles = np.load(path / "pixels.npy")
        self.p, self.q, self.y = [unpack_sequences(path / (s + ".npz"))
                                  for s in ("passages", "questions", "answers")]
        self.groups = [r["kind"] for r in load(path / "sources.json")]


def segments(p, q, y, memory_indices):
    """Longest available physical run, preferring passage on exact ties."""
    seq = np.concatenate((p, q))
    offset = 0
    cursor = -1
    result = []
    while offset < len(y):
        starts = np.flatnonzero(seq == y[offset])
        length = int(bool(len(starts)))
        for n in range(2, min(len(y) - offset, MAX_SPAN) + 1):
            idx = starts + n - 1
            legal = (idx < len(seq)) & ((starts < len(p)) == (idx < len(p)))
            keep = starts[legal]
            keep = keep[seq[keep+n-1] == y[offset+n-1]]
            if not len(keep):
                break
            starts, length = keep, n
        if length:
            start = int(starts[0])
            result.append((COPY, start, length, cursor, -1, offset, starts))
            cursor = start + length
            offset += length
        else:
            result.append((WRITE, 0, 1, cursor, memory_indices[int(y[offset])], offset, []))
            offset += 1
    return result + [(STOP, 0, 0, cursor, -1, offset, [])]


def tensor(value, device):
    return torch.as_tensor(value, device=device)


@torch.no_grad()
def source_fields(prior, ds, questions, device):
    features = prior.features(ds.tiles)
    fields = []
    for i in range(0, len(ds.p), BATCH):
        ps, qs = ds.p[i:i+BATCH], questions[i:i+BATCH]
        p, q = [tensor(padded(seqs), device) for seqs in (ps, qs)]
        plen, qlen = [tensor(list(map(len, seqs)), device) for seqs in (ps, qs)]
        _, starts, ends, _ = prior.reader(features, p, q, plen, qlen)
        for j, (passage, question) in enumerate(zip(ps, qs, strict=True)):
            length = len(passage)
            feat = np.zeros((length + len(question), 6), np.float32)
            feat[length:, 0] = 1
            feat[:, 1] = np.concatenate((np.linspace(0, 1, length), np.linspace(0, 1, len(question))))
            feat[:, 2] = np.concatenate((np.isin(passage, question), np.isin(question, passage)))
            for k, scores in enumerate((starts, ends), 3):
                z = scores[j, :length]
                feat[:length, k] = (z - z.mean()).clamp(-12, 12).cpu().numpy()
            feat[:, 5] = np.log1p(np.arange(len(feat))) / 6
            fields.append(feat)
    return fields


def batch_inputs(ds, questions, fields, indices, device):
    seqs = [np.concatenate((ds.p[i], questions[i])) for i in indices]
    ids = tensor(padded(seqs), device)
    valid = torch.arange(ids.shape[1], device=device) < tensor(list(map(len, seqs)), device)[:, None]
    f = np.zeros((*ids.shape, 6), np.float32)
    for j, i in enumerate(indices):
        f[j, :len(fields[i])] = fields[i]
    return ids, tensor(f, device), valid


def teacher_batch(ds, alignments, indices, shape):
    batch, width = len(indices), max(len(alignments[i]) for i in indices)
    target = np.full((batch, width), -100)
    starts, ends, cur, mem = [np.zeros((batch, width), np.int64) for _ in range(4)]
    first, last = np.zeros_like(starts), np.zeros_like(starts)
    mean_ids = np.zeros((batch, width, MAX_SPAN), np.int32)
    counts = np.ones_like(starts)
    alternatives = np.zeros((batch, width, shape), bool)
    for j, i in enumerate(indices):
        for k, (act, begin, n, pos, lit, a, aliases) in enumerate(alignments[i]):
            target[j,k], starts[j,k], ends[j,k] = act, begin, begin + max(n-1, 0)
            cur[j,k], mem[j,k] = pos, max(lit, 0)
            alternatives[j,k,aliases] = True
            if k+1 < len(alignments[i]):
                seq = ds.y[i][a:a+n]
                first[j,k+1], last[j,k+1] = seq[0], seq[-1]
                mean_ids[j,k+1,:n] = seq
                counts[j,k+1] = n
    return target, starts, ends, cur, mem, first, last, mean_ids, counts, alternatives


def train(net, prior, ds, memory, device):
    fields = source_fields(prior, ds, ds.q, device)
    lookup = {int(idx): k for k, idx in enumerate(memory)}
    alignments = [segments(p, q, y, lookup) for p,q,y in zip(ds.p,ds.q,ds.y,strict=True)]
    bits = tensor(np.unpackbits(ds.tiles, axis=1), device)
    thumb = descriptors(bits)
    mem = tensor(memory, device)
    opt = torch.optim.AdamW(net.parameters(), lr=.0007, weight_decay=.01)
    rng = np.random.default_rng(SEED)
    trace = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(ds.y))
        totals = []
        for i in range(0,len(order),BATCH):
            ix = order[i:i+BATCH]
            ids, f, valid = batch_inputs(ds, ds.q, fields, ix, device)
            data = teacher_batch(ds, alignments, ix, ids.shape[1])
            act,start,end,cur,literal,first,last,mean_ids,counts,aliases = [tensor(a,device) for a in data]
            ink = net.ink(thumb)
            x, h = net.encode(ink[ids], f, valid)
            mask = torch.arange(MAX_SPAN, device=device) < counts[...,None]
            mean = (ink[mean_ids] * mask.unsqueeze(-1)).sum(-2) / counts[...,None]
            feedback = net.feedback_ink(ink[first], mean, ink[last])
            feedback = feedback * (torch.arange(act.shape[1], device=device)>0)[None,:,None]
            state, _ = net.recurrence(feedback,h)
            gate, pointer, focus = net.plan(x,f,valid,state,cur)
            loss = F.cross_entropy(gate.flatten(0,1), act.flatten())
            copy = act==COPY
            write = act==WRITE
            if copy.any():
                logp = pointer[copy].log_softmax(-1).masked_fill(~aliases[copy], -1e4)
                loss = loss - logp.logsumexp(-1).mean()
                ending = net.endings(x,f,valid,focus,start)
                loss = loss + F.cross_entropy(ending[copy],end[copy])
            if write.any():
                ctx = focus[write]
                logits = net.bitmap_scores(ctx, ink[mem])
                anchor = bits[mem[logits.detach().argmax(-1)]]
                target = bits[mem[literal[write]]].float()
                raster = net.raster_logits(ctx, anchor)
                loss = loss + F.cross_entropy(logits,literal[write]) + F.binary_cross_entropy_with_logits(raster,target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            totals.append(loss.detach())
        trace.append(float(torch.stack(totals).mean()))
        print(f"epoch {epoch+1}/{EPOCHS} loss {trace[-1]:.4f}",flush=True)
    return trace


class InputImages:
    """Compact exactly the pixels in the supplied inputs before invoking any model."""
    def __init__(self, passages, questions):
        lengths = [len(seq) for seq in (*passages,*questions)]
        self.tiles, ids = np.unique(np.concatenate((*passages,*questions)), axis=0, return_inverse=True)
        sequences = np.split(ids, np.cumsum(lengths)[:-1])
        self.p, self.q = sequences[:len(passages)], sequences[len(passages):]


@torch.inference_mode()
def generate(net, prior, memory_tiles, inputs, arm, device):
    if arm not in ("copy","finite","raster"):
        raise ValueError(arm)
    fields = source_fields(prior, inputs, inputs.q, device)
    all_bits = tensor(np.unpackbits(inputs.tiles,axis=1),device)
    ink = net.ink(descriptors(all_bits))
    memory_bits = tensor(np.unpackbits(memory_tiles,axis=1),device)
    memory_ink = net.ink(descriptors(memory_bits))
    outputs, origins, stopped = [], [], []
    for offset in range(0,len(inputs.p),BATCH):
        ix = list(range(offset,min(offset+BATCH,len(inputs.p))))
        ids, f, valid = batch_inputs(inputs,inputs.q,fields,ix,device)
        x,h = net.encode(ink[ids],f,valid)
        count = len(ix)
        rows = torch.arange(count,device=device)
        fb = x.new_zeros(count,1,WIDTH)
        cursor = torch.full((count,1), -1, device=device)
        buf = [[] for _ in ix]
        labels = [[] for _ in ix]
        done = np.zeros(count,bool)
        finished = np.zeros(count,bool)
        for _ in range(LIMIT+1):
            state,h = net.recurrence(fb,h)
            gate,pointer,focus = net.plan(x,f,valid,state,cursor)
            if arm=="copy":
                gate[:,:,WRITE] = -torch.inf
            action = gate[:,0].argmax(-1).cpu().numpy()
            start = pointer.argmax(-1)
            stop = net.endings(x,f,valid,focus,start).argmax(-1) + 1
            beginnings = start[:,0].tolist()
            endings = stop[:,0].tolist()
            selected = x.new_zeros(count,3,WIDTH)
            writers = np.flatnonzero((action==WRITE) & ~done)
            if len(writers):
                w = tensor(writers,device)
                context = focus[w,0]
                literal = net.bitmap_scores(context,memory_ink).argmax(-1)
                tile = memory_bits[literal]
                if arm=="raster":
                    tile = (net.raster_logits(context,tile)>=0).to(torch.uint8)
                glyphs = np.packbits(tile.cpu().numpy(),axis=1)
                ink_feedback = net.ink(descriptors(tile))
                selected[w] = ink_feedback[:,None]
            for j in np.flatnonzero(~done):
                if action[j]==STOP:
                    finished[j] = done[j] = True
                    continue
                if action[j]==COPY:
                    l,r = beginnings[j],endings[j]
                    idx = ids[j,l:r]
                    part = inputs.tiles[idx.tolist()]
                    selected[j] = torch.stack((ink[idx[0]],ink[idx].mean(0),ink[idx[-1]]))
                    cursor[j,0] = r
                else:
                    k = int(np.searchsorted(writers,j))
                    part = glyphs[k:k+1]
                available = LIMIT - len(labels[j])
                buf[j].append(part[:available].tobytes())
                labels[j].extend([int(action[j])] * min(available,len(part)))
                if len(part)>available:
                    done[j] = True
            if done.all():
                break
            fb = net.feedback_ink(*selected.unbind(1))[:,None]
        outputs.extend(b"".join(parts) for parts in buf)
        origins.extend(labels)
        stopped.extend(finished.tolist())
    return outputs, origins, stopped


def archive(path, streams, origins, stops):
    lens = [len(s)//128 for s in streams]
    assert lens==list(map(len,origins)) and len(stops)==len(lens)
    np.savez_compressed(path, pixels=np.frombuffer(b"".join(streams),np.uint8).reshape(-1,128),
                        origin=np.asarray(sum((list(x) for x in origins),[]),np.uint8),
                        offsets=np.r_[0,np.cumsum(lens)],stopped=np.array(stops,bool))


def bindings():
    paths = [Path(__file__), ROOT/"ilm/visual_lm/composed_answer_writer.py",
             ROOT/"references/composed_writer_protocol.md", PRIOR,
             ROOT/"scripts/audit_composed_writer_outputs.py",
             ROOT/"ilm/visual_lm/mixed_ink_reader.py", ROOT/"ilm/visual_lm/passage_composer.py",
             ROOT/"ilm/visual_lm/ordered_passage_planner.py", ROOT/"ilm/visual_lm/idiom_ink.py",
             ROOT/"scripts/train_paired_passage.py",DATA/"preparation.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def run():
    device = torch.device("cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or "4090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("The registered comparison uses physical GPU 0.")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    train_data = Split("train")
    dev = Split("development")
    memory = np.unique(np.concatenate(train_data.y))
    prior = OrderedPassageComposer().to(device).eval()
    load_model(prior,str(PRIOR))
    net = ComposedAnswerWriter().to(device)
    tiles = train_data.tiles[memory]
    OUT.mkdir()
    np.save(OUT/"memory.npy",tiles)
    t = monotonic()
    losses = train(net,prior,train_data,memory,device)
    elapsed = monotonic()-t
    save_model(net,str(OUT/"final.safetensors"))
    write_json(OUT/"fit.json",dict(seed=SEED,epochs=EPOCHS,batch_size=BATCH,losses=losses,
                updates=EPOCHS*((len(train_data.y)+BATCH-1)//BATCH),training_seconds=elapsed,
                writer_parameters=sum(p.numel() for p in net.parameters()),prior_parameters=sum(p.numel() for p in prior.parameters()),
                memory_bitmaps=len(tiles),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_sha256=sha(OUT/"final.safetensors"),memory_sha256=sha(OUT/"memory.npy")))
    del net
    net = ComposedAnswerWriter().to(device).eval()
    load_model(net,str(OUT/"final.safetensors"))
    passages = [dev.tiles[s] for s in dev.p]
    questions = [dev.tiles[s] for s in dev.q]
    inputs = InputImages(passages,questions)
    raw = [prior.answer(p,q) for p,q in zip(passages,questions,strict=True)]
    streams, stops = zip(*raw)
    archive(OUT/"released.npz",streams,[[COPY]*(len(s)//128) for s in streams],stops)
    for arm in ARMS:
        exchange = arm=="exchanged"
        presented = InputImages(passages,questions[1:]+questions[:1]) if exchange else inputs
        outputs = generate(net,prior,tiles,presented,"raster" if exchange else arm,device)
        archive(OUT/(arm+".npz"),*outputs)
        print(arm,"saved",len(outputs[0]),flush=True)
    write_json(OUT/"execution.json",dict(total_seconds=monotonic()-t,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                archives={arm:sha(OUT/(arm+".npz")) for arm in ("released",*ARMS)}))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","run"))
    args = parser.parse_args()
    if args.action=="register":
        if REG.exists():
            raise FileExistsError(REG)
        write_json(REG,bindings())
    else:
        if load(REG)!=bindings():
            raise RuntimeError("Registered sources changed")
        with (ROOT/"artifacts/composed_answer_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
