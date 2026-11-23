"""Fit a visual correspondence and matched writers using alternate printed questions."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.typeface_map import TypefaceMap, unpack
from scripts.fit_font_invariant_reader import Face, FONTS, KAI
from scripts.interval_teacher_corpus import ROOT, WORK as PREVIOUS, load_rows
from scripts.pointing_glyph_corpus import Row
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_span_choice import gpu_owner

WORK = ROOT / "artifacts/typeface_writer_v1"
PUBLIC = ROOT / "references/evidence/typeface_writer_20261124"
BASE = PREVIOUS / "distilled/page"
SEED = 241204
FACES = [(FONTS[0], 2), (FONTS[2], 2), KAI, (FONTS[1], 2)]
HELD = (Path("/usr/share/fonts/truetype/hanazono/HanaMinA.ttf"), 0)


def training_rows():
    path = PREVIOUS / "pairs_train.jsonl"
    receipt = json.loads((PREVIOUS / "corpus.json").read_text())
    assert sha(path) == receipt["additional_train"]["sha256"]
    return list(map(json.loads, path.read_text().splitlines()))


def glyph_views(rows, faces, packed):
    lookup = {bytes(row): i for i, row in enumerate(packed.numpy(), 1)}
    chars = sorted(set("".join(v["p"] + v["q"] + v["refs"][0] for v in rows)))
    by_id = {}
    # Coalesce identical canonical bitmaps, ignoring unrepresentable characters.
    for c in chars:
        if c.isspace() or ord(c) not in faces[0].coverage:
            continue
        idx = lookup.get(faces[0].cell(c))
        if idx is not None and idx not in by_id:
            by_id[idx] = c
    ids = sorted(by_id)
    cells = [[face.cell(by_id[i]) if ord(by_id[i]) in face.coverage else bytes(128)
              for face in faces] for i in ids]
    bits = torch.from_numpy(np.frombuffer(b"".join(b"".join(g) for g in cells),np.uint8).copy())
    return torch.tensor(ids), bits.reshape(len(ids),len(faces),128)


def fit_correspondence(rows, faces):
    path = WORK / "typeface.safetensors"
    if path.exists():
        raise FileExistsError(path)
    t = monotonic()
    torch.manual_seed(SEED)
    packed = load_file(str(BASE / "interface.safetensors"))["packed"]
    ids, views = glyph_views(rows, faces, packed)
    valid = views.any(-1).bool().cuda()
    bits = unpack(views.cuda()).reshape(len(ids),len(faces),32,32)
    net = TypefaceMap(ids.cuda(), torch.empty(len(ids),192,device="cuda")).cuda()
    net.metric.load_state_dict(load_file(str(BASE / "metric.safetensors")))
    net.metric.eval()
    with torch.no_grad():
        centers = torch.cat([net.metric(v[:,0]) for v in bits.split(256)])
    centers = torch.nn.Parameter(centers)
    net.metric.train()
    opt = torch.optim.AdamW([{"params":net.metric.parameters(),"lr":.0004},
                             {"params":[centers],"lr":.0015}], fused=True)
    trace = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1200):
        labels = torch.randperm(len(ids),device="cuda")[:192]
        selection = torch.rand(192,len(faces),device="cuda").masked_fill(~valid[labels],-1).argmax(1)
        x = bits[labels,selection]
        dy,dx = torch.randint(-1,2,(2,)).tolist()
        x = F.pad(x,(1,1,1,1))[:,1+dy:33+dy,1+dx:33+dx]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits = net.metric(x) @ F.normalize(centers,dim=1).T
        loss = F.cross_entropy(logits.float()/.04,labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*net.metric.parameters(),centers],1.,error_if_nonfinite=True)
        scale = .15 + .85*(1+math.cos(math.pi*step/1200))/2
        for group,lr in zip(opt.param_groups,(.0004,.0015),strict=True):
            group["lr"]=lr*scale
        opt.step()
        if (step+1)%100 == 0:
            trace.append(float(loss))
            print("correspondence",step+1,round(trace[-1],3),flush=True)
            if monotonic()-t>1200 or torch.cuda.max_memory_allocated()>20*2**30:
                raise RuntimeError("Visual correspondence exceeded its training allowance")
    net.prototypes.copy_(centers.detach())
    save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()}, str(path),
              metadata={"alphabet_sha256":hashlib.sha256(packed.numpy().tobytes()).hexdigest()})
    record(PUBLIC/"correspondence.json",dict(classes=len(ids),views=int(valid.sum()),steps=1200,
           presentations=1200*192,seconds=monotonic()-t,peak_bytes=torch.cuda.max_memory_allocated(),
           trainable=sum(p.numel() for p in net.metric.parameters())+centers.numel(),
           fonts={str(p):sha(p) for p,_ in FACES},sha256=sha(path),loss_samples=trace))


def train(arms):
    WORK.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    weights = json.loads((PREVIOUS/"distilled/page_fit.json").read_text())["weights"]
    assert all(sha(BASE/n)==h for n,h in weights.items())
    rows = training_rows()
    faces = [Face(v) for v in FACES]
    if "correspondence" in arms:
        fit_correspondence(rows,faces)
    writer_arms = [a for a in arms if a != "correspondence"]
    if not writer_arms:
        return
    material = load_rows()
    stats = Counter()
    for i,v in enumerate(rows):
        name = (i//2)%3 + 1
        font = faces[name]
        fields = v["p"],v["q"],v["refs"][0]
        if any(not c.isspace() and ord(c) not in font.coverage for s in fields for c in s):
            stats["omitted_"+str(name)] += 1
            continue
        material.append(Row("typeface_"+str(name),v["id"],*map(font,fields),b""))
        stats["included_"+str(name)] += 1
    record(PUBLIC/"exposures.json",dict(rows=len(material),views=stats,seed=SEED))
    for arm in writer_arms:
        folder=WORK/arm
        folder.mkdir()
        mapper=None
        if arm == "mapped":
            mapper = WORK/"typeface.safetensors"
            assert sha(mapper)==json.loads((PUBLIC/"correspondence.json").read_text())["sha256"]
        fit_rows(folder,BASE,weights,material,SEED,typeface_map=mapper)
        record(PUBLIC/(arm+"_fit.json"),json.loads((folder/"page_fit.json").read_text()))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms",nargs="+",choices=("correspondence","replay","mapped"))
    with gpu_owner():
        train(ap.parse_args().arms)
