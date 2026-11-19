"""Learn a glyph metric from alternate typefaces and fit a dense visual-semantic map."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
from time import monotonic

from fontTools.ttLib import TTFont
import numpy as np
from PIL import ImageFont
from safetensors.torch import load_file,save_file
import torch
from torch.nn import functional as F
from transformers import BertForQuestionAnswering,BertTokenizerFast

from ilm.visual_lm.font_invariant_reader import GlyphMetric
from scripts.fit_continuous_page_reader import (
    FONTDIR,FONTS,PUBLIC as PARENT_PUBLIC,ROOT,WORK as PARENT,Renderer,sha,snapshot,write_json,
)

PUBLIC=ROOT/"references/evidence/font_invariant_reader_20261119"
WORK=ROOT/"artifacts/font_invariant_reader_v1"
TRAIN_FACES=(*[(f,2) for f in (FONTS[0],FONTDIR/"NotoSansCJK-Light.ttc",FONTDIR/"NotoSansCJK-Black.ttc",
              FONTS[1],FONTDIR/"NotoSerifCJK-Black.ttc")],
             (Path("/usr/share/fonts/truetype/arphic/uming.ttc"),0),
             (Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),0))
KAI=(Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),0)
SEED=37431
WIDTH=12288


class Face(Renderer):
    def __init__(self,desc):
        path,index=desc
        self.font=ImageFont.truetype(str(path),26,index=index)
        with TTFont(path,fontNumber=index) as font:
            self.coverage=set(font.getBestCmap())


def bindings():
    refs=[Path(__file__),ROOT/"scripts/evaluate_font_invariant_reader.py",ROOT/"ilm/visual_lm/font_invariant_reader.py",
          ROOT/"references/font_invariant_reader_protocol.md",PARENT_PUBLIC/"fit.json",PARENT_PUBLIC/"result.json"]
    return {**{str(x.relative_to(ROOT)):sha(x) for x in refs},
            **{p.name:sha(p) for p,i in (*TRAIN_FACES,KAI,(FONTS[2],2))}}


def tables():
    source=snapshot()
    tok=BertTokenizerFast.from_pretrained(source,local_files_only=True)
    teacher=BertForQuestionAnswering.from_pretrained(source,local_files_only=True)
    emb=teacher.bert.embeddings.word_embeddings.weight.detach()
    fonts=[Face(x) for x in TRAIN_FACES]
    rows,indices=[],[]
    for c,i in sorted(tok.get_vocab().items(),key=lambda kv:kv[1]):
        if len(c)!=1 or c.isspace() or tok(c,add_special_tokens=False).input_ids!=[i]:
            continue
        cells=[font.cell(c) if ord(c) in font.coverage else bytes(128) for font in fonts]
        if sum(any(x) for x in cells)>=2:
            rows.append(b"".join(cells))
            indices.append(i)
    packed=np.frombuffer(b"".join(rows),np.uint8).reshape(len(rows),len(fonts),128)
    bits=torch.from_numpy(np.unpackbits(packed,axis=-1)).cuda()
    return bits,bits.any(-1).bool(),emb[indices].cuda()


def fit():
    WORK.mkdir()
    orig=json.loads((PARENT_PUBLIC/"fit.json").read_text())
    for name in ("config.json","context.safetensors"):
        assert sha(PARENT/name)==orig[name.split('.')[0]+"_sha256"]
        os.link(PARENT/name,WORK/name)
    pix,available,targets=tables()
    torch.manual_seed(SEED)
    metric=GlyphMetric().cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(metric.parameters(),lr=.0015)
    steps=2000
    trace=[]
    epoch_cost=0.
    torch.cuda.reset_peak_memory_stats()
    start=monotonic()
    for step in range(steps):
        idx=torch.randperm(len(pix),device="cuda")[:256]
        select=torch.rand(available[idx].shape,device="cuda").masked_fill(~available[idx],-1).topk(2).indices
        batch=pix[idx[:,None],select].reshape(256,2,32,32).transpose(0,1)
        # Shift both axes by at most one cell pixel. Erase rolled-in border pixels.
        moves=torch.randint(-1,2,(2,2)).tolist()
        views=[]
        for x,(dy,dx) in zip(batch,moves,strict=True):
            x=F.pad(x,(1,1,1,1))[:,1-dy:33-dy,1-dx:33-dx]
            views.append(x)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            encoded=metric(torch.cat(views)).float()
        a,b=encoded.chunk(2)
        logits=a@b.T/.06
        labels=torch.arange(len(a),device="cuda")
        loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(metric.parameters(),1.,error_if_nonfinite=True)
        opt.param_groups[0]["lr"]=(.00015+.00135*(1+math.cos(math.pi*step/steps))/2)*min(1,(step+1)/40)
        opt.step()
        epoch_cost+=float(loss.detach())
        if (step+1)%200==0:
            trace.append(epoch_cost/200);epoch_cost=0.
            print(step+1,trace[-1],flush=True)
            if monotonic()-start>900 or torch.cuda.max_memory_allocated()>15*1024**3:
                raise RuntimeError("Metric fit exceeded its allowance")
    metric.eval()
    with torch.no_grad():
        projection=torch.randn(192,WIDTH,device="cuda")
        gram=torch.zeros(WIDTH,WIDTH,device="cuda")
        cross=torch.zeros(WIDTH,targets.shape[1],device="cuda")
        for face in range(len(TRAIN_FACES)):
            which=torch.where(available[:,face])[0]
            for ids in which.split(512):
                features=F.relu(metric(pix[ids,face])@projection)*(2/WIDTH)**.5
                gram.addmm_(features.T,features)
                cross.addmm_(features.T,targets[ids])
        gram.diagonal().add_(.003)
        regression=torch.cholesky_solve(cross,torch.linalg.cholesky(gram))
        assert torch.isfinite(regression).all()
    seconds=monotonic()-start;peak=torch.cuda.max_memory_allocated()
    if seconds>900 or peak>15*1024**3:raise RuntimeError("Visual fit exceeded its allowance")
    markers=load_file(str(PARENT/"vision.safetensors"))["markers"]
    save_file({k:v.detach().cpu().contiguous() for k,v in metric.state_dict().items()},str(WORK/"metric.safetensors"))
    save_file(dict(projection=projection.cpu(),regression=regression.cpu(),markers=markers),str(WORK/"vision.safetensors"))
    fit=dict(updates=steps,glyph_classes=len(pix),images=int(available.sum()),seconds=seconds,peak_bytes=peak,
             training_loss=trace,weights={n:sha(WORK/n) for n in ("metric.safetensors","vision.safetensors","context.safetensors","config.json")},
             metric_parameters=sum(p.numel() for p in metric.parameters()),regression_coefficients=regression.numel(),
             projection_coefficients=projection.numel(),registration=sha(PUBLIC/"registration.json"))
    write_json(WORK/"fit.json",fit);write_json(PUBLIC/"fit.json",fit)
    print(fit,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register",action="store_true")
    reg=PUBLIC/"registration.json"
    if ap.parse_args().register:
        PUBLIC.mkdir(parents=True,exist_ok=True)
        if reg.exists():raise FileExistsError(reg)
        write_json(reg,bindings());return
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0" and bindings()==json.loads(reg.read_text())
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        fit()
        from scripts.evaluate_font_invariant_reader import evaluate
        evaluate()


if __name__=='__main__':
    main()
