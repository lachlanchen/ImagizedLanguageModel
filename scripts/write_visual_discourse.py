"""Write an open-ended answer or book continuation from an aligned page image."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.discourse_writer import DiscourseWriter
from ilm.visual_lm.portable_writer import read_image,read_text,DEFAULT_FONT,save_grid

DEFAULT=Path(__file__).resolve().parents[1]/"models/visual_discourse"


def encode_pages(palette, pages):
    values=list(map(bytes,palette))
    lookup={s:i for i,s in enumerate(values) if i>=2}
    sequences=[]
    for page in pages:
        if page.ndim!=2 or page.shape[1]!=128 or not len(page):raise ValueError("Expected one or more 32-pixel cells")
        ids=[]
        for glyph in map(bytes,page):
            if glyph not in lookup:
                lookup[glyph]=len(values);values.append(glyph)
            ids.append(lookup[glyph])
        sequences.append(ids)
    return np.frombuffer(b"".join(values),np.uint8).reshape(-1,128),sequences


@torch.inference_mode()
def generate(model,bank,sequences,n,limit,temperature=0.,seed=1946):
    if limit<1 or temperature<0:raise ValueError("Expected positive output cap and nonnegative temperature")
    generator=torch.Generator(device=bank.device).manual_seed(seed)
    history=[list(s) for s in sequences];answers=[[] for s in history];stopped=[False]*len(history)
    for _ in range(limit):
        active=[i for i,v in enumerate(stopped) if not v]
        if not active:break
        recent=[history[i][-(model.context-1):] for i in active]
        position=[len(s) for s in recent];width=max(position)+1
        raw=np.zeros((len(active),width),np.int32)
        for j,s in enumerate(recent):raw[j,:len(s)]=s
        ids=torch.as_tensor(raw,device=bank.device)
        where=torch.tensor(position,device=bank.device)[:,None]
        logits,pointer,gate=model.distributions(bank,ids,n,where)
        prob=logits.new_zeros((len(ids),len(bank)))
        prob[:,1:n]=logits[:,0].softmax(-1)*(1-gate)
        prob.scatter_add_(1,ids.long(),pointer[:,0]*gate)
        if temperature==0:choice=prob.argmax(-1)
        else:
            score=prob.log()/temperature
            choice=torch.multinomial(score.softmax(-1),1,generator=generator)[:,0]
        for row,idx in zip(active,choice.tolist(),strict=True):
            if idx==1:stopped[row]=True
            else:answers[row].append(idx);history[row].append(idx)
    return answers,stopped


class Writer:
    def __init__(self,bundle=DEFAULT,arm="ink-pair",device="cpu"):
        meta=json.loads((bundle/"glyphs.json").read_text())
        if hashlib.sha256((bundle/"glyphs.npy").read_bytes()).hexdigest()!=meta["sha256"]:raise ValueError("Changed physical palette")
        weights=bundle/(arm+".safetensors")
        receipt=json.loads((bundle/"release.json").read_text())
        if arm not in receipt["visual_models"] or hashlib.sha256(weights.read_bytes()).hexdigest()!=receipt["visual_models"][arm]:raise ValueError("Changed visual writer")
        self.model=DiscourseWriter().to(device)
        self.model.load_state_dict(load_file(str(weights),device=device));self.model.eval()
        self.palette=np.load(bundle/"glyphs.npy",allow_pickle=False);self.transcript=meta["transcript"];self.device=device

    @torch.inference_mode()
    def write(self,pages,limit,temperature=0,seed=1946):
        shapes,ids=encode_pages(self.palette,pages)
        features=torch.from_numpy(observations({"tiles":shapes},"ink")).to(self.device)
        bank=self.model.encode(features)
        seq,stopped=generate(self.model,bank,ids,len(self.palette),limit,temperature,seed)
        return [(shapes[np.asarray(a,dtype=int)],s,"".join(self.transcript[i] if i<len(self.transcript) else "�" for i in a)) for a,s in zip(seq,stopped,strict=True)]


def main():
    p=argparse.ArgumentParser(description=__doc__);src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",type=Path);src.add_argument("--text")
    p.add_argument("--cells",type=int,help="Occupied input cells, including meaningful trailing whitespace")
    p.add_argument("--output",required=True,type=Path);p.add_argument("--max-glyphs",type=int,default=96)
    p.add_argument("--temperature",type=float,default=0.);p.add_argument("--seed",type=int,default=1946)
    p.add_argument("--bundle",type=Path,default=DEFAULT);p.add_argument("--arm",choices=["ink-pair","ink-independent","ink-warm"],default="ink-pair")
    p.add_argument("--device",choices=["cuda","cpu"],default="cpu");a=p.parse_args()
    if a.device=="cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(3)
    raw=read_image(a.input,a.cells) if a.input else read_text(a.text,DEFAULT_FONT)
    model=Writer(a.bundle,a.arm,a.device)
    glyphs,stopped,transcript=model.write([raw],a.max_glyphs,a.temperature,a.seed)[0]
    a.output.mkdir(parents=True,exist_ok=False)
    save_grid(raw.tobytes(),a.output/"input.png")
    if len(glyphs):save_grid(glyphs.tobytes(),a.output/"answer.png")
    else:Image.new("L",(32,32),255).save(a.output/"answer.png")
    (a.output/"answer.bin").write_bytes(glyphs.tobytes())
    report=dict(observed_cells=len(raw),emitted_cells=len(glyphs),stopped=stopped,transcript=transcript)
    (a.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(transcript)


if __name__=="__main__":main()
