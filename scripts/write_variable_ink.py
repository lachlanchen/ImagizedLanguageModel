"""Write a paragraph from a pictured record, keeping original ink in names."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from ilm.visual_lm.variable_ink import ROLES,bind_card,expand
from scripts.train_variable_ink import MODEL,BASE,PUB,read,bindings,load_writer
from scripts.train_semantic_listener import load_listener
from scripts.write_semantic_page import card,card_image,read_card,printed


def palette_and_prefixes():
    prefixes,anchors=bindings()
    physical=np.load(BASE/"palette.npy")
    actions=np.concatenate((physical,*[anchors[k] for k in ROLES]))
    return physical,actions,prefixes,anchors


class CardWriter:
    def __init__(self,device="cpu"):
        self.device=device
        self.writer=load_writer("ink",device)
        self.listener=load_listener(device,PUB,MODEL)
        self.physical,actions,self.prefixes,self.anchors=palette_and_prefixes()
        self.observed=torch.as_tensor(observations({"tiles":actions},"ink"),device=device)

    @torch.inference_mode()
    def write(self,packed):
        normalized,bound=bind_card(packed,self.prefixes,self.anchors)
        source=observations({"tiles":normalized.reshape(-1,128)},"ink")
        rows=torch.as_tensor(source,device=self.device).reshape(*normalized.shape[:2],256)
        context=self.writer.encode_rows(rows)[None]
        visible=torch.as_tensor(normalized.any(axis=(1,2)),device=self.device)[None]
        allowed=torch.ones((1,len(self.observed)),dtype=torch.bool,device=self.device)
        allowed[0,-len(ROLES):]=torch.tensor([role in bound for role in ROLES],device=self.device)
        _,(ids,length)=beam_write(self.writer,context,visible,self.observed,self.listener,allowed=allowed)
        return expand(ids[0,:length.item()].tolist(),self.physical,bound)


def transcript(pixels,physical):
    chars=read(BASE/"readback.json")
    lookup={tile.tobytes():c for tile,c in zip(physical[1:],chars[1:])}
    return "".join(lookup.get(tile.tobytes(),"\ufffc") for tile in pixels)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input",type=Path)
    g.add_argument("--record",type=Path,help="JSON object of restaurant fields, rendered before inference")
    ap.add_argument("--language",choices=("en","zh","mixed"),default="zh")
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    if a.device=="cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(2)
    source=read_card(a.input) if a.input else card(list(read(a.record).items()),a.language)
    writer=CardWriter(a.device); result=writer.write(source)
    a.output.mkdir(parents=True)
    card_image(source).save(a.output/"input.png")
    printed(result).save(a.output/"answer.png")
    (a.output/"answer.bin").write_bytes(result.tobytes())
    line=transcript(result,writer.physical)
    (a.output/"answer.txt").write_text(line+"\n")
    print(line)


if __name__=="__main__":
    main()
