"""Replay the fixed open-writing illustrations through the released PNG entry point."""

import hashlib
import shutil
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_visual_discourse import BASE,MODEL,PUBLIC,read,dump
from scripts.write_visual_discourse import Writer
from scripts.evaluate_visual_discourse import draw


def main():
    torch.set_num_threads(3)
    model=Writer(arm="ink-pair");receipt={}
    for kind in ("instruction-en","instruction-zh","book-en","book-zh"):
        folder=BASE/"examples"/"ink-pair"/kind/"0";r=read(folder/"readout.json")
        source=read_image(folder/"input.png",r["cells"])
        out,stop,line=model.write([source],96)[0]
        h=hashlib.sha256(out.tobytes()).hexdigest()
        dest=MODEL/"examples"/kind;dest.mkdir(parents=True,exist_ok=True)
        for n in ("input.png","answer.png","reference.png","readout.json"):shutil.copyfile(folder/n,dest/n)
        draw(dest,"cpu-answer",out)
        receipt[kind]=dict(cpu_sha256=h,cuda_sha256=r["ink_sha256"],same=h==r["ink_sha256"],stop=stop,transcript=line)
    dump(PUBLIC/"physical-open.json",receipt)
    print({k:r["same"] for k,r in receipt.items()})


if __name__=="__main__":main()
