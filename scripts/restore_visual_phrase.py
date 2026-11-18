"""Restore an indicated phrase from a typed or pictured Chinese claim with the calibrated writer."""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch

from ilm.visual_lm.span_backoff import SpanBackoff
from scripts.ask_visual_inference import input_strip
from scripts.infer_pictured_entry import page
from scripts.read_grounded_glyph_answer import respond
from scripts.train_visual_inference_writer import Corpus, PARENT, RESTORE, WORK, Rasterizer, load_visual_writer, sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("premise","claim"):
        source=p.add_mutually_exclusive_group(required=True)
        source.add_argument("--"+name+"-text")
        source.add_argument("--"+name+"-image",type=Path)
        p.add_argument("--"+name+"-cells",type=int)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--checkpoint-dir",type=Path,default=WORK/"paired")
    p.add_argument("--spelling-dir",type=Path,default=WORK/"span_backoff")
    p.add_argument("--memory",type=Path,default=PARENT/"memory.npy")
    p.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.mode="restore"
    torch.set_num_threads(2)
    r=Rasterizer()
    premise,claim=[input_strip(args,k,r) for k in ("premise","claim")]
    instruction=b"".join(r.tile(c) for c in RESTORE)
    cells=lambda x:np.frombuffer(x,np.uint8).reshape(-1,128)
    writer=load_visual_writer(args.checkpoint_dir,args.device)
    neural,stopped=respond(writer,np.load(args.memory),cells(premise),cells(instruction+claim),args.device)
    spelling=SpanBackoff(args.spelling_dir)
    coeff=np.load(args.spelling_dir/"coefficients.npy")
    raw,ended=spelling.choose(premise,claim,Corpus.mask_tile(),neural,stopped,coeff)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as work:
        dest=Path(work)/"answer";dest.mkdir()
        for name,b in (("premise",premise),("claim",claim),("neural",neural),("answer",raw)):
            page(b).save(dest/(name+".png"))
        (dest/"answer.bin").write_bytes(raw)
        write_json(dest/"receipt.json",dict(stopped=bool(ended),neural_stopped=bool(stopped),
            input_cells=[len(premise)//128,len(claim)//128],cells=len(raw)//128,
            answer_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=sha(args.checkpoint_dir/"final.safetensors"),
            spelling_hashes={name:sha(args.spelling_dir/name) for name in ("neighbors.npz","coefficients.npy","glyphs.npy")}))
        dest.rename(args.output)
    print(args.output/"answer.png")


if __name__=="__main__":
    main()
