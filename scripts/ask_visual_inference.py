"""Read a Chinese premise and claim from text or a glyph grid and write the answer as an image."""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.infer_continuous_image_language import image_cells
from scripts.infer_pictured_entry import page
from scripts.read_grounded_glyph_answer import respond
from scripts.train_visual_inference_writer import (Corpus, DECIDE, RESTORE, WORK, PARENT, Rasterizer,
                                                   load_visual_writer, sha, write_json)


def input_strip(args,field,raster):
    png=getattr(args,field+"_image")
    count=getattr(args,field+"_cells")
    if png:
        values=image_cells(png,count).reshape(-1,1024)
        packed=np.packbits(values,axis=1).tobytes()
    else:
        if count is not None:
            raise ValueError(f"--{field}-cells applies to an image grid")
        value=getattr(args,field+"_text")
        if not value:
            raise ValueError(f"Empty {field}")
        packed=b"".join(Corpus.mask_tile() if field=="claim" and args.mode=="restore" and c=="□"
                        else raster.tile(c) for c in value)
    if not packed:
        raise ValueError(f"Empty {field} image")
    if field=="claim" and args.mode=="restore":
        count=sum(packed[j:j+128]==Corpus.mask_tile() for j in range(0,len(packed),128))
        if count!=1:
            raise ValueError("Restoration needs one □ in typed text or one matching mask tile in the claim grid")
    return packed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("decide","restore"),required=True)
    for f in ("premise","claim"):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--"+f+"-text")
        group.add_argument("--"+f+"-image",type=Path)
        parser.add_argument("--"+f+"-cells",type=int,help="Valid cells excluding trailing image-grid padding")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint-dir",type=Path,default=WORK/"paired")
    parser.add_argument("--memory",type=Path,default=PARENT/"memory.npy")
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    raster=Rasterizer()
    p=input_strip(args,"premise",raster)
    claim=input_strip(args,"claim",raster)
    q=b"".join(raster.tile(c) for c in (DECIDE if args.mode=="decide" else RESTORE))+claim
    net=load_visual_writer(args.checkpoint_dir,args.device)
    memory=np.load(args.memory)
    cells=lambda b:np.frombuffer(b,np.uint8).reshape(-1,128)
    answer,stopped=respond(net,memory,cells(p),cells(q),args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        dest=Path(tmp)/"result";dest.mkdir()
        for name,b in (("premise",p),("claim",claim),("answer",answer)):
            page(b).save(dest/(name+".png"))
        (dest/"answer.bin").write_bytes(answer)
        write_json(dest/"receipt.json",dict(mode=args.mode,stopped=stopped,premise_cells=len(p)//128,
                   claim_cells=len(claim)//128,answer_cells=len(answer)//128,answer_sha256=hashlib.sha256(answer).hexdigest(),
                   checkpoint_sha256=sha(args.checkpoint_dir/"final.safetensors"),memory_sha256=sha(args.memory)))
        dest.rename(args.output)
    print(args.output/"answer.png")


if __name__=="__main__":
    main()
