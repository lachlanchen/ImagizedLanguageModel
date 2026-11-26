"""Translate a pictured glyph grid and write the network's 32 by 32 binary outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from ilm.visual_lm.neural_glyph_motor import MotorTranslator

MODEL=Path(__file__).resolve().parents[1]/"models/neural_glyph_motor"


def load_model(device):
    path=MODEL/"weights.safetensors"
    receipt=json.loads((MODEL/"fit.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt["weights"]:
        raise ValueError("Motor bundle digest mismatch")
    net=MotorTranslator(*receipt["choices"])
    net.load_state_dict(load_file(str(path)))
    return net.eval().to(device)


def grid_bits(path,n):
    with Image.open(path) as img:
        a=np.asarray(img.convert("L"))<128
    h,w=a.shape
    if not n or not 1<=n<=96 or h%32 or w%32 or n>(h//32)*(w//32):
        raise ValueError("Invalid occupied cell count or input grid geometry")
    return a.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,32,32)[:n].copy()


def save_ink(bits,path):
    n=len(bits)
    cols=min(32,max(1,n))
    rows=max(1,(n+cols-1)//cols)
    page=np.zeros((rows*cols,32,32),dtype=np.uint8)
    page[:n]=bits
    ink=page.reshape(rows,cols,32,32).transpose(0,2,1,3).reshape(rows*32,cols*32)
    Image.fromarray(255-ink*255).save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--cells",type=int,required=True)
    p.add_argument("--to",choices=("en","zh"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    torch.set_num_threads(2)
    net=load_model(args.device)
    x=torch.from_numpy(grid_bits(args.input,args.cells)).to(args.device)[None]
    ink,count,stop=net.write(x,torch.tensor([args.cells],device=x.device),("en","zh").index(args.to))
    save_ink(ink[0,:int(count[0])].cpu().numpy(),args.output)
    print(json.dumps(dict(cells=int(count[0]),stopped=bool(stop[0]))))


if __name__=="__main__":
    main()
