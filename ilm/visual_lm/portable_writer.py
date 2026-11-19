"""Run a bundled visual writer on pictured or typed glyph streams with no training corpus."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
import torch

from ilm.visual_lm.composed_answer_writer import descriptors
from ilm.visual_lm.grounded_glyph_decoder import GroundedGlyphDecoder
from ilm.visual_lm.scoped_question import ScopedQuestion

DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def read_image(path, count):
    with Image.open(path) as im:
        pixels = np.asarray(im.convert("L")) < 128
    h,w = pixels.shape
    if h%32 or w%32:
        raise ValueError("Image dimensions must be multiples of 32")
    cells = pixels.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,1024)
    if count is not None:
        if not 1<=count<=len(cells):
            raise ValueError("Occupied cell count is outside this grid")
        cells=cells[:count]
    return np.packbits(cells,axis=1)


def read_text(text, font_path):
    # The same baseline and 26-pixel Noto Sans CJK SC face as the training renderer.
    from fontTools.ttLib import TTFont
    font=ImageFont.truetype(str(font_path),26,index=2)
    with TTFont(font_path,fontNumber=2) as face:
        coverage=set(face.getBestCmap())
    chars={}
    if not text:
        raise ValueError("Empty typed stream")
    for c in set(text):
        im=Image.new("L",(32,32),255)
        if not c.isspace() and unicodedata.category(c)!="Cf":
            if ord(c) not in coverage:raise ValueError(f"Missing glyph U+{ord(c):04X}")
            draw=ImageDraw.Draw(im)
            draw.text(((32-draw.textlength(c,font=font))/2,26),c,font=font,fill=0,anchor="ls")
            if im.getextrema()[0]>=128:raise ValueError(f"Empty glyph U+{ord(c):04X}")
        chars[c]=np.packbits(np.asarray(im)<128)
    return np.stack([chars[c] for c in text])


def save_grid(raw, path):
    count=len(raw)//128
    width=min(32,max(1,count))
    height=max(1,(count+width-1)//width)
    grid=np.zeros((height*width,32,32),dtype=np.uint8)
    grid[:count]=np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(-1,32,32)
    pixels=grid.reshape(height,width,32,32).transpose(0,2,1,3).reshape(height*32,width*32)
    Image.fromarray(255*(1-pixels)).save(path)


@torch.inference_mode()
def run(bundle, passage, question, device):
    manifest=json.loads((bundle/"writer.json").read_text())
    for name in ("final.safetensors","memory.npy"):
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=manifest["sha256"][name]:
            raise ValueError(f"Changed {name}")
    memory=np.load(bundle/"memory.npy",allow_pickle=False)
    if memory.dtype!=np.uint8 or memory.ndim!=2 or memory.shape[1]!=128:
        raise ValueError("Invalid shape memory")
    tiles=[bytes(128),*map(bytes,memory)]
    ids={v:i for i,v in enumerate(tiles[1:],1)}
    if len(ids)!=len(memory):raise ValueError("Duplicate stored shapes")

    def handles(sequence):
        index=[]
        for cell in map(bytes,sequence):
            if cell not in ids:
                ids[cell]=len(tiles)
                tiles.append(cell)
            index.append(ids[cell])
        return torch.tensor(index,device=device)[None]

    p,q=handles(passage),handles(question)
    model=GroundedGlyphDecoder()
    model.question=ScopedQuestion(ordered=True)
    model.load_state_dict(load_file(str(bundle/"final.safetensors")),strict=True)
    model.to(device).eval()
    bits=np.unpackbits(np.frombuffer(b"".join(tiles),np.uint8).reshape(-1,128),axis=1)
    ink=model.ink(descriptors(torch.from_numpy(bits).to(device)))
    mem=torch.arange(1,len(memory)+1,device=device)
    if "interval.safetensors" in manifest["sha256"]:
        from ilm.visual_lm.interval_writer import IntervalWriter
        path = bundle/"interval.safetensors"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
            raise ValueError("Changed source-run checkpoint")
        locator = IntervalWriter().to(device).eval()
        locator.load_state_dict(load_file(str(path)),strict=True)
        output,stopped=locator.generate(model,ink,p,q,model.write_key(ink[mem]),mem,limit=manifest["limit"])
    else:
        output,stopped=model.generate(ink,p,q,model.write_key(ink[mem]),mem,limit=manifest["limit"])
    return b"".join(tiles[i] for i in output[0].tolist() if i),bool(stopped[0])


def main(bundle=None, sentence_selection=False):
    parser=argparse.ArgumentParser(description=__doc__)
    if bundle is None:parser.add_argument("--bundle",type=Path,required=True)
    for field in ("passage","question"):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--"+field+"-image",type=Path)
        group.add_argument("--"+field+"-text")
        parser.add_argument("--"+field+"-cells",type=int)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--font",type=Path,default=DEFAULT_FONT)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    if sentence_selection:
        parser.add_argument("--evidence", choices=("whole","overlap"), default="whole",
                            help="Use the complete passage or its most overlapping sentence")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    streams=[]
    for field in ("passage","question"):
        img=getattr(args,field+"_image");n=getattr(args,field+"_cells")
        if img is None and n is not None:parser.error("A cell count requires an image")
        streams.append(read_image(img,n) if img else read_text(getattr(args,field+"_text"),args.font))
    selected = None
    if sentence_selection and args.evidence == "overlap":
        from ilm.visual_lm.pictured_sentence import locate
        path = bundle/"sentence_ends.npy"
        manifest = json.loads((bundle/"writer.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
            raise ValueError("Changed sentence boundary images")
        selected = locate(*streams, np.load(path, allow_pickle=False))
        streams[0] = streams[0][slice(*selected)]
    torch.set_num_threads(2)
    answer,stop=run(bundle if bundle is not None else args.bundle,*streams,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        dest=Path(tmp)/"answer";dest.mkdir()
        save_grid(answer,dest/"answer.png")
        (dest/"answer.bin").write_bytes(answer)
        (dest/"receipt.json").write_text(json.dumps(dict(stopped=stop,answer_cells=len(answer)//128,
            sha256=hashlib.sha256(answer).hexdigest(), **({"sentence_cells":selected} if selected else {})),indent=2)+"\n")
        dest.rename(args.output)
    print(args.output/"answer.png")


if __name__=="__main__":
    main()
