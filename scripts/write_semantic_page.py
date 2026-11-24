"""Read a bilingual fact-card PNG and write an English paragraph in physical glyphs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_semantic_page import MODEL,WORK,PUB,KEYS,ZH,VALUES,WIDTH,Rasterizer
from scripts.train_semantic_page import read,palette,load,sha


def card(facts,language):
    raster=Rasterizer()
    rows=[]
    for i,(k,v) in enumerate(facts):
        zh=language=="zh" or (language=="mixed" and i%2==0)
        line=ZH[KEYS.index(k)]+"："+VALUES.get(v,v) if zh else k+": "+v
        if len(line)>WIDTH:raise ValueError("Row exceeds the supported card width")
        bits=[np.frombuffer(raster.tile(c),np.uint8) for c in line]
        rows.append(np.pad(bits,((0,WIDTH-len(bits)),(0,0))))
    if not rows:raise ValueError("Empty record")
    return np.array(rows,np.uint8)


def card_image(packed):
    rows,width=packed.shape[:2]
    bits=np.unpackbits(packed,axis=-1).reshape(rows,width,32,32)
    return Image.fromarray(255-255*bits.transpose(0,2,1,3).reshape(rows*32,width*32))


def read_card(path):
    with Image.open(path) as im:
        im=Image.alpha_composite(Image.new("RGBA",im.size,"white"),im.convert("RGBA")).convert("L")
        a=np.asarray(im)<128
    h,w=a.shape
    if w%32 or h%32 or w>WIDTH*32 or h>32*32:
        raise ValueError("Use a row-aligned card of 32-pixel cells, up to 40 columns and 32 rows")
    cells=a.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(h//32,w//32,1024)
    return np.pad(np.packbits(cells,axis=-1),((0,0),(0,WIDTH-w//32),(0,0)))


def printed(pixels,columns=740):
    """Tighten only blank sidebearings; wrap words using physically blank glyphs."""
    words=[];word=[]
    for tile in pixels:
        cell=np.unpackbits(tile).reshape(32,32)
        ink=np.flatnonzero(cell.any(0))
        if len(ink):
            word.append(Image.fromarray(255-255*cell[:,ink[0]:ink[-1]+1]))
        elif word:
            words.append(word);word=[]
    if word:words.append(word)
    x=y=0;positions=[]
    for w in words:
        width=sum(im.width+2 for im in w)-2
        if x and x+width>columns:x=0;y+=39
        for glyph in w:
            positions.append((glyph,x,y));x+=glyph.width+2
        x+=7
    width=max([1]+[x+im.width for im,x,y in positions])
    canvas=Image.new("L",(width+32,y+64),255)
    for im,x,y in positions:canvas.paste(im,(x+16,y+16))
    return canvas


@torch.inference_mode()
def render(net,packed,device,listener=None):
    features=observations({"tiles":packed.reshape(-1,128)},"ink")
    x=torch.as_tensor(features,device=device).reshape(len(packed),WIDTH,256)
    context=net.encode_rows(x)[None]
    visible=torch.as_tensor(packed.any(axis=(1,2)),device=device)[None]
    if listener is not None:
        from ilm.visual_lm.semantic_listener import beam_write
        _,(ids,n)=beam_write(net,context,visible,palette(device),listener)
        return ids[0,:n.item()].cpu().numpy(),True
    ids,n,stop=net.write(context,visible,palette(device))
    return ids[0,:n.item()].cpu().numpy(),bool(stop.item())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",type=Path,help="One printed field per row of 32-pixel cells")
    group.add_argument("--record",type=Path,help="JSON object of E2E fields and values to render")
    ap.add_argument("--language",choices=("en","zh","mixed"),default="zh")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",choices=("cuda","cpu"),default="cpu")
    ap.add_argument("--greedy",action="store_true",help="Use the faster original decoder without readback")
    args=ap.parse_args()
    if args.device=="cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(2)
    packed=read_card(args.input) if args.input else card(list(read(args.record).items()),args.language)
    from scripts.train_semantic_listener import load_listener
    listener=None if args.greedy else load_listener(args.device)
    ids,stopped=render(load("ink",args.device),packed,args.device,listener)
    pixels=np.load(MODEL/"palette.npy")[ids]
    chars=read(MODEL/"readback.json")
    text="".join(chars[i] for i in ids)
    args.output.mkdir(exist_ok=False,parents=True)
    card_image(packed).save(args.output/"input.png")
    printed(pixels).save(args.output/"answer.png")
    (args.output/"answer.bin").write_bytes(pixels.tobytes())
    (args.output/"answer.txt").write_text(text+"\n")
    print(f"Stopped={stopped}, glyphs={len(ids)}\n{text}")


if __name__=="__main__":
    main()
