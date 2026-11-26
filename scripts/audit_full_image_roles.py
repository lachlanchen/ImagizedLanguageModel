"""Verify split isolation, complete source retention, raw answers and code-level reversals."""
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
import torch
from scripts.evaluate_full_image_roles import ARCHIVE,ARMS,rows
from scripts.prepare_full_image_roles import PUB,WORK,ROOT,BASE,BOOK,REHEARSAL,MODELS,Ink,read,dump,sha
from scripts.write_full_image_roles import PicturedTranslator
from ilm.visual_lm.full_image_roles import pictured_input


def image_cells(blob,n):
    with Image.open(BytesIO(blob)) as im:
        p=255-np.asarray(im.convert("L"))
    h,w=p.shape
    assert h%32==w%32==0 and np.isin(p,(0,255)).all()
    return np.packbits(p.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,1024)[:n]>0,axis=1)


def main():
    torch.set_num_threads(2)
    grade=read(PUB/"assessment.json")
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]==grade["archive"]
    initial=np.load(BOOK/"cells.npy")
    with np.load(REHEARSAL) as z:
        initial[z["indices"]]=z["cells"]
    allpairs=np.concatenate((initial,np.load(WORK/"events.npy")))
    seen=[{r.tobytes() for r in allpairs[:,t]} for t in (0,1)]
    pal=np.load(BASE/"palette.npy")
    ink=Ink(pal)
    names=read(BASE/"readback.json")
    byink={cell.tobytes():names[str(j)] for j,cell in enumerate(pal) if j>=2 and str(j) in names}
    overlap=Counter()
    replay={}
    with ZipFile(ARCHIVE) as z:
        data=json.loads(z.read("records.json"))
        assert len(data)==len(rows())
        selected=(0,1,96,98,101,len(data)-8)
        for arm in ARMS:
            writer=PicturedTranslator(arm)
            matches=0
            for i,row in enumerate(data):
                o=row["answers"][arm]
                blob=z.read(f"{i:04d}/{arm}.png")
                assert sha256(blob).hexdigest()==o["png"]
                pixels=image_cells(blob,o["cells"])
                assert "".join(byink[x.tobytes()] for x in pixels)==o["text"]
                if i in selected:
                    img=image_cells(z.read(f"{i:04d}/source.png"),len(row["source"]))
                    (actual,s),=writer.translate([img],row["t"])
                    matches+=int(s==o["stop"] and np.array_equal(actual,pixels))
            replay[arm]=dict(matches=matches,sampled=len(selected))
        for i,row in enumerate(data):
            for col,t in (("source",1-row["t"]),("ref",row["t"])):
                ids=ink.encode(row[col])
                padded=np.zeros(96,np.uint16)
                padded[:len(ids)]=ids
                overlap[f'{row["group"]}/{col}']+=int(padded.tobytes() in seen[t])
                if col=="source":
                    assert np.array_equal(ink.array()[ids],image_cells(z.read(f"{i:04d}/source.png"),len(ids)))
    history=[read(MODELS/arm/"fit.json") for arm in ("full","marked")]
    for a,b in zip(history[0]["epochs"],history[1]["epochs"],strict=True):
        for k in ("pairs","positions","updates","schedule"):
            assert a[k]==b[k]
    # Each source contains only physical bitmap addresses in the complete ordered view.
    collisions=Counter()
    for form in ("plain","subject","object"):
        panel=[r for r in data if r["group"]==form]
        for a,b in zip(panel[::2],panel[1::2],strict=True):
            ia,ib = [torch.tensor([ink.encode(r["source"])]) for r in (a,b)]
            for ids in (ia,ib):
                sample=pictured_input(ids,torch.zeros((*ids.shape,3),dtype=torch.uint8))
                assert torch.equal(ids,sample[:,:,0])
            collisions[form]+=int(torch.equal(ia,ib))
        assert len(panel)==472
    dump(PUB/"audit.json",dict(overlap=overlap,reader_weights=history[1]["marker"],cases=len(data),
         answer_pngs=len(data)*len(ARMS),cpu_replay=replay,full_view_collisions=collisions,polarity_pairs_per_form=236,
         proof=sha(ROOT/"formal/span_list/IntactPicturedSource.lean"),archive=sha(ARCHIVE),
         paired_schedules=[h["schedule"] for h in history[0]["epochs"]]))
    print(replay,overlap,collisions,flush=True)


if __name__=="__main__":
    main()
