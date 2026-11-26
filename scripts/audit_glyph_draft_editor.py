"""Replay original model answers on CPU and quantify changes caused by draft exchange."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile
import json
import numpy as np
from PIL import Image
import torch
from scripts.evaluate_glyph_draft_editor import ARCHIVE,ARMS,png,base,PUB
from scripts.write_glyph_draft_editor import PictureEditor
from scripts.write_covered_phrases import CoveredWriter


def unpack(data,count):
    a=np.asarray(Image.open(BytesIO(data)).convert('L'))
    tiles=(a==0).reshape(a.shape[0]//32,32,a.shape[1]//32,32).transpose(0,2,1,3)
    return np.packbits(tiles.reshape(-1,1024)[:count],axis=1)


def main():
    torch.set_num_threads(3)
    assert base.sha(ARCHIVE)==base.read(PUB/'generation.json')['archive']
    selection=base.read(PUB/'selection.json')
    strips=base.read(base.TABLE)['strip']
    measured=Counter()
    details=[]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        face=base.Ink(np.load(base.BASE/'palette.npy'))
        for arm in ('editor','matched'):
            net=PictureEditor(arm)
            for i in (0,1,24,25,48,49,72,73,96,97,332,333):
                r=rows[i]
                source=unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source'])))
                p,stop=net.translate([source],r['t'])[0]
                matched=sha256(png(p)).hexdigest()==r['answers'][arm]['png'] and stop==r['answers'][arm]['stop']
                details.append(dict(i=i,arm=arm,same=matched,actual=sha256(png(p)).hexdigest(),stop=stop))
        from ilm.visual_lm.optical_lexicon import StripInventory
        names=base.read(base.BASE/'readback.json')
        for t in (0,1):
            writer=CoveredWriter(t)
            target=StripInventory(strips[t],base.read(base.BASE/'inventories.json')[t])
            for i,r in enumerate(rows[:96]):
                if r['t']!=t:
                    continue
                s=unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source'])))
                d=unpack(z.read(f'{i:04d}/phrase.png'),r['answers']['phrase']['cells'])
                assert np.array_equal(writer.translate(s)[0],d)
                ref=face.encode(r['ref'])
                gold=set(target.source_strips(ref)[1:])
                # Count distinct content per request, skipping whitespace and punctuation-only strips.
                content=lambda k:any(c.isalnum() for cell in strips[t][k] for c in names[str(cell)])
                gold={k for k in gold if content(k)}
                measured['reference_content']+=len(gold)
                for arm in ARMS:
                    a=r['answers'][arm]
                    p=unpack(z.read(f'{i:04d}/{arm}.png'),a['cells'])
                    ids=[writer.by_image[x.tobytes()] for x in p]
                    pred=target.source_strips(ids)[1:]
                    measured[arm+'_covered']+=len(set(pred)&gold)
                    grams=list(zip(pred,pred[1:],pred[2:]))
                    measured[arm+'_repeated']+=int(len(set(grams))!=len(grams))
                measured['donor_draft_different']+=r['answers']['phrase']['png']!=rows[r['donor']]['answers']['phrase']['png']
    base.dump(PUB/'audit.json',dict(archive=base.sha(ARCHIVE),selection=selection['hashes']['panel.json'],
              cpu_replay=details,same=sum(x['same'] for x in details),book_drafts_replayed=96,strip_diagnosis=measured))
    print(measured, sum(x['same'] for x in details),len(details),flush=True)


if __name__=='__main__':
    main()
