"""Measure unaided last-word generation with two clipped visual context lengths."""
import json
import re
import unicodedata
from urllib.request import urlretrieve

import numpy as np
import torch
from scripts.evaluate_book_order_prior import PUB,ROOT,ARMS,base,restore,gpu_owner,configure_cuda_cli
from scripts.train_book_order_prior import neutral

URL='https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl'
DIGEST='4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226'
WORD=re.compile("[A-Za-z]+(?:['-][A-Za-z]+)*")


def load():
    path=ROOT/'data/raw/pg19-optical/lambada-test.jsonl'
    if not path.exists():urlretrieve(URL,path)
    assert base.sha(path)==DIGEST
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    glyph={}
    repertoire=set(base.read(base.BASE/'inventories.json')[0])
    panel=[]
    excluded={}
    for i,line in enumerate(path.read_text().splitlines()):
        text=unicodedata.normalize('NFKC',json.loads(line)['text']).translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','—':'-','–':'-','…':'...'}))
        s,word=' '.join(text.split()).rsplit(' ',1)
        s+=' '
        for c in set(s[-96:]+word)-glyph.keys():
            glyph[c]=face.encode(c)[0]
        why=None
        if len(s)<96:why='short prefix'
        elif len(word)>23 or not WORD.fullmatch(word):why='word shape or length'
        elif any(c in face.bad or glyph[c] not in repertoire for c in s[-96:]+word):why='unrepresented ink'
        if why:
            excluded[why]=excluded.get(why,0)+1
            continue
        panel.append((i,[glyph[c] for c in s[-96:]],word,[bool(re.search(r"(?<![A-Za-z'-])"+re.escape(word)+r"(?![A-Za-z'-])",sub,re.I)) for sub in (s[-32:],s[-96:],s)],len(s)))
    assert len(panel)+sum(excluded.values())==5153
    return panel,excluded


@torch.inference_mode()
def emit(w,prefix):
    net=w.model
    bank=net.optical(w.features)
    x=torch.cat((net.begin[0].expand(len(prefix),1,-1),bank[prefix]),1)
    ctx=neutral(net,x)
    h,past=net.decode(x,ctx,None)
    proto=net.prototypes(bank,w.inv[0])
    result=[]
    done=torch.zeros(len(x),device=w.device,dtype=torch.bool)
    for j in range(24):
        idx=w.inv[0][net.distribution(h[:,-1:],proto,None)[:,0].argmax(1)]
        done|=idx.eq(1)
        result.append(idx.masked_fill(done,0))
        if done.all():break
        h,past=net.decode(bank[idx][:,None],ctx,None,past,len(prefix[0])+1+j)
    return torch.stack(result,1).cpu().numpy(),done.cpu().numpy()


def main():
    panel,excluded=load()
    names=base.read(base.BASE/'readback.json')
    scores={}
    arrays={}
    with gpu_owner():
        for arm in ARMS:
            w=restore(arm,'prior','cuda')
            for width in (32,96):
                raw=np.zeros((len(panel),24,128),dtype=np.uint8)
                counts=np.zeros(len(panel),dtype=np.uint8)
                stops=np.zeros(len(panel),dtype=bool)
                hits=np.zeros(len(panel),dtype=bool)
                for j in range(0,len(panel),128):
                    batch=panel[j:j+128]
                    prefixes=torch.tensor([r[1][-width:] for r in batch],device='cuda')
                    ids,ended=emit(w,prefixes)
                    for k,(seq,stop,r) in enumerate(zip(ids,ended,batch,strict=True),j):
                        a=seq[seq>1]
                        s=''.join(names[str(int(v))] for v in a).replace('　',' ')
                        match=WORD.match(s)
                        hits[k]=bool(match and match[0]==r[2] and (match.end()<len(s) or stop))
                        raw[k,:len(a)]=w.palette[a];counts[k]=len(a);stops[k]=stop
                key=f'{arm}_{width}'
                arrays[key]=raw;arrays[key+'_count']=counts;arrays[key+'_stopped']=stops;arrays[key+'_hit']=hits
                scores[key]=int(hits.sum())
                print(key,scores[key],len(panel),flush=True)
    file=ROOT/'publication/ilm-nature/evidence/book_order_discourse.npz'
    np.savez_compressed(file,ids=np.array([r[0] for r in panel]),**arrays)
    base.dump(PUB/'discourse.json',dict(source=URL,source_hash=DIGEST,arrays=base.sha(file),eligible=len(panel),
        excluded=excluded,prefix_median=float(np.median([r[4] for r in panel])),
        mention_counts=np.array([r[3] for r in panel]).sum(0).tolist(),exact=scores,
        limits=[32,96],max_generated=24,rule='Strict case-sensitive first complete word in greedy ink; original full context is clipped for both conditions.'))


if __name__=='__main__':
    configure_cuda_cli()
    if not (PUB/'discourse.json').exists():
        main()
