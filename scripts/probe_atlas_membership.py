"""Measure exact image addressing with and without the requested dictionary entry."""
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.glyph_atlas import mentions
from scripts.atlas_questions import render_question
from scripts.read_glyph_atlas import get_book
from scripts.prepare_glyph_atlas import WORK,PUB,MODEL,read,dump,Rasterizer
from scripts.train_glyph_atlas import load_model,gpu_owner,snapshot,consumed

FRAMES=(("在古文字辭典裡，{h}的{f}怎麼寫？","我正在核對{h}的{f}，之前看的是{d}。"),
        ("I'm reading an inscription. Can I see the {f} for {h}?","My last note was on {d}. This entry is about {h}. Give its {f}."))


@torch.inference_mode()
def main():
    book,meta=get_book(WORK)
    tiles=book["tiles"][book["heads"]]
    lookup={tile.tobytes():i for i,tile in enumerate(tiles)}
    pool=[i for i,e in enumerate(meta) if e["split"]=="development"]
    queries=[]
    for j,a in enumerate(pool):
        b=pool[(j+1)%len(pool)]
        for lang in (0,1):
            for foil,frame in enumerate(FRAMES[lang]):
                for f in (4,1):
                    prompt,*_=render_question(frame,meta[a]["char"],meta[b]["char"],[f],lang,0)
                    queries.append((a,lang,foil,f,prompt))
    result=[]
    with gpu_owner():
        energy=snapshot()
        net=load_model(read(MODEL/"release.json")["reader"],"cuda")
        tile_cache={}
        raster=Rasterizer()
        for start in range(0,len(queries),96):
            group=queries[start:start+96]
            width=max(len(q[-1]) for q in group)
            ids=np.full((len(group),width),-1,np.int32)
            bits=np.zeros((len(group),width,128),np.uint8)
            for i,(a,lang,foil,f,s) in enumerate(group):
                for k,c in enumerate(s):
                    if c not in tile_cache:
                        tile=raster.tile(c)
                        tile_cache[c]=(lookup.get(tile,-1),np.frombuffer(tile,np.uint8))
                    ids[i,k],bits[i,k]=tile_cache[c]
            obs=observations({"tiles":bits.reshape(-1,128)},"ink").reshape(len(group),width,256)
            valid=np.arange(width)<np.array([len(q[-1]) for q in group])[:,None]
            scores,tags=net(torch.tensor(obs,device="cuda"),torch.tensor(valid,device="cuda"))
            scores=scores.cpu().numpy()
            tags=tags.argmax(-1).cpu().numpy()
            for i,(a,lang,foil,f,s) in enumerate(group):
                q=ids[i]; h=int(scores[i].argmax()); plan=mentions(tags[i,:len(s)])
                def best(available):
                    return int(q[np.where(available,scores[i],-np.inf).argmax()]) if available.any() else -1
                first=int(q[h])
                constrained=best(q>=0)
                absent=-1 if first==a else first
                missing=best((q>=0)&(q!=a))
                result.append((lang,foil,f,a,first,constrained,absent,missing,int(plan==(f,))))
        energy=consumed(energy,snapshot())
    arr=np.array(result)
    def summary(selected):
        x=arr[selected]
        return dict(n=len(x),unconstrained=int(((x[:,4]==x[:,3])&(x[:,8]>0)).sum()),
                    constrained=int(((x[:,5]==x[:,3])&(x[:,8]>0)).sum()),
                    false_original=int((x[:,6]>=0).sum()),false_constrained=int((x[:,7]>=0).sum()))
    np.savez_compressed(PUB/"membership.npz",decisions=arr)
    dump(PUB/"membership.json",dict(all=summary(np.ones(len(arr),bool)),
         by_group={f"{l}-{f}":summary((arr[:,0]==l)&(arr[:,1]==f)) for l in (0,1) for f in (0,1)},energy=energy))
    print(summary(np.ones(len(arr),bool)))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()
