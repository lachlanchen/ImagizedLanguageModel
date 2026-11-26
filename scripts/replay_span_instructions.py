"""Replay the word-origin and contrast probes, including the source specimens."""
from collections import Counter
from zipfile import ZipFile
import json
import torch
from scripts.probe_span_instructions import ARCHIVE,PUB,OpticalSpanWriter,cells,dump,np,png,prompts,read,sha,write_condition


def main():
    torch.set_num_threads(2)
    assert sha(ARCHIVE)==read(PUB/"word-origin-images.json")["archive"]
    counts=Counter()
    differences=[]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        grids=[]
        for i,(r,(_,forms)) in enumerate(zip(rows,prompts(),strict=True)):
            n=len(r["question"])+len(forms)
            grid=cells(z.read(f"{i:03d}/input.png"),n).copy()
            if forms:
                assert np.array_equal(grid[-3:],np.stack([forms[k] for k in (2,0,1)]))
                assert png(np.stack(forms))==z.read(f"{i:03d}/requested.png")
            grids.append(grid)
        directions=[("en","zh").index(r["to"]) for r in rows]
        def match(i,m,blob,n,stop,edges):
            old=rows[i][m]
            if (blob,n,stop,edges)==(z.read(f"{i:03d}/{m}.png"),old["cells"],old["stop"],old["edges"]):
                counts[m]+=1
            else:
                differences.append([i,m])
                (PUB/f"probe-replay-{i:03d}-{m}.png").write_bytes(blob)
        for i,blob,n,stop in write_condition("guided","cpu",grids,directions,range(len(rows)),12):
            match(i,"single",blob,n,stop,None)
        for t,lang in enumerate(("en","zh")):
            ix=[i for i,d in enumerate(directions) if d==t]
            net=OpticalSpanWriter(lang,"cpu")
            for j in range(0,len(ix),12):
                group=ix[j:j+12]
                for i,(bits,stop,edges) in zip(group,net.write([grids[k] for k in group]),strict=True):
                    match(i,"span",png(bits),len(bits),stop,edges)
    dump(PUB/"word-origin-replay.json",dict(archive=sha(ARCHIVE),same=counts,differences=differences))
    print(counts,differences,flush=True)


if __name__=="__main__":
    main()
