"""Measure whether a paired source identifies its true complete target under draft interventions."""
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from scripts.evaluate_glyph_draft_editor import ARCHIVE,PUB,base
from scripts.write_glyph_draft_editor import PictureEditor,pictured_input,pack


@torch.inference_mode()
def score(net,inputs,targets,drafts,t):
    src=pack(inputs,'cpu')
    tgt=pack([x+[1] for x in targets],'cpu')
    ds=pack([x or [0] for x in drafts],'cpu') if drafts is not None else None
    field=(255*net.marker(net.features,src).sigmoid()*src.ne(0).unsqueeze(-1)).round().byte()
    prob=net.model(net.features,pictured_input(src,field),tgt,net.inv[t],t,ds,net.local[t])
    correct=prob.gather(-1,net.local[t][tgt].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (correct*tgt.ne(0)).double().sum(1).tolist()


def main():
    from scripts.audit_glyph_draft_editor import unpack
    torch.set_num_threads(3)
    net=PictureEditor('editor')
    assert base.read(PUB/'generation.json')['archive']==base.sha(ARCHIVE)
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        face=base.Ink(net.palette)
        src=[net.addresses(unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source'])))) for i,r in enumerate(rows)]
        ref=[face.encode(r['ref']) for r in rows]
        draft=[net.addresses(unpack(z.read(f'{i:04d}/phrase.png'),r['answers']['phrase']['cells'])) for i,r in enumerate(rows)]
    assert not face.bad
    groups={'book':[(j+t,j+2+t) for j in range(0,96,4) for t in (0,1)],
            'role':[(j+k,j+2+k) for j in range(96,len(rows),4) for k in (0,1)]}
    detail={}
    summary={}
    for key,pairs in groups.items():
        detail[key]={}
        summary[key]={}
        for mode in ('both','masked','fixed_draft','fixed_source'):
            out=[]
            for t in (0,1):
                chosen=[(a,b) for a,b in pairs if rows[a]['t']==t]
                for j in range(0,len(chosen),8):
                    source,targets,drafts=[],[],[]
                    chunk=chosen[j:j+8]
                    for a,b in chunk:
                        assert rows[a]['t']==rows[b]['t'] and rows[a]['doc']==rows[b]['doc']
                        donor=rows[a]['donor']
                        for x,y in ((a,a),(a,b),(b,a),(b,b)):
                            source.append(src[donor if mode=='fixed_source' else x])
                            drafts.append(draft[donor if mode=='fixed_draft' else x])
                            targets.append(ref[y])
                    values=score(net,source,targets,None if mode=='masked' else drafts,t)
                    for k,(a,b) in enumerate(chunk):
                        four=values[4*k:4*k+4]
                        delta=four[0]+four[3]-four[1]-four[2]
                        out.append(dict(a=a,b=b,scores=four,delta=delta))
            detail[key][mode]=out
            delta=np.array([r['delta'] for r in out])
            summary[key][mode]=dict(n=len(out),positive=int((delta>1e-5).sum()),negative=int((delta< -1e-5).sum()),
                     ties=int((np.abs(delta)<=1e-5).sum()),median=float(np.median(delta)))
    base.dump(PUB/'source-probe.json',dict(archive=base.sha(ARCHIVE),implementation=base.sha(Path(__file__)),
              summary=summary,pairs=detail))
    print(summary,flush=True)


if __name__=='__main__':
    main()
