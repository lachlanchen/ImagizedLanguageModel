"""Compare source-conditioned probabilities of emitted answers and held reference glyphs."""
import json
from zipfile import ZipFile
import numpy as np
import torch
from scripts.probe_glyph_draft_source import score
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_glyph_draft_editor import ARCHIVE,ARMS,PUB,base
from scripts.write_glyph_draft_editor import PictureEditor


def main():
    torch.set_num_threads(3)
    net=PictureEditor('editor')
    face=base.Ink(net.palette)
    assessment=base.read(PUB/'assessment.json')
    assert base.sha(ARCHIVE)==assessment['archive']
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        inputs=[net.addresses(unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source'])))) for i,r in enumerate(rows)]
        drafts=[net.addresses(unpack(z.read(f'{i:04d}/phrase.png'),r['answers']['phrase']['cells'])) for i,r in enumerate(rows)]
        targets={'reference':[face.encode(r['ref']) for r in rows]}
        for arm in ARMS:
            targets[arm]=[net.addresses(unpack(z.read(f'{i:04d}/{arm}.png'),r['answers'][arm]['cells'])) for i,r in enumerate(rows)]
    scores=[{} for r in rows]
    assert not face.bad
    for t in (0,1):
        order=[i for i,r in enumerate(rows) if r['t']==t]
        for arm,all_targets in targets.items():
            for j in range(0,len(order),12):
                idx=order[j:j+12]
                values=score(net,[inputs[i] for i in idx],[all_targets[i] for i in idx],[drafts[i] for i in idx],t)
                for i,x in zip(idx,values,strict=True):
                    n=len(all_targets[i])+1
                    scores[i][arm]=dict(total=x,cells_with_stop=n,per_cell=x/n)
    out={}
    for scale in ('total','per_cell'):
        delta=np.array([r['reference'][scale]-r['editor'][scale] for r in scores])
        chosen=[max(ARMS,key=lambda arm:r[arm][scale]) for r in scores]
        out[scale]=dict(reference_wins=int((delta>1e-5).sum()),reference_ties=int((np.abs(delta)<=1e-5).sum()),
                   median_delta=float(np.median(delta)),reranked=sum(g[m] for g,m in zip(assessment['grades'],chosen,strict=True)))
    union=[i for i,g in enumerate(assessment['grades']) if any(g[m] for m in ARMS)]
    base.dump(PUB/'path-comparison.json',dict(archive=base.sha(ARCHIVE),scores=scores,summary=out,credited_union=union))
    print(out,union)


if __name__=='__main__':
    main()
