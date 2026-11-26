"""Replay archived inputs and independently score memory interventions along reference prefixes."""
from collections import Counter,defaultdict
from hashlib import sha256
from zipfile import ZipFile
import json
import numpy as np
import torch
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_retrieved_glyphs import ARCHIVE,ARMS,PUB,ROOT,WORK,base,png,records
from scripts.write_retrieved_glyphs import PassageWriter,PictureEditor,pack,pictured_input


@torch.inference_mode()
def scores(w,src,draft,target):
    net=w.neural
    ix=pack([src],'cpu')
    y=pack([target+[1]],'cpu')
    d=pack([draft or [0]],'cpu')
    field=(255*net.marker(net.features,ix).sigmoid()*ix.ne(0).unsqueeze(-1)).round().byte()
    return net.model(net.features,pictured_input(ix,field),y,net.inv[w.t],w.t,d,net.local[w.t])[0].exp().numpy()


def main():
    torch.set_num_threads(3)
    generation=base.read(PUB/'generation.json')
    assert generation['archive']==base.sha(ARCHIVE)
    grade=base.read(PUB/'assessment.json')
    assert grade['archive']==generation['archive']
    official,_=records()
    net=PictureEditor('editor','cpu')
    face=base.Ink(net.palette)
    checks=[]
    totals=defaultdict(Counter)
    details=[]
    strength=base.read(PUB/'calibration.json')['chosen']
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'));trace=json.loads(z.read('supports.json'))
        for i,(row,orig) in enumerate(zip(rows,official,strict=True)):
            assert all(row[k]==orig[k] for k in ('t','source','ref','group','donor'))
            for m in ARMS:
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==row['answers'][m]['png']
        for t in (0,1):
            w=PassageWriter(t,'cpu',net)
            for i,r in enumerate(rows):
                if r['t']!=t:continue
                inp=unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source'])))
                src=net.addresses(inp)
                if i in (0,1,24,25,48,49,72,73,96,97,332,333):
                    for mode in ('related','random'):
                        p,stop=w.translate([inp],mode)[0]
                        digest=sha256(png(p)).hexdigest()
                        checks.append(dict(i=i,mode=mode,hash=digest,same=digest==r['answers'][mode]['png'] and stop==r['answers'][mode]['stop']))
                if i>=96:continue
                ref=face.encode(r['ref'])
                draft=net.addresses(w.phrase.translate(inp)[0])
                law=scores(w,src,draft,ref)
                raw=law.argmax(-1)
                targets=ref+[1]
                for mode in ('related','random'):
                    support=w.support(src,mode)
                    assert support.ids==trace[i][mode]
                    hist=(0,0,0)
                    cnt=Counter();gain=0.
                    for step,g in enumerate(targets):
                        # Recompute the literal convex mixture independently of the free decoder's rescaling.
                        origin=law[step];mix=origin
                        match=support.votes.get(hist)
                        if match:
                            local,counts,n=match
                            gate=strength[t]*n/(2+n)
                            mix=(1-gate)*origin
                            mix[local]+=gate*counts/n
                            cnt['matched']+=1
                        correct=w.lookup[g]
                        changed=int(mix.argmax())
                        cnt['positions']+=1
                        cnt['parent_correct']+=int(raw[step]==correct)
                        cnt['rescued']+=int(changed==correct and raw[step]!=correct)
                        cnt['spoiled']+=int(changed!=correct and raw[step]==correct)
                        cnt['changes']+=int(changed!=raw[step])
                        gain+=float(np.log(max(1e-30,mix[correct]))-np.log(max(1e-30,origin[correct])))
                        hist=(*hist[1:],g)
                    cnt['ref_log_gain']=gain
                    totals[mode].update(cnt)
                    details.append(dict(i=i,mode=mode,counts=cnt))
        counts=Counter()
        for i,r in enumerate(rows[:96]):
            ans=dict(r['answers'],qwen=grade['qwen'][i],reference=dict(stop=True),foil=dict(stop=True))
            flags={m:bool(a['stop'] and all(v[m] for v in grade['votes'][2*i:2*i+2])) for m,a in ans.items()}
            assert flags==grade['grades'][i];counts.update(flags)
        assert all(grade['counts']['all'][m]==n for m,n in counts.items())
        donor=sum(r['answers']['exchanged']['png']==rows[r['donor']]['answers']['related']['png'] for r in rows)
    assert not face.bad
    base.dump(PUB/'audit.json',dict(archive=generation['archive'],cpu=checks,cpu_same=sum(x['same'] for x in checks),
            donor_same=donor,path_totals=totals,details=details,proof=base.sha(ROOT/'formal/span_list/RetrievedGlyphVote.lean')))
    print('replay',sum(x['same'] for x in checks),len(checks),'donor',donor,'true-prefix totals',dict(totals))


if __name__=='__main__':main()
