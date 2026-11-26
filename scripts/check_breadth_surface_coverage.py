"""Find shortest legal pictured output paths spelling the pretrained native translations."""
from collections import Counter,defaultdict
import gzip,json
from zipfile import ZipFile
import numpy as np
from scripts.compare_training_breadth_transfer import PUB,RAW,COUNTS,base,unpack
from scripts.evaluate_visual_training_breadth import ARCHIVE


def phase_tables(entries):
    tables=[defaultdict(set),defaultdict(set)]
    for phase,_,sequence,_ in entries:
        assert phase in (0,1)
        glyphs=tuple(sequence)
        tables[phase][glyphs[:1]].add(glyphs)
    return tables


def minimum_actions(cells,tables):
    n=len(cells)
    if n>128:return None
    if not n:return 0
    sequence=tuple(cells)
    shortest=[129]*(n+1)
    # A silent initial emission can enter the continuation phase.
    if () in tables[0].get((),set()):
        shortest[0]=1
    for span in tables[0].get(sequence[:1],()):
        if span==sequence[:len(span)]:
            shortest[len(span)]=1
    for j in range(n):
        if shortest[j]>=128:continue
        for span in tables[1].get(sequence[j:j+1],()):
            if span==sequence[j:j+len(span)]:
                k=j+len(span)
                shortest[k]=min(shortest[k],shortest[j]+1)
    return shortest[n] if shortest[n]<=128 else None


def main():
    report=base.read(PUB/'transfer.json')
    assert report['archive']==base.sha(RAW)
    palette=np.load(base.BASE/'palette.npy')
    face=base.Ink(palette)
    reverse={v.tobytes():i for i,v in enumerate(palette) if i>=2}
    with ZipFile(ARCHIVE) as z: held=json.loads(z.read('records.json'))[:96]
    results=[]
    totals=defaultdict(Counter)
    with ZipFile(RAW) as z:
        rows=json.loads(z.read('records.json'))
        for t,lang in enumerate(('en','zh')):
            cfg=base.read(COUNTS/(lang+'-fit.json'))
            weights=COUNTS/(lang+'-span-counts.json.gz')
            assert base.sha(weights)==cfg['weights']
            entries=json.loads(gzip.decompress(weights.read_bytes()))
            tables=phase_tables(entries)
            keys=defaultdict(set)
            pooled=defaultdict(set)
            for phase,address,span,count in entries:
                if count>0:
                    keys[(phase,tuple(span))].add(address)
                    pooled[tuple(span)].add(address)
            inventory=set(base.read(base.BASE/'inventories.json')[t])
            for i,r in enumerate(held):
                if r['t']!=t:continue
                observed=rows[i]['visual']
                actual=unpack(z.read(f'{i:04d}/pretrained.png'),observed['cells'])
                glyphs=[reverse[cell.tobytes()] for cell in actual]
                a=minimum_actions(glyphs,tables)
                actions=ambiguous=mixed=0
                if observed['stop']:
                    start=0
                    for j,end in enumerate(observed['spans']):
                        span=tuple(glyphs[start:end])
                        addresses=keys[(int(j>0),span)]
                        assert addresses
                        actions+=1
                        ambiguous+=int(len(addresses)>1)
                        mixed+=int(len(pooled[span])>1)
                        start=end
                    assert start==len(glyphs)
                assert not observed['stop'] or a is not None
                text=rows[i]['native']['text']
                cells=face.encode(text)
                supported=set(cells)<=inventory and not face.bad.intersection(text)
                b=minimum_actions(cells,tables) if supported else None
                grade=report['grades'][i]
                item=dict(i=i,physical_actions=a,native_actions=b,glyphs=len(cells),within_inventory=bool(supported),
                          native_credit=grade['native'],visual_credit=grade['visual'],actions=actions,ambiguous=ambiguous,pooled=mixed)
                results.append(item)
                for group in (lang,'all'):
                    totals[group].update(dict(actions=actions,ambiguous_actions=ambiguous,pooled_actions=mixed,streams_with_pooled=mixed>0,
                                             n=1,reachable=b is not None,within_inventory=bool(supported),length_over=len(cells)>128,
                                             native_only=grade['native'] and not grade['visual'],
                                             reachable_native_only=b is not None and grade['native'] and not grade['visual']))
    base.dump(PUB/'surface-coverage.json',dict(transfer=base.sha(PUB/'transfer.json'),cases=results,totals=totals))
    print(dict(totals),flush=True)


if __name__=='__main__':
    main()
