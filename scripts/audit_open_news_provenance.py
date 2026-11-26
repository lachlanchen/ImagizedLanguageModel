"""Trace relaxed textual overlaps between WMT17 external cases and the earlier OPUS corpus."""
from scripts.reserve_open_news_external import WORK,PUB,base,key
from scripts.prepare_bilingual_ink import source,HASH

def main():
    panel=base.read(WORK/'test.json')
    assert base.sha(WORK/'test.json')==base.read(PUB/'selection.json')['panel']
    index=[{key(r[s]):i for i,r in enumerate(panel)} for s in ('target','source')]
    matches=[set(),set()];paired=set();total=0
    for en,zh in source('train'):
        total+=1
        a,b=index[0].get(key(en)),index[1].get(key(zh))
        if a is not None:matches[0].add(a)
        if b is not None:matches[1].add(b)
        if a is not None and a==b:paired.add(a)
    base.dump(PUB/'earlier-overlap.json',dict(panel=base.sha(WORK/'test.json'),official=HASH['train'],rows=total,
              target=sorted(matches[0]),source=sorted(matches[1]),paired=sorted(paired)))
    print(total,len(matches[1]),len(matches[0]),len(paired))

if __name__=='__main__':
    main()
