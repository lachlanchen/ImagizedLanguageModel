"""Recount saved free streams, stratify repetition, and measure visually novel contexts."""
from collections import Counter
import hashlib
import numpy as np
import pyarrow.parquet as pq
from scripts.diagnose_visual_context_gradients import load,sha,require,write_new
from scripts.fit_pictured_idioms import DATA,OUT
from scripts.prepare_pictured_idioms import RAW,TAG


def raw_cases():
    examples={}
    for batch in pq.ParquetFile(RAW/'validation.parquet').iter_batches(batch_size=64):
        for row in batch.to_pylist():
            tags=[int(n) for text in row['content'] for n in TAG.findall(text)]
            words=dict(zip(tags,row['answers']['text'],strict=True))
            for text in row['content']:
                for hit in TAG.finditer(text):
                    tag=int(hit[1])
                    examples[tag]=(TAG.sub('□',text[:hit.start()])[-24:],
                                   TAG.sub('□',text[hit.end():])[:24],words[tag])
    return examples


def main():
    data=load(OUT/'result.json')
    with np.load(DATA/'validation.npz') as f:x,y,tag=f['context'],f['target'],f['tag']
    with np.load(DATA/'train.npz') as f:train_x,train_y=f['context'],f['target']
    px=np.load(DATA/'pixels.npy')
    with np.load(OUT/'generated.npz') as f:a,n,end=f['packed'],f['length'],f['stopped']
    known={b.tobytes():i for i,b in enumerate(px)}
    gt=[tuple(row) for row in y]
    read=[tuple(known.get(tile.tobytes(),-1) for tile in row[:length]) for row,length in zip(a,n)]
    exact=[bool(stopped and output==target) for output,target,stopped in zip(read,gt,end)]
    require(sum(exact)==data['complete_exact'] and len(exact)==len(y),'Archive differs')
    history=set(map(tuple,train_y))
    signatures={tuple(row[16:32]) for row in train_x}
    contexts={tuple(row) for row in train_x}
    require(all(tuple(row) not in contexts for row in x),'Train prompt repeated')
    outcomes=Counter();why={};rng=np.random.default_rng(419)
    source=raw_cases()
    previous=set()
    for i in range(len(y)):
        gold=gt[i]; novelty=tuple(x[i,16:32]) not in signatures
        missing=sum(k not in x[i] for k in gold)
        for k,on in [('all',True),('central_16_new',novelty),('whole_expression_unseen',gold not in history),
                     ('all_four_absent',missing==4),('two_or_more_absent',missing>=2)]:
            if on:
                outcomes[(k,'eligible')]+=1;outcomes[(k,'exact')]+=exact[i]
        if exact[i] and missing and gold not in previous and len(why)<6:
            left,right,word=source[int(tag[i])]
            why[str(i)]=dict(excerpt_before=left,excerpt_after=right,answer=word,missing=missing,
                             central_16_new=novelty,output_sha256=hashlib.sha256(a[i,:n[i]].tobytes()).hexdigest())
            previous.add(gold)
    named=Counter(se for ok,se in zip(exact,gt) if ok)
    report=dict(recounted=len(y),sha256=sha(OUT/'generated.npz'),groups={key:{v:outcomes[key,v] for v in ('eligible','exact')}
        for key in sorted({s[0] for s in outcomes})},distinct_correct_recount=len(named),
        most_frequent_correct_count=max(named.values(),default=0),first_distinct_correct_with_absent=why,
        scope='Whole-answer comparisons on physical bytes. Source strings used only after output generation.')
    write_new(OUT/'recount.json',report)
    print(report['groups'],why)


if __name__=='__main__':
    main()
