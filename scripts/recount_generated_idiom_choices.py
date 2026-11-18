"""Recount all candidate pixels, selection indices and source answer alignments."""
import io
from zipfile import ZipFile
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from scripts.diagnose_visual_context_gradients import load,require,sha,write_new
from scripts.evaluate_generated_idioms import OUT,DATA,ALIGNED
from scripts.prepare_pictured_idioms import RAW,TAG
from scripts.prepare_visual_sequence_corpus import Rasterizer


def main():
    expected=load(OUT/'result.json')
    with np.load(DATA/'validation.npz') as f:y,tag=f['target'],f['tag']
    px=np.load(DATA/'pixels.npy');mask=Rasterizer();gold={}
    for batch in pq.ParquetFile(RAW/'validation.parquet').iter_batches(batch_size=64):
        for r in batch.to_pylist():
            tags=[int(n) for passage in r['content'] for n in TAG.findall(passage)]
            for t,word,idx in zip(tags,r['answers']['text'],r['answers']['candidate_id'],strict=True):
                require(r['candidates'][idx]==word and t not in gold,'Inconsistent upstream answer')
                gold[t]=word
    for t,ids in zip(tag,y):
        require(b''.join(mask.tile(c) for c in gold[int(t)])==px[ids].tobytes(),'Incorrect packed reference')
    with np.load(OUT/'generated.npz') as f:chosen,n,s=(f[k] for k in ('packed','length','stopped'))
    with np.load(ALIGNED/'generated.npz') as f:beam,bn,bs=(f[k] for k in ('packed','length','stopped'))
    counts=Counter();offset=0
    path=OUT/'proposals.zip'
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            require(int(name[:-4])==offset,'Discontinuous output archive')
            with np.load(io.BytesIO(archive.read(name))) as f:
                a,length,stop,choice=(f[k] for k in ('pixels','length','stopped','choice'))
            size=len(choice);rows=np.arange(size);region=slice(offset,offset+size)
            require(a.shape==(size,33,8,128) and (choice<33).all(),'Proposal coverage')
            for v,l,e,k,kl,ke in [(a[rows,choice],length[rows,choice],stop[rows,choice],chosen,n,s),
                                  (a[:,0],length[:,0],stop[:,0],beam,bn,bs)]:
                require(np.array_equal(v,k[region]) and np.array_equal(l,kl[region]) and np.array_equal(e,ke[region]),'Physical streams changed')
            exact=(a[:,:,:4]==px[y[region]][:,None]).all((2,3)) & (length==4) & stop
            counts.update(dict(complete_exact=int(exact[rows,choice].sum()),oracle_proposal_exact=int(exact.any(1).sum()),
                               first_random_exact=int(exact[:,1].sum()),beam_exact=int(exact[:,0].sum())))
            offset+=size
    require(offset==len(y) and all(expected[k]==v for k,v in counts.items()),'Recount differs')
    write_new(OUT/'proposal_audit.json',dict(examples=offset,candidate_streams=33*offset,scored_counts=dict(counts),
              source_gold_images_verified=len(y),archive_sha256=sha(path)))
    print(dict(counts))

if __name__=='__main__':
    main()
