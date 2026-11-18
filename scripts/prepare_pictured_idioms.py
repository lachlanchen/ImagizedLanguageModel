"""Convert source idiom cloze passages to ordered pixel windows without choices."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import time

import numpy as np
import pyarrow.parquet as pq

from scripts.diagnose_visual_context_gradients import load,require,sha,write_new
from scripts.prepare_visual_sequence_corpus import FONT,ROOT,Rasterizer

RAW=ROOT/'data/external/chid_clue_v1'
OUT=ROOT/'data/pictured_idioms_v1'
BIND={'train':'f28b2eb381d0e522b4b77e938712feca9524dd0e9e5f84bf1e35029002e9767e',
      'validation':'41ac76386a396910ac99803634c4a80ed8bccd4e6c1e4795e0d965c889bfeb16'}
TAG=re.compile(r'#idiom(\d+)#')
WIDTH=24


def main():
    start=time.monotonic()
    OUT.mkdir(exist_ok=False)
    raster=Rasterizer()
    book={bytes(128):0}
    train_hashes=set();details={}
    def tile(c):
        b=raster.tile(c)
        return book.setdefault(b,len(book))
    for split,digest in BIND.items():
        source=RAW/(split+'.parquet')
        require(sha(source)==digest,'Source digest differs')
        xs=[];ys=[];groups=[];tags=[];seen=set();counts=Counter()
        for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                group=row['idx']
                if split=='train' and hashlib.sha256(str(group).encode()).digest()[0]>=32:
                    continue
                counts['selected_groups']+=1
                answers=row['answers']['text']
                ordered=[int(n) for s in row['content'] for n in TAG.findall(s)]
                require(ordered==sorted(set(ordered)) and len(ordered)==len(answers),'Ambiguous gold order')
                gold=dict(zip(ordered,answers))
                for passage in row['content']:
                    for hit in TAG.finditer(passage):
                        key=int(hit[1]);y=gold[key]
                        left=TAG.sub('□',passage[:hit.start()])[-WIDTH:]
                        right=TAG.sub('□',passage[hit.end():])[:WIDTH]
                        if len(y)!=4 or not all('\u3400'<=c<='\u9fff' for c in y):
                            counts['non_four_han']+=1;continue
                        try:
                            x=tuple([0]*(WIDTH-len(left))+[tile(c) for c in left+right]+[0]*(WIDTH-len(right)))
                            target=[tile(c) for c in y]
                        except ValueError:
                            counts['unsupported']+=1;continue
                        if x in seen or (split=='validation' and x in train_hashes):
                            counts['duplicate_or_train_overlap']+=1;continue
                        seen.add(x);xs.append(x);ys.append(target);tags.append(key);groups.append(group)
        if split=='train':train_hashes=seen
        arrays=dict(context=np.asarray(xs,np.uint16),target=np.asarray(ys,np.uint16),group=np.array(groups,np.int32),tag=np.array(tags,np.int32))
        require(len(book)<65536 and len(xs)>0,'Empty or oversized physical inventory')
        path=OUT/(split+'.npz');np.savez(path,**arrays)
        details[split]=dict(examples=len(xs),groups=len(set(groups)),source_sha256=digest,sha256=sha(path),counts=dict(counts),
            distinct_answers=len(set(tuple(y) for y in ys)),at_least_one_absent=sum(any(k not in x for k in y) for x,y in zip(xs,ys)))
    pixels=np.frombuffer(b''.join(book),np.uint8).reshape(-1,128)
    np.save(OUT/'pixels.npy',pixels)
    write_new(OUT/'report.json',dict(splits=details,pixels_sha256=sha(OUT/'pixels.npy'),font_sha256=sha(FONT),
              program_sha256=sha(Path(__file__)),seconds=time.monotonic()-start))
    print(details,flush=True)


if __name__=='__main__':
    main()
