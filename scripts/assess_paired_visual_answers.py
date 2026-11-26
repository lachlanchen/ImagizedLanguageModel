"""Read both autonomous image answers back and measure whether their meanings change with the source."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from time import monotonic
import json
from zipfile import ZipFile
import numpy as np
from scripts.qualify_paired_translation_teacher import (
    ROOT,PUB,WORK,base,population,MODELS,STORE,QWEN,server,chat,PROMPT,SCHEMA,gpu_owner,configure_cuda_cli)
from scripts.measure_paired_visual_writer import ARCHIVE,unpack


def main():
    if (PUB/'free-meaning.json').exists():
        raise SystemExit('Assessment already finished')
    _, families=population()
    teacher=base.read(PUB/'teacher.json')
    assert base.sha(ARCHIVE)==base.read(PUB/'writer.json')['archive']
    assert base.sha(STORE/'blobs'/('sha256-'+MODELS[QWEN]))==MODELS[QWEN]
    palette=np.load(base.BASE/'palette.npy')
    readback=base.read(base.BASE/'readback.json')
    letters={cell.tobytes():readback[str(i)] for i,cell in enumerate(palette) if str(i) in readback}
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read('records.json'))
        answers={}
        for r in records:
            i,t,s=r['i'],r['t'],r['side']
            data=z.read(f'{i}/{t}/{s}/answer.png')
            assert sha256(data).hexdigest()==r['answer_sha256']
            ink=unpack(data,r['cells'])
            text=''.join(letters[c.tobytes()] for c in ink)
            assert text==r['text'] and r['source']==families[i][s][1-t]
            answers[i,t,s]=(text,r['stop'])
    ids=teacher['audit']
    loc=WORK/'free-votes'
    loc.mkdir(exist_ok=True)
    binding=sha256((base.sha(ARCHIVE)+base.sha(PUB/'teacher.json')+PROMPT+json.dumps(SCHEMA,sort_keys=True)+MODELS[QWEN]).encode()).hexdigest()
    start=monotonic()
    with gpu_owner(),server() as url,ThreadPoolExecutor(4) as workers:
        def vote(job):
            i,t,s,rev=job
            path=loc/('-'.join(map(str,job))+'.json')
            if path.exists():
                old=base.read(path)
                assert old['binding']==binding
                return old['scores']
            a,b=(1,0) if rev else (0,1)
            packet=dict(source=families[i][s][1-t],target=('English','Mandarin')[t],A=answers[i,t,a][0],B=answers[i,t,b][0])
            response=chat(url,QWEN,PROMPT+json.dumps(packet,ensure_ascii=False),SCHEMA)
            flags=json.loads(response['text'])
            assert response['stop'] and all(type(flags[k]) is bool for k in 'AB')
            scores=[flags[k] for k in ('BA' if rev else 'AB')]
            base.dump(path,dict(binding=binding,scores=scores))
            if monotonic()-start>7200:raise RuntimeError('Time ceiling exceeded')
            return scores
        jobs=[(i,t,s,rev) for i in ids for t in (0,1) for s in (0,1) for rev in (0,1)]
        votes=np.asarray(list(workers.map(vote,jobs))).reshape(len(ids),2,2,2,2)
    scores=votes.all(axis=3)
    own=scores.diagonal(axis1=2,axis2=3).copy()
    for j,i in enumerate(ids):
        for t in (0,1):
            for s in (0,1):
                own[j,t,s] &= answers[i,t,s][1]
    exclusive=(votes==np.eye(2,dtype=bool)[None,None,:,None,:]).all(axis=(3,4)) & own
    natural=set(base.read(PUB/'provenance.json')['natural']['audit'])
    groups={'all':ids,'corpus':[i for i in ids if i in natural],'constructed_included':[i for i in ids if i not in natural]}
    totals={}
    for k,indices in groups.items():
        j=[ids.index(i) for i in indices]
        totals[k]=dict(families=len(j),own=int(own[j].sum()),exclusive=int(exclusive[j].sum()),
                       both_rows=int(exclusive[j].all(axis=2).sum()),all_four=int(exclusive[j].all(axis=(1,2)).sum()))
    base.dump(PUB/'free-meaning.json',dict(binding=binding,ids=ids,votes=votes.tolist(),totals=totals,
              protocol=base.sha(ROOT/'references/paired_translation_free_assay.md'),seconds=monotonic()-start))
    print(totals,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    main()
