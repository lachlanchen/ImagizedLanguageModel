"""Measure which answers change when a frozen writer sees an unfamiliar source bitmap."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import numpy as np
import torch
from scripts.calibrate_translation_graders import (PUB, WORK, ROOT, base,panel,ONE,BINARY,QWEN,MODELS,STORE,
                                                  gpu_owner,configure_cuda_cli,server,chat)
from scripts.measure_paired_visual_writer import png,unpack,restore
from ilm.visual_lm.open_source_cells import OpenSourceCells

ARCHIVE=ROOT/'publication/ilm-nature/evidence/open_news_ink.zip'
PREVIOUS=ROOT/'publication/ilm-nature/evidence/wmt_news_visual_answers.zip'
RULE=ROOT/'references/open_news_ink_protocol.md'


def generate():
    torch.set_num_threads(2)
    cases,_=panel()
    writer=restore('broad','cpu')
    reader=OpenSourceCells(writer,0)
    names=base.read(base.BASE/'readback.json')
    face=base.Ink()
    codes=[face.encode(s) for s,_,_ in cases]
    assert not face.bad
    tiles=face.array()
    assert base.sha(PREVIOUS)==base.read(PUB/'writer.json')['archive']
    replay=0
    records=[]
    temp=ARCHIVE.with_suffix('.partial')
    with ZipFile(PREVIOUS) as old, ZipFile(temp,'w',ZIP_DEFLATED) as z:
        old_rows={r['index']:r for r in json.loads(old.read('records.json'))}
        for i,cells in enumerate(codes):
            if len(cells)>96:continue
            image=tiles[cells]
            if i in old_rows:
                assert png(image)==old.read(f'{i}/source.png')
                ink,stop=reader.translate(image)
                assert png(ink)==old.read(f'{i}/answer.png') and stop==old_rows[i]['stop']
                replay+=1
                continue
            blob=png(image)
            z.writestr(f'{i}/source.png',blob)
            row=dict(index=i,source=sha256(blob).hexdigest(),cells=len(cells))
            for tag in ('ink','hidden'):
                output,stop=reader.translate(image,hide_unfamiliar=tag=='hidden')
                blob=png(output)
                z.writestr(f'{i}/{tag}.png',blob)
                row[tag]=dict(sha256=sha256(blob).hexdigest(),text=''.join(names[str(k)] for k in writer.addresses(output)),
                              cells=len(output),stop=bool(stop))
            records.append(row)
        z.writestr('records.json',json.dumps(records,ensure_ascii=False))
    new_replay=0
    with ZipFile(temp) as z:
        for r in records[:8]:
            i=r['index'];image=unpack(z.read(f'{i}/source.png'),r['cells'])
            for tag in ('ink','hidden'):
                ink,stop=reader.translate(image,hide_unfamiliar=tag=='hidden')
                assert png(ink)==z.read(f'{i}/{tag}.png') and stop==r[tag]['stop']
                new_replay+=1
    temp.replace(ARCHIVE)
    receipt=dict(archive=base.sha(ARCHIVE),previous=base.sha(PREVIOUS),protocol=base.sha(RULE),
                 panel=base.sha(PUB/'panel.json'),new=len(records),original_replay=replay,new_replay=new_replay,
                 changed=sum(r['ink']['sha256']!=r['hidden']['sha256'] for r in records),
                 stops={k:sum(r[k]['stop'] for r in records) for k in ('ink','hidden')})
    base.dump(PUB/'open-ink.json',receipt)
    print(receipt,flush=True)


def assess():
    configure_cuda_cli()
    cases,sel=panel()
    assert base.sha(ARCHIVE)==base.read(PUB/'open-ink.json')['archive']
    assert base.sha(STORE/'blobs'/('sha256-'+MODELS[QWEN]))==MODELS[QWEN]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        palette=np.load(base.BASE/'palette.npy')
        names=base.read(base.BASE/'readback.json')
        letters={c.tobytes():names[str(i)] for i,c in enumerate(palette) if str(i) in names}
        for r in rows:
            for k in ('ink','hidden'):
                data=z.read(f"{r['index']}/{k}.png")
                assert sha256(data).hexdigest()==r[k]['sha256']
                assert ''.join(letters[c.tobytes()] for c in unpack(data,r[k]['cells']))==r[k]['text']
    keys=sorted({(r['index'],r[k]['text']) for r in rows for k in ('ink','hidden')})
    with gpu_owner(),server() as url,ThreadPoolExecutor(4) as workers:
        def vote(pair):
            i,text=pair
            reply=chat(url,QWEN,ONE+json.dumps(dict(source=cases[i][0],target=text),ensure_ascii=False),BINARY)
            assert reply['stop']
            answer=json.loads(reply['text'])['faithful']
            assert type(answer) is bool
            return answer
        votes=dict(zip(keys,workers.map(vote,keys),strict=True))
    doc,idx=np.unique([sel['records'][r['index']]['doc'] for r in rows],return_inverse=True)
    for r in rows:
        for k in ('ink','hidden'):
            r[k]['vote']=votes[r['index'],r[k]['text']]
            r[k]['credit']=r[k]['stop'] and r[k]['vote']
    diff=[int(r['ink']['credit'])-int(r['hidden']['credit']) for r in rows]
    draws=np.random.default_rng(57711).integers(len(doc),size=(10000,len(doc)))
    ci=np.quantile(np.bincount(idx,weights=diff)[draws].sum(1)/np.bincount(idx)[draws].sum(1), [.025,.975]).tolist()
    result=dict(archive=base.sha(ARCHIVE),model=MODELS[QWEN],records=rows,unique_gradings=len(keys),interval=ci,
                credits={k:sum(r[k]['credit'] for r in rows) for k in ('ink','hidden')})
    base.dump(PUB/'open-meaning.json',result)
    print(result['credits'],ci,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('generate','assess'))
    args=cli.parse_args()
    (generate if args.stage=='generate' else assess)()
