"""Translate the same natural news sentence from real pixels or text with a fixed local teacher."""
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from time import monotonic
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import requests
from scripts.calibrate_translation_graders import (PUB, WORK, ROOT, base, panel, ONE, BINARY,
    MODELS, QWEN, JUDGE, STORE, server, chat, gpu_owner, configure_cuda_cli)
from scripts.evaluate_visual_training_breadth import png

PROMPT = 'Translate the supplied Chinese sentence into natural English. Give only the translation.'
RULE = ROOT/'references/visual_news_teacher_protocol.md'
ARCHIVE = WORK/'news-teacher-images.zip'


def main():
    configure_cuda_cli()
    cases, selection = panel()
    panel_path = PUB/'panel.json'
    assert base.read(panel_path) == json.loads(json.dumps(selection))
    for name in (JUDGE,QWEN):
        assert base.sha(STORE/'blobs'/('sha256-'+MODELS[name])) == MODELS[name]
    face = base.Ink()
    cells = [face.encode(c[0]) for c in cases]
    assert not face.bad
    tiles = face.array()
    images = [png(tiles[s]) for s in cells]
    old = ROOT/'publication/ilm-nature/evidence/wmt_news_visual_answers.zip'
    assert base.sha(old) == base.read(PUB/'writer.json')['archive']
    with ZipFile(old) as z:
        rows = json.loads(z.read('records.json'))
        represented = {r['index'] for r in rows}
        assert all(images[i] == z.read(f'{i}/source.png') for i in represented)
    with ZipFile(ARCHIVE,'w',ZIP_DEFLATED) as z:
        for i,blob in enumerate(images):
            z.writestr(f'{i}.png',blob)
    binding = sha256((base.sha(RULE)+base.sha(panel_path)+base.sha(Path(__file__))+
                      PROMPT+ONE+json.dumps(BINARY)+MODELS[JUDGE]+MODELS[QWEN]).encode()).hexdigest()
    jobs = [(i,k) for i in range(len(cases)) for k in ('pixels','text')]
    cache = WORK/'news-teacher-votes'
    cache.mkdir(exist_ok=True)
    start = monotonic()
    def cached(path, fn):
        if path.exists():
            record=base.read(path)
            assert record['binding']==binding
            return record['value']
        value=fn()
        base.dump(path,dict(binding=binding,value=value))
        return value
    with gpu_owner(), server() as url, ThreadPoolExecutor(4) as workers:
        def translate(job):
            i,k=job
            def ask():
                message=dict(role='user', content=PROMPT)
                if k=='pixels':
                    message['images']=[base64.b64encode(images[i]).decode('ascii')]
                else:
                    message['content'] += '\n'+cases[i][0]
                response=requests.post(url+'/api/chat',json=dict(model=JUDGE,messages=[message],think=False,
                    stream=False,options=dict(temperature=0,seed=713581,num_ctx=4096,num_predict=256)),timeout=180)
                response.raise_for_status()
                result=response.json()
                return dict(text=result['message']['content'].strip(),stop=result.get('done_reason')=='stop')
            return cached(cache/f'{i}-{k}-writing.json',ask)
        answers=dict(zip(jobs,workers.map(translate,jobs),strict=True))
        def assess(job):
            i,k=job
            def ask():
                packet=dict(source=cases[i][0],target=answers[job]['text'])
                result=chat(url,QWEN,ONE+json.dumps(packet,ensure_ascii=False),BINARY)
                assert result['stop']
                vote=json.loads(result['text'])['faithful']
                assert type(vote) is bool
                return vote
            return cached(cache/f'{i}-{k}-grade.json',ask)
        votes=dict(zip(jobs,workers.map(assess,jobs),strict=True))
    data=[dict(i=i,image=sha256(images[i]).hexdigest(),admitted=i in represented,
               **{k:dict(**answers[i,k],vote=votes[i,k],credit=bool(votes[i,k] and answers[i,k]['stop']))
                  for k in ('pixels','text')}) for i in range(len(cases))]
    totals={}
    for name,group in [('all',data),('represented',[r for r in data if r['admitted']]),
                       ('excluded',[r for r in data if not r['admitted']])]:
        doc,idx=np.unique([selection['records'][r['i']]['doc'] for r in group],return_inverse=True)
        differences=[int(r['pixels']['credit'])-int(r['text']['credit']) for r in group]
        sums=np.bincount(idx,weights=differences);sizes=np.bincount(idx)
        draws=np.random.default_rng(813002).integers(len(doc),size=(10000,len(doc)))
        ci=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist()
        totals[name]=dict(count=len(group),pixels=sum(r['pixels']['credit'] for r in group),
                          text=sum(r['text']['credit'] for r in group),difference_interval=ci)
    base.dump(PUB/'image-teacher.json',dict(binding=binding,records=data,totals=totals,seconds=monotonic()-start))
    print(totals,flush=True)


if __name__=='__main__':
    main()
