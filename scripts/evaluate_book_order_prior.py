"""Measure pictured suffixes and archive free visual writing on books and Chinese news."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from ilm.visual_lm.open_source_cells import OpenSourceCells
from scripts.measure_paired_visual_writer import png, unpack
from scripts.train_book_order_prior import (
    PUB, WORK, ROOT, MODEL, base, restore, next_logits, gpu_owner, configure_cuda_cli,
)

EVIDENCE = ROOT / 'publication/ilm-nature/evidence'
BOOK = EVIDENCE / 'book_order_continuations.zip'
ANSWERS = EVIDENCE / 'book_order_translations.zip'
ARMS = ('baseline', 'intact', 'scrambled')


def emit(archive, name, cells):
    blob = png(cells)
    assert np.array_equal(unpack(blob, len(cells)), cells)
    archive.writestr(name + '.png', blob)
    return dict(cells=len(cells), sha256=sha256(blob).hexdigest())


def readback(writer, cells, names):
    return ''.join(names[str(i)] for i in writer.addresses(cells))


@torch.inference_mode()
def continue_stream(writer, prefix, limit=32):
    bank = writer.model.optical(writer.features)
    inventory = writer.inv[0]
    history = prefix
    stopped = torch.zeros(len(prefix), dtype=torch.bool, device=writer.device)
    result = []
    for _ in range(limit):
        scores = next_logits(writer.model, bank, inventory, history)
        choice = inventory[scores[:, -1].argmax(-1)]
        stopped |= choice.eq(1)
        result.append(choice.masked_fill(stopped, 0))
        if stopped.all():
            break
        history = torch.cat((history, choice[:, None]), dim=1)
    return torch.stack(result, dim=1).cpu().numpy(), stopped.cpu().tolist()


@torch.inference_mode()
def prose():
    data = base.read(PUB / 'data.json')
    assert base.sha(WORK / 'validation.npz') == data['files']['validation.npz']
    with np.load(WORK / 'validation.npz') as held:
        y = torch.as_tensor(held['intact'].astype('int64'), device='cuda')
        groups = held['book'].copy()
    sample = np.concatenate([np.flatnonzero(groups == i)[:4] for i in np.unique(groups)])
    rows = [dict(window=int(i), book=int(groups[i]), answers={}) for i in sample]
    totals = {}
    scores = {}
    names = base.read(base.BASE / 'readback.json')
    tmp = BOOK.with_suffix('.partial')
    with ZipFile(tmp, 'w', ZIP_DEFLATED) as archive:
        for arm in ARMS:
            for stage in ('prior', 'adapted'):
                w = restore(arm, stage, 'cuda')
                bank = w.model.optical(w.features)
                suffix = []
                for chunk in y.split(64):
                    law = next_logits(w.model, bank, w.inv[0], chunk[:, :-1])[:, 32:]
                    truth = chunk[:, 32:]
                    valid = truth.gt(1)
                    local = w.local[0][truth].clamp_min(0)
                    bits = -law.gather(-1, local[..., None]).squeeze(-1) / np.log(2)
                    correct = law.argmax(-1).eq(local) & valid
                    suffix.append(torch.stack(((bits*valid).double().sum(1), correct.sum(1), valid.sum(1)),1).cpu().numpy())
                key = arm + '_' + stage
                totals[key] = np.stack([np.concatenate(suffix)[groups == i].sum(0) for i in np.unique(groups)])
                continuation, stopped = continue_stream(w, y[sample, :32])
                for j, (row, seq, stop) in enumerate(zip(rows, continuation, stopped, strict=True)):
                    i = sample[j]
                    if key == 'baseline_prior':
                        for label, addresses in [('prefix', y[i,:32]), ('reference', y[i,32:64])]:
                            cell = w.palette[addresses[addresses.gt(1)].cpu().tolist()]
                            row[label] = emit(archive, f'{i}/{label}', cell)
                            row[label]['text'] = readback(w, cell, names)
                    cell = w.palette[seq[seq > 1]]
                    item = emit(archive, f'{i}/{key}', cell)
                    row['answers'][key] = dict(**item, stop=stop, text=readback(w,cell,names))
                bits, hits, count = totals[key].sum(0)
                scores[key] = dict(bits=float(bits/count), accuracy=float(hits/count), count=int(count),
                                   stops=sum(stopped), book_totals=totals[key].tolist())
        archive.writestr('records.json',json.dumps(rows,ensure_ascii=False))
    tmp.replace(BOOK)
    draws = np.random.default_rng(762035).integers(16,size=(10000,16))
    ci = {}
    for stage in ('prior','adapted'):
        a,b = [totals[arm+'_'+stage][draws].sum(1) for arm in ('scrambled','intact')]
        ci[stage] = np.quantile((a[:,0]-b[:,0])/a[:,2],[.025,.975]).tolist()
    base.dump(PUB/'prose.json',dict(archive=base.sha(BOOK), scores=scores, saved_windows=len(rows),
                                    paired_bits_saved_intervals=ci))
    print({k:round(v['bits'],4) for k,v in scores.items()},ci,flush=True)


def requests():
    news = [dict(group='news', direction=0, **r) for r in base.read(WORK/'panel.json')]
    edits = [dict(group='edit', direction=0, **r) for r in base.read(WORK/'edits.json')]
    with ZipFile(EVIDENCE/'visual_training_breadth.zip') as z:
        old = json.loads(z.read('records.json'))[96:]
    assert len(old)==472 and all(r['t']==1 and r['doc']=='roles' for r in old)
    controls = [dict(group='roles',direction=1,source=r['source'],target=r['ref']) for r in old]
    reverse = [dict(group='reversed_roles',direction=0,source=r['ref'],target=r['source']) for r in old]
    return news+edits+controls+reverse


@torch.inference_mode()
def translations():
    w = restore('baseline','adapted','cpu')
    face = base.Ink(w.palette)
    names = base.read(base.BASE/'readback.json')
    rows = requests()
    inputs = [(face.encode(r['source']),face.encode(r['target'])) for r in rows]
    assert not face.bad
    palette = face.array()
    tmp = ANSWERS.with_suffix('.partial')
    with ZipFile(tmp,'w',ZIP_DEFLATED) as archive:
        for i,(r,(s,t)) in enumerate(zip(rows,inputs,strict=True)):
            r['input'] = emit(archive,f'{i}/source',palette[s])
            r['reference'] = emit(archive,f'{i}/reference',palette[t])
            r['answers'] = {}
        for arm in ARMS:
            w = restore(arm,'adapted','cuda')
            for direction in (0,1):
                route = OpenSourceCells(w,direction)
                groups = defaultdict(list)
                for i,(r,(s,_)) in enumerate(zip(rows,inputs,strict=True)):
                    if r['direction']!=direction:
                        continue
                    known = all(cell.tobytes() in route.known for cell in palette[s])
                    if r['group']=='news':
                        assert known != r['open']
                    groups[(known,len(s))].append(i)
                for (known,_), members in groups.items():
                    for start in range(0,len(members),32):
                        ix=members[start:start+32]
                        images=[palette[inputs[i][0]] for i in ix]
                        answers = w.translate(images,direction) if known else [route.translate(img) for img in images]
                        for i,(cell,stop) in zip(ix,answers,strict=True):
                            item = emit(archive,f'{i}/{arm}',cell)
                            rows[i]['answers'][arm] = dict(**item, stop=bool(stop),text=readback(w,cell,names),
                                exact=bool(stop and np.array_equal(cell,palette[inputs[i][1]])))
            print('generated',arm,len(rows),flush=True)
        archive.writestr('records.json',json.dumps(rows,ensure_ascii=False))
    tmp.replace(ANSWERS)
    replay = {}
    with ZipFile(ANSWERS) as archive:
        for arm in ARMS:
            w = restore(arm,'adapted','cpu')
            routes = [OpenSourceCells(w,t) for t in (0,1)]
            sample = [i for group in ('news','edit','roles','reversed_roles') for i in [j for j,r in enumerate(rows) if r['group']==group][:4]]
            sample += [i for i,r in enumerate(rows) if r.get('open')][:4]
            checked = []
            for i in sample:
                r = rows[i]
                source = unpack(archive.read(f'{i}/source.png'),r['input']['cells'])
                output,stop = routes[r['direction']].translate(source)
                checked.append(dict(index=i,same=bool(stop==r['answers'][arm]['stop'] and png(output)==archive.read(f'{i}/{arm}.png'))))
            replay[arm] = checked
    base.dump(PUB/'generation.json',dict(archive=base.sha(ANSWERS),fits={m:base.sha(MODEL/m/'adapted/fit.json') for m in ARMS[1:]},
                                         rows=len(rows),replay=replay,data=base.sha(PUB/'data.json')))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('task',choices=['prose','translations','both'])
    args=cli.parse_args()
    configure_cuda_cli()
    with gpu_owner():
        for task,receipt,run in [('prose','prose.json',prose),('translations','generation.json',translations)]:
            if args.task in (task,'both') and not (PUB/receipt).exists():
                run()
