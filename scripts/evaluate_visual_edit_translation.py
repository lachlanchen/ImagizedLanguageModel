"""Score stopped translations and quotations against real chapter-held-out image pairs."""
import argparse
import fcntl
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file

from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, STOP, VisualEditWriter, visual_input
from scripts.train_visual_edit_translation import OUT, ROOT, bank, guard, inputs, load, padded, require, sha, write_new
from scripts.train_visual_extractive_reader import save_page


def local_program(rows):
    totals = [defaultdict(Counter),defaultdict(Counter)]
    for row in rows:
        x = row['x']
        steps = [[] for _ in range(len(x)+1)]
        cursor = 0
        for action,tile in zip(row['a'],row['o']):
            if action<=INSERT:
                steps[cursor].append(tile)
            if action in (COPY,DROP,REPLACE):
                cursor += 1
        padded_x = [-1,*x,-2,-3]
        for j,block in enumerate(steps):
            key = tuple(padded_x[j:j+3])
            totals[0][key[1]][tuple(block)]+=1
            totals[1][key][tuple(block)]+=1
    best = []
    for table in totals:
        best.append({k:(v.most_common(1)[0][0],sum(v.values())) for k,v in table.items()})
    return best


def local_answer(x,program):
    padded_x = [-1,*x,-2,-3]
    output = []
    for j in range(len(x)+1):
        key = tuple(padded_x[j:j+3])
        block,n = program[1].get(key,(None,0))
        if n<2:
            block = program[0].get(key[1],((x[j],) if j<len(x) else (),0))[0]
        output.extend(block)
    return output


def restore(checkpoint,device):
    weights = load_file(str(checkpoint))
    model = VisualEditWriter(weights['landmarks']).to(device).eval()
    model.load_state_dict(weights)
    return model


@torch.inference_mode()
def streams(net,ink,rows,question,start):
    results = []
    q = torch.as_tensor(question,device=ink.device)[None]
    for i in range(0,len(rows),48):
        guard(start)
        x,lens = padded(rows[i:i+48])
        b = len(x)
        qlens = torch.full((b,),q.shape[1],device=ink.device)
        pixels,n,stop,actions = net.generate(ink[torch.as_tensor(x,device=ink.device)],
            torch.as_tensor(lens,device=ink.device).long(),visual_input(ink[q.expand(b,-1)],net.landmarks),qlens)
        pic = np.packbits(pixels.cpu().numpy(),axis=-1)
        for im,length,done,act in zip(pic,n.cpu().numpy(),stop.cpu().numpy(),actions.cpu().numpy()):
            results.append((im[:length].copy(),bool(done),act[act>=0].astype(np.int8)))
    return results


def ci(rows,control):
    groups = defaultdict(lambda:np.zeros(2))
    for r in rows:
        groups[r['group']] += [r[control]['distance']-r['translation']['distance'],r['length']]
    counts = np.stack(list(groups.values()))
    rng = np.random.default_rng(4901)
    sets = rng.integers(len(counts),size=(2000,len(counts)))
    draws = counts[sets].sum(1)
    return (100*np.quantile(draws[:,0]/draws[:,1],[.025,.975])).tolist()


def evaluate(name,model,tiles,program,train,start,out):
    records = inputs(name)
    raw = [(i,c) for i,c in enumerate(records) if c['x']]
    cases = [c for i,c in raw]
    x = [r['x'] for r in cases]
    prompts = load(OUT/'preparation.json')['question_tiles']
    order = sorted(range(len(x)),key=lambda i:len(x[i]))
    other = np.arange(len(x))
    other[order] = np.roll(order,-1)
    require(all(x[i]!=x[j] for i,j in enumerate(other)), 'Exchange requires different pictured sources')
    ink = torch.from_numpy(np.unpackbits(tiles,axis=1)).cuda().bool()
    streams_by_mode = {}
    for mode,seq,task in [('translation',x,0),('quote',x,1),('exchanged',[x[j] for j in other],0)]:
        streams_by_mode[mode]=streams(model,ink,seq,prompts[task][0],start)
    mapping = {bytes(v):i for i,v in enumerate(tiles)}
    seen_targets = {tuple(v['y']) for v in train}
    train_shapes = set(np.load(OUT/'train_shapes.npy').tolist())
    out.mkdir()
    scores = defaultdict(Counter)
    evidence,rows = {},[]
    with ZipFile(out/'streams.zip','x',ZIP_DEFLATED,compresslevel=2) as z:
        for j,(index,c) in enumerate(raw):
            source,gold = c['x'],c['y']
            glyphs = set(source)
            missing = [i for i,k in enumerate(gold) if k not in glyphs and '\u3400'<=c['target'][i]<='\u9fff']
            # Count full target length, distinct copied vs absent Han, and true nonverbatim pairs.
            hard = len(gold)>8 and bool(missing) and any(k in glyphs for k in gold)
            row = dict(index=index,group=c['g'],length=len(gold),hard=hard,missing_han=len(missing),
                       same_reference=source==gold,new_reference=tuple(gold) not in seen_targets)
            for k in ('translation','quote','exchanged','copy','local'):
                if k in streams_by_mode:
                    pix,stopped,actions = streams_by_mode[k][j]
                    ids = [mapping.get(bytes(v),-1) for v in pix]
                    z.writestr(f'{k}/{index}.bin',pix.tobytes())
                    z.writestr(f'{k}/{index}.actions',actions.tobytes())
                else:
                    ids = source if k=='copy' else local_answer(source,program)
                    pix,stopped = tiles[np.asarray(ids,np.int32)],True
                target = source if k=='quote' else gold
                dist = Levenshtein.distance(ids,target)
                exact = bool(stopped and dist==0)
                score = scores[k]
                score.update(cases=1,exact=int(exact),stopped=int(stopped),edits=dist,target_cells=len(target),
                    output_cells=len(ids),known_cells=sum(v in train_shapes for v in ids),hard=int(hard),
                    hard_exact=int(hard and exact),novel_hard_exact=int(hard and exact and row['new_reference']))
                row[k]=dict(cells=len(ids),distance=dist,exact=exact,stopped=stopped)
                if k=='translation':
                    for key,condition in [('first_long_mixed',hard and exact),('first_error',not exact),
                                          ('first_changed_success',source!=gold and exact)]:
                        if condition and key not in evidence:
                            evidence[key]=dict(index=index,source=c['source'],target=c['target'],chapter=c['g'],
                                               output_cells=len(ids),copied=int((actions==COPY).sum()),
                                               drawn=int(np.isin(actions,[REPLACE,INSERT]).sum()))
                            for field,img in [('source',tiles[source]),('reference',tiles[gold]),('actual',pix)]:
                                if len(img): save_page(out/(key+'_'+field+'.png'),img)
            rows.append(row)
    write_new(out/'rows.json',rows)
    intervals = {k:ci(rows,k) for k in ('copy','local','exchanged')}
    for values in scores.values():
        values['cer'] = values['edits']/values['target_cells']
    stats=dict(all_requests=len(records),pictured=len(raw),unsupported=len(records)-len(raw),groups=len(set(r['g'] for r in records)),
        outcome={k:dict(v) for k,v in scores.items()},improvement_ci_pp=intervals,examples=evidence,
        distinct_questions_both=sum(r['translation']['exact'] and r['quote']['exact'] and not r['same_reference'] for r in rows),
        different_answers=sum(not r['same_reference'] for r in rows),rows_sha256=sha(out/'rows.json'),streams_sha256=sha(out/'streams.zip'))
    stats['pass']=bool(scores['translation']['hard_exact']>0 and min(intervals[k][0] for k in ('copy','local'))>0
                       and scores['quote']['exact']>=.99*len(records))
    write_new(out/'report.json',stats)
    print(name,stats['outcome']['translation'],stats['improvement_ci_pp'],stats['pass'],flush=True)
    return stats


def run(checkpoint,destination):
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    net = restore(checkpoint,'cuda')
    train = inputs('train')
    local = local_program(train)
    tiles = bank()
    destination.mkdir()
    checks = {}
    for name in ('heldout','confirmation'):
        checks[name] = evaluate(name,net,tiles,local,train,start,destination/name)
        if name=='heldout' and not checks[name]['outcome']['translation']['hard_exact']:
            break
    write_new(destination/'report.json',dict(splits=list(checks),qualified=len(checks)==2 and all(v['pass'] for v in checks.values()),
               seconds=time.monotonic()-start,checkpoint_sha256=sha(checkpoint),
               peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=OUT/'fit/editor.safetensors')
    parser.add_argument('--destination',type=Path,default=OUT/'evaluation')
    args=parser.parse_args()
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.checkpoint,args.destination)
