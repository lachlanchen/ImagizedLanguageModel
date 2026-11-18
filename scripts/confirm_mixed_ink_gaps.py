"""Evaluate the frozen copy/write policy on new rendered passages and replay old interfaces."""
import fcntl
import time

import numpy as np
import torch

from scripts.confirm_joint_reference_writer import integrated_check
from scripts.distil_glyph_planner import (OUT as TRANSFER, FRESH, HINTS, JOINT, PREVIOUS, DIVERSITY,
    WEIGHT, NEIGHBOR, PLAN, DATA, corpora, load_split, local_counts, matching_pairs, prepare)
from scripts.fit_mixed_ink_reader import (OUT, ROOT, PROMPTS, PixelBank, Rasterizer, draw, guard, load,
    predict, require, sha, verified_split, visual_input, write_new)
from scripts.infer_mixed_ink import restore, WEIGHTS


def run():
    began = time.monotonic()
    guard(began)
    torch.set_num_threads(2)
    replay = load(OUT/'replay/report.json')
    require(replay['identical']==replay['total'],'Unresolved historical discrepancy')
    folder = OUT/'fresh'
    folder.mkdir()
    original,combined = corpora()
    older = (FRESH,HINTS/'confirmation',JOINT/'confirmation',PREVIOUS,DIVERSITY/'fresh',DIVERSITY/'pool',
             WEIGHT/'fresh',NEIGHBOR/'fresh',PLAN/'fresh',TRANSFER/'fresh')
    data,raw = prepare(original,load_split(DATA/'validation.npz'),np.load(older[-1]/'pixels.npy'),
                       buckets=range(140,148),destination=folder,excluded=[verified_split(p) for p in older])
    bank = PixelBank(raw)
    renderer = Rasterizer()
    prompts = [[bank.encode([renderer.tile(c) for c in s]) for s in v] for v in PROMPTS]
    tiles = np.frombuffer(b''.join(bank.tiles),np.uint8).reshape(-1,128)
    reader,writer = restore(WEIGHTS,'cuda')
    features = visual_input(torch.from_numpy(np.unpackbits(tiles,axis=1)).cuda(),writer.landmarks)
    x,y = data['context'],data['target']
    torch.set_float32_matmul_precision('high')
    pic,n,stop = draw(writer,features,x,began)
    path = folder/'generated.npz'
    np.savez_compressed(path,packed=pic,length=n,stopped=stop)
    compatibility = integrated_check(writer,data,raw,began,generated=path)
    torch.set_float32_matmul_precision('highest')
    eligible = (x[:,20:28]!=0).all(1)
    absent = ~(y[:,:,None]==x[:,None]).any((1,2))
    i,j = matching_pairs(x,y,data['group'])
    report = dict(windows=len(x),groups=len(np.unique(data['group'])),all_absent=int(absent.sum()),
                  eligible_neighbors=int(eligible.sum()),pairs=len(i),local_exact=int(local_counts(combined,data).sum()),
                  historical=compatibility)
    for mode,version in [('familiar',0),('unseen',2)]:
        scores,all_hits = {},[]
        for role,texts in enumerate(prompts):
            pred = predict(reader,features,[(v,texts[version]) for v in x])
            truth = raw[(y,x[:,20:24],x[:,24:28])[role]]
            hits = []
            for k,(gen,a,b) in enumerate(pred):
                cells = pic[k,:n[k]] if gen else raw[x[k,a:b]]
                hits.append(bool((stop[k] if gen else True) and np.array_equal(cells,truth[k])))
            hits = np.array(hits)
            all_hits.append(hits)
            keep = eligible if role else np.ones(len(x),bool)
            scores[('gap','left','right')[role]] = dict(exact=int(hits[keep].sum()),total=int(keep.sum()),
                generated=int(pred[keep,0].sum()),**({} if role else dict(all_absent_exact=int(hits[absent].sum()),
                both_correct=int((hits[i]&hits[j]).sum()))))
            np.savez_compressed(folder/f'{mode}_{role}.npz',actions=pred,exact=hits)
        scores['all_three'] = int((eligible & np.logical_and.reduce(all_hits)).sum())
        report[mode] = scores
    guard(began)
    report.update(seconds=time.monotonic()-began,output_sha256=sha(path),preparation_sha256=sha(folder/'preparation.json'))
    write_new(folder/'report.json',report)
    print(report,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
