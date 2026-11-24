"""Measure conditional phrase preference and actual emitted Song bitmaps."""
import json
from time import monotonic

import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.verse_phrase import phrase_score, search_pair
from scripts.evaluate_verse_canvas import aligned, cases, generate
from scripts.prepare_dialogue_ink import dump, sha
from scripts.prepare_verse_canvas import MODEL, corpus
from scripts.prepare_verse_contrast import PUB, WORK, load_song
from scripts.train_span_choice import gpu_owner
from scripts.train_verse_canvas import banks
from scripts.write_verse_canvas import load
from scripts.measure_writer_energy import snapshot, consumed


@torch.inference_mode()
def ordered(net, bank, vocab, rows, donor, gaps, exchange=False):
    count = len(rows)
    pred = rows[:,9:16].copy()
    cert = np.empty(count,bool)
    pref = np.full((count,4),np.nan)
    ranks = np.empty((count,2),np.int32) if not exchange else None
    for n in (5,7):
        subset = np.flatnonzero(rows[:,1]==n)
        for i in range(0,len(subset),48):
            sel = subset[i:i+48]
            image,foreign,truth = [torch.as_tensor(arr.copy(),device=bank.device).long() for arr in
                  (rows[sel,2:2+n],rows[donor[sel],2:2+n],rows[sel,9:9+n])]
            hidden = torch.tensor(gaps[sel,:n],device=bank.device)
            start = hidden.long().argmax(1)
            target = bank[0,truth.masked_fill(hidden,0)]
            selected_source = foreign if exchange else image
            written,certificate = search_pair(net,bank[0,selected_source],target,~hidden,bank[0])
            cert[sel] = certificate.cpu().numpy()
            # Copy each result, including its certificate, before the next batch.
            for j in (0,1):
                pred[sel,(start+j).cpu().numpy()] = written[:,j].cpu().numpy()
            if exchange:
                continue
            donor_truth = torch.as_tensor(rows[donor[sel],9:9+n].copy(), device=bank.device).long()
            donor_start = torch.tensor(gaps[donor[sel],:n], device=bank.device).long().argmax(1)
            for k,source in enumerate((image,foreign)):
                if k == 0:
                    score,order = phrase_score(net,bank[0,source],truth,bank[0],start,vocab,return_ranks=True)
                    ranks[sel] = order.cpu().numpy()
                else:
                    score = phrase_score(net,bank[0,source],truth,bank[0],start,vocab)
                pref[sel,k] = score.cpu().numpy()
                pref[sel,k+2] = phrase_score(net,bank[0,source],donor_truth,bank[0],donor_start,vocab).cpu().numpy()
    return pred,cert,pref,ranks


def author_interval(left,right,rows):
    _,ids = np.unique(rows[:,0],return_inverse=True)
    delta = left.astype(float)-right.astype(float)
    totals = np.bincount(ids,weights=delta)
    sizes = np.bincount(ids)
    sample = np.random.default_rng(89176).integers(len(sizes),size=(5000,len(sizes)))
    boot = totals[sample].sum(1)/sizes[sample].sum(1)
    return dict(difference=float(delta.mean()),interval=np.quantile(boot,[.025,.975]).tolist(),
                gain=int((left&~right).sum()),loss=int((~left&right).sum()))


def strip(tiles,line,path):
    bits = np.unpackbits(tiles[line],axis=1).reshape(-1,32,32)
    Image.fromarray((255-255*np.concatenate(list(bits),axis=1)).astype(np.uint8)).save(path)


def report(rows,gaps,donor,out,cert,pref,ranks,cost,tang,song,idx,meta):
    length = rows[:,1]
    full = np.arange(7)<length[:,None]
    truth = rows[:,9:16]
    training_pairs = set()
    for r in tang["train"]:
        line = r[9:9+r[1]]
        training_pairs.update(zip(line[:-1],line[1:]))
    masks = {"gap":gaps,"blank":full}
    stats = {}
    exact = {}
    for key,prediction in out.items():
        task = key.split("/")[1]
        mask = masks[task]
        if not np.array_equal(prediction[~mask],truth[~mask]):
            raise ValueError("Changed supplied ink")
        hit = truth == prediction
        correct = (hit|~mask).all(1)
        exact[key] = correct
        stats[key] = dict(cells=int(mask.sum()),correct=int((hit&mask).sum()),exact=int(correct.sum()))
        if task=="gap":
            written = prediction[gaps].reshape(-1,2)
            stats[key].update(in_training=sum(tuple(pair) in training_pairs for pair in written),
                              repetition=int((written[:,0]==written[:,1]).sum()))
    preference = {}
    for key,p in pref.items():
        ok = np.isfinite(p[:,:2]).all(1)
        scored = p[np.isfinite(p).all(1)]
        margin = (scored[:,0]-scored[:,1])+(scored[:,3]-scored[:,2])
        preference[key] = dict(scorable=int(ok.sum()),win=int((p[ok,0]>p[ok,1]).sum()),
                               ties=int((p[ok,0]==p[ok,1]).sum()),
                               reciprocal_scorable=len(scored),reciprocal_win=int((margin>0).sum()),
                               reciprocal_ties=int((margin==0).sum()))
    main = "ink-contrast/gap/pair"
    comparisons = {k:author_interval(exact[main],value,rows) for k,value in exact.items() if "/gap/" in k}
    index = {"first":0}
    sensitive = np.flatnonzero(exact[main]&~exact["ink-contrast/gap/exchanged-pair"])
    if len(sensitive):
        index["repair"] = int(sensitive[0])
    readback = json.loads((WORK/"readback.json").read_text())
    sample = {}
    for name,i in index.items():
        n = length[i]
        lines = dict(source=rows[i,2:2+n],damaged=np.where(gaps[i,:n],0,truth[i,:n]),
                     output=out[main][i,:n],reference=truth[i,:n])
        sample[name] = {"poet":readback["authors"][rows[i,0]],"row":int(idx[i])}
        for label,ids in lines.items():
            strip(song["tiles"][0],ids,PUB/f"{name}-{label}.png")
            sample[name][label] = "".join(readback["shapes"][j] if j else "□" for j in ids)
    rank_summary = {k:dict(first_top8=int((v[:,0]<=8).sum()),
                          second_first_if_true_first=int((v[:,1]==1).sum()),
                          both_reachable=int(((v[:,0]<=8)&(v[:,1]==1)).sum()))
                    for k,v in ranks.items()}
    archive = PUB/"outputs.npz"
    np.savez_compressed(archive,rows=rows,indices=idx,gaps=gaps,donor=donor,**out,
                        **{"rank/"+k:v for k,v in ranks.items()},**{"cert/"+k:v for k,v in cert.items()},**{"score/"+k:v for k,v in pref.items()})
    known = (truth[gaps]<=meta["train_shapes"]).reshape(-1,2).all(1)
    report = dict(pages=len(rows),poets=len(set(rows[:,0])),same_poet_donors=int((rows[:,0]==rows[donor,0]).sum()),
                  available_pairs=int(known.sum()),unknown_blank_cells=int(((truth>meta["train_shapes"])&full).sum()),
                  summary=stats,comparisons=comparisons,source_preference=preference,
                  ranks=rank_summary,certified={k:int(v.sum()) for k,v in cert.items()},costs=cost,
                  archive=sha(archive),examples=sample)
    dump(PUB/"results.json",report)
    print({k:v for k,v in stats.items() if k.endswith("/pair")},sample,flush=True)


def main():
    meta,tang = corpus()
    _,song = load_song()
    idx,rows,donor,gaps = cases(song,seed=25173)
    full = np.arange(7)<rows[:,1,None]
    same = np.arange(len(rows))
    out,cert,pref,ranks,cost = {},{},{},{},{}
    with gpu_owner():
        for mode in ("ink","byte"):
            bank = banks(song,mode,"cuda")
            for arm in ("bound","paired","contrast"):
                name = mode+"-"+arm
                clock = monotonic()
                before = snapshot()
                net,manifest = load(MODEL,name,"cuda")
                for task,mask in (("blank",full),("gap",gaps)):
                    for policy,d in (("confidence",same),("exchanged",donor),("left",same)):
                        out[f"{name}/{task}/{policy}"] = generate(
                            net,bank,rows,d,mask,0,"left" if policy=="left" else "confidence")
                for policy in ("pair","exchanged-pair"):
                    prediction,certificate,score,order = ordered(net,bank,manifest["vocab"],rows,donor,gaps,exchange=policy=="exchanged-pair")
                    out[f"{name}/gap/{policy}"] = prediction
                    cert[f"{name}/{policy}"] = certificate
                    if policy == "pair":
                        pref[name] = score
                        ranks[name] = order
                torch.cuda.synchronize()
                cost[name] = dict(seconds=monotonic()-clock,energy=consumed(before,snapshot()))
                print(name,cost[name],flush=True)
    count = aligned(tang["train"],rows)
    for task,mask in (("blank",full),("gap",gaps)):
        out[f"counts/{task}/original"] = np.where(mask,count,rows[:,9:16])
    report(rows,gaps,donor,out,cert,pref,ranks,cost,tang,song,idx,meta)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()
