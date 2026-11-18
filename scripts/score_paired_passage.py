"""Decode ordered question images and archive every generated answer and termination."""
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_model

from scripts.fit_mixed_ink_reader import predict
from scripts.train_paired_passage import WORK, pad
from scripts.prepare_pictured_entry_answers import sha, write_json


def archive(name, streams, stops):
    sizes = [len(s)//128 for s in streams]
    np.savez_compressed(WORK/(name+".npz"), pixels=np.frombuffer(b"".join(streams),np.uint8).reshape(-1,128),
                        offsets=np.cumsum([0]+sizes), stopped=np.array(stops,bool))


@torch.inference_mode()
def evaluate(net, corpus):
    began = monotonic()
    features = net.features(corpus.bank)
    prompts, labels = corpus.requests()
    plans, stops, tag_exact = [], [], []
    groups = list(corpus.pairs[:,0]) + corpus.groups
    assert len(groups)==len(prompts)
    work, index, instructions = [], {}, []
    for start in range(0,len(prompts),128):
        chunk = prompts[start:start+128]
        q,lens = pad(chunk, features.device)
        logits = net.plan(features,q,lens).argmax(-1).cpu().numpy()
        for i, (row,actions) in enumerate(zip(chunk,logits,strict=True),start):
            tags = actions[:len(row)]
            tag_exact.append(bool(np.array_equal(tags,labels[i])))
            segments, stop = net.regions(tags)
            stops.append(stop)
            program = []
            for kind,a,b in segments:
                if kind=="prompt":
                    program.append(corpus.bank[row[a:b]].tobytes())
                else:
                    g=groups[i]
                    part = row[a:b]
                    key = (g,part.tobytes())
                    if key not in index:
                        index[key]=len(work)
                        work.append((corpus.passages[g],part))
                    program.append(index[key])
            instructions.append(program)
    counts = {}
    n = len(corpus.pairs)
    targets = [corpus.target(i) for i in range(len(prompts))]
    for name in ("candidate","original_reader"):
        if name=="original_reader":
            load_model(net,str(WORK/"planner_with_original_reader.safetensors"))
        decoded = predict(net.reader,features,work)
        pieces = [corpus.bank[p[a:b]].tobytes() for (p,_),(_,a,b) in zip(work,decoded,strict=True)]
        streams = [b"".join(pieces[v] if isinstance(v,int) else v for v in program)
                   for program in instructions]
        archive(name,streams,stops)
        ok = np.array([s and o==t for s,o,t in zip(stops,streams,targets,strict=True)])
        counts[name] = dict(pair_exact=int(ok[:n].sum()), both_orders=int(ok[:n].reshape(-1,2).all(1).sum()),
                            single_exact=int(ok[n:].sum()), pair_stopped=sum(stops[:n]),
                            single_stopped=sum(stops[n:]), archive_sha256=sha(WORK/(name+".npz")))
        if name=="candidate":
            # Canonicalize by intact question images. The other order has identical pixels
            # with the same passage, so reuse the deterministic output for both requests.
            erased, ended = [], []
            for i in range(0,n,2):
                g,a,b=corpus.pairs[i]
                assert tuple(corpus.pairs[i+1])==(g,b,a)
                j = i+int(corpus.bank[corpus.questions[a]].tobytes() > corpus.bank[corpus.questions[b]].tobytes())
                erased.extend([streams[j]]*2)
                ended.extend([stops[j]]*2)
            archive("erased",erased,ended)
            ok = np.array([s and o==t for s,o,t in zip(ended,erased,targets[:n],strict=True)])
            counts["erased"]=dict(pair_exact=int(ok.sum()),both_orders=int(ok.reshape(-1,2).all(1).sum()),
                                  archive_sha256=sha(WORK/"erased.npz"))
    write_json(WORK/"evaluation.json",dict(counts=counts, held_pairs=n, held_singles=len(prompts)-n,
               entire_plan_exact_pairs=sum(tag_exact[:n]),entire_plan_exact_singles=sum(tag_exact[n:]),
               unique_source_queries=len(work), seconds=monotonic()-began))
    print(counts,flush=True)
