"""Measure how pooling changes the physical next-span decision at real book prefixes."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer
from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.optical_span_rollout import OpticalSpanWriter
from scripts.check_optical_span_feedback import segment
from scripts.reserve_optical_span_rollout import PUB, dump, sha, read
from scripts.reserve_target_glyph_book_panel import library
from scripts.train_field_to_glyph import material, pack


def compare(h, p, truth, modal, choices):
    latent = (h@p.keys.T+p.bias).softmax(-1)
    scores = torch.zeros((len(h),choices))
    scores.index_add_(1,p.surface_index,latent[:,p.key_index]*p.frequency)
    top, index = scores.topk(2)
    best=index[:,0]
    greedy=modal[latent.argmax(1)]
    alternative=scores.gather(1,greedy[:,None])[:,0]
    return Counter(n=len(h),same=int(best.eq(greedy).sum()),pooled_correct=int(best.eq(truth).sum()),
                   single_correct=int(greedy.eq(truth).sum()),unknown=int(truth.lt(0).sum()),
                   margin_gt_002=int((top[:,0]-top[:,1]>.02).sum()),
                   regret_mass=float((top[:,0]-alternative).double().sum()),
                   abs_mass_error=float((scores.sum(1)-1).abs().double().sum()))


@torch.inference_mode()
def run(lang,t,dev):
    model=OpticalSpanWriter(lang,"cpu")
    entries=json.loads(gzip.decompress((PUB/(lang+"-span-counts.json.gz")).read_bytes()))
    spans=sorted({tuple(s) for _,_,s,_ in entries})
    indices={s:i+1 for i,s in enumerate(spans)}
    freq=Counter()
    for _,_,s,n in entries: freq[tuple(s)]+=n
    lengths=defaultdict(list)
    for s in spans: lengths[len(s)].append(s)
    fallback={}
    for length,items in lengths.items():
        w=torch.tensor([freq[s] for s in items])
        fallback[length]=(model.feedback[[indices[s] for s in items]]*w[:,None]).sum(0)/w.sum()
    default=sum(fallback[k]*sum(freq[s] for s in vals) for k,vals in lengths.items())/sum(freq.values())
    info=read(PUB/(lang+"-fit.json"))
    local=snapshot_download(info["model"],revision=info["revision"],local_files_only=True)
    tok=MarianTokenizer.from_pretrained(local,local_files_only=True)
    modes=[]
    for p in model.bank:
        # Stable tie rule uses the smaller surface index; test records are never consulted.
        best=[None]*len(p.keys)
        for key,s,prob in zip(p.key_index.tolist(),p.surface_index.tolist(),p.frequency.tolist(),strict=True):
            pair=(prob,-s)
            if best[key] is None or pair>best[key]: best[key]=pair
        modes.append(torch.tensor([-s for _,s in best]))
    features=material("cpu")[0]
    results=[Counter(),Counter()]
    for j in range(0,len(dev),12):
        rows=dev[j:j+12]
        labels=tok(text_target=[r["text"][t] for r in rows]).input_ids
        texts=[segment(tok,r,t,ids) for r,ids in zip(rows,labels,strict=True)]
        w=max(map(len,texts))+1
        inp=model.begin.repeat(len(rows),w,1)
        truth=torch.full((len(rows),w),-2)
        for b,seq in enumerate(texts):
            for i,s in enumerate(seq):
                truth[b,i]=indices.get(s,-1)
                inp[b,i+1]=model.feedback[indices[s]] if s in indices else fallback.get(len(s),default)
            truth[b,len(seq)]=0
        f,mask,mass=model.reader.states(features,pack([r["cells"][1-t] for r in rows],"cpu"))
        with weighted_cross_attention(model.attention_owner,mass):
            h=model.decoder(inputs_embeds=inp,encoder_hidden_states=f,encoder_attention_mask=mask,use_cache=False).last_hidden_state
        for phase in (0,1):
            active=(torch.arange(w)[None,:].gt(0)==bool(phase)) & truth.ne(-2)
            results[phase].update(compare(h[active],model.bank[phase],truth[active],modes[phase],len(model.shapes)))
    return dict(first=results[0],later=results[1])


if __name__=="__main__":
    torch.set_num_threads(2)
    dev=library("dev")
    out={lang:run(lang,t,dev) for t,lang in enumerate(("en","zh"))}
    dump(PUB/"surface-mass.json",dict(results=out,pairs=len(dev),program=sha(Path(__file__)),
                                      interpretation="Supplied target boundaries and pictured history; full stop included."))
    print(out,flush=True)
