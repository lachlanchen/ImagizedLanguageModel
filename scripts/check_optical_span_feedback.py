"""Measure visual target-span feedback into a frozen decoder with supplied boundaries."""
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic
import numpy as np
from huggingface_hub import snapshot_download
import torch
from transformers import MarianTokenizer, MarianMTModel
from ilm.visual_lm.field_to_glyph import load_weights
from ilm.visual_lm.optical_state_reader import OpticalStateReader
from ilm.visual_lm.glyph_measure import weighted_cross_attention
from scripts.reserve_target_glyph_book_panel import PUB, ROOT, dump, library, sha
from scripts.train_field_to_glyph import material, pack
from scripts.train_readout_objective import TEACHERS, decoder_hidden, project


def segment(tok, row, direction, encoded):
    target = row["text"][direction]
    cells = row["cells"][direction]
    pieces = tok.spm_target.encode_as_immutable_proto(target).pieces
    if tok.convert_tokens_to_ids([p.piece for p in pieces])+[tok.eos_token_id]!=encoded:
        raise ValueError("Target codec mismatch")
    result=[];last=0
    for p in pieces:
        if p.surface!=target[p.begin:p.end] or p.begin!=last:
            raise ValueError("Incomplete original optical span")
        result.append(tuple(cells[p.begin:p.end]))
        last=p.end
    if last!=len(cells): raise ValueError("Pictured target slice mismatch")
    return result


def averages(counts, embeddings):
    ids = list(counts)
    if len(ids)==1: return embeddings[ids[0]]
    w = torch.tensor([counts[i] for i in ids],dtype=embeddings.dtype)
    return (embeddings[ids]*w[:,None]).sum(0)/w.sum()


@torch.inference_mode()
def measure(t, train, dev, features):
    name, revision=TEACHERS[t]
    folder=snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=revision,local_files_only=True)
    tok=MarianTokenizer.from_pretrained(folder,local_files_only=True)
    model=MarianMTModel.from_pretrained(folder,local_files_only=True,attn_implementation="eager").eval()
    lookup=model.model.decoder.embed_tokens.weight
    counts=defaultdict(Counter); lengths=defaultdict(Counter); global_counts=Counter()
    for start in range(0,len(train),256):
        group=train[start:start+256]
        labels=tok(text_target=[r["text"][t] for r in group]).input_ids
        for row,ids in zip(group,labels,strict=True):
            for key,k in zip(segment(tok,row,t,ids),ids[:-1],strict=True):
                counts[key][k]+=1;lengths[len(key)][k]+=1;global_counts[k]+=1
    bank={key:averages(c,lookup) for key,c in counts.items()}
    length_bank={n:averages(c,lookup) for n,c in lengths.items()}
    default=averages(global_counts,lookup)
    pools=defaultdict(list)
    for key in bank:pools[len(key)].append(key)
    rng=np.random.default_rng(3241+t)
    exchanged={}
    for group in pools.values():
        rng.shuffle(group)
        for i,key in enumerate(group):exchanged[key]=bank[group[(i+1)%len(group)]]
    reader=OpticalStateReader().eval()
    receipt,weights=load_weights(ROOT/"models/readout_objective"/("en","zh")[t]/"prediction")
    reader.load_state_dict(weights)
    stats=Counter()
    diagnostics={k:Counter() for k in ("span","exchanged")}
    for j in range(0,len(dev),16):
        group=dev[j:j+16]
        strings=[r["text"][t] for r in group]
        labels=tok(text_target=strings,padding=True,return_tensors="pt").input_ids
        mask=labels.ne(tok.pad_token_id)
        src=pack([r["cells"][1-t] for r in group],"cpu")
        field,valid,mass=reader.states(features,src)
        ids=model.prepare_decoder_input_ids_from_labels(labels)
        genuine=model.model.decoder.embed_tokens(ids)
        folded=genuine.clone(); swapped=genuine.clone()
        for b,row in enumerate(group):
            original=labels[b,mask[b]].tolist()
            for p,key in enumerate(segment(tok,row,t,original),1):
                base=length_bank.get(len(key),default)
                folded[b,p]=bank.get(key,base);swapped[b,p]=exchanged.get(key,base)
                stats["preceding_spans"]+=1
                stats["seen"]+=key in bank
                stats["empty"]+=not key
                stats["multiple_training_ids"]+=len(counts.get(key,{}))>1
        with weighted_cross_attention(model,mass):
            def decode(v):
                return model.model.decoder(inputs_embeds=v,encoder_hidden_states=field,encoder_attention_mask=valid.long(),
                                           use_cache=False).last_hidden_state
            original=decoder_hidden(model,field,valid.long(),labels)
            if not torch.equal(original,decode(genuine)): raise ValueError("Embedding input path changed native states")
            mask[:,0]=False
            truth=labels[mask]
            p=project(model,original[mask]).log_softmax(-1)
            stats["native_nll"]-=p.gather(1,truth[:,None]).sum(dtype=torch.float64).item()
            for mode,data in (("span",folded),("exchanged",swapped)):
                q=project(model,decode(data)[mask]).log_softmax(-1)
                a=diagnostics[mode]
                a["same"]+=(p.argmax(-1)==q.argmax(-1)).sum().item()
                a["kl"]+=(p.exp()*(p-q)).sum(dtype=torch.float64).item()
                a["nll"]-=q.gather(1,truth[:,None]).sum(dtype=torch.float64).item()
    n=stats["preceding_spans"]
    return dict(preceding=n,initial=len(dev),seen=stats["seen"],empty=stats["empty"],
                multiple=stats["multiple_training_ids"],types=len(counts),training_spans=sum(global_counts.values()),
                reader_sha256=receipt["weights"],revision=revision,native_nll=stats["native_nll"]/n,
                conditions={k:dict(same=v["same"],agreement=v["same"]/n,nll=v["nll"]/n,kl=v["kl"]/n)
                            for k,v in diagnostics.items()})


def main():
    begin=monotonic();torch.set_num_threads(2)
    train,dev=library("train"),library("dev")
    features=material("cpu")[0]
    results={lang:measure(t,train,dev,features) for t,lang in enumerate(("en","zh"))}
    passed=all(r["conditions"]["span"]["agreement"]>.9 and
        r["conditions"]["span"]["agreement"]-r["conditions"]["exchanged"]["agreement"]>=.05 for r in results.values())
    dump(PUB/"span_feedback.json",dict(results=results,passed=passed,seconds=monotonic()-begin,
         program=sha(Path(__file__)),protocol=sha(ROOT/"references/optical_span_feedback_protocol.md")))
    print(results,passed,flush=True)


if __name__=="__main__":
    main()
