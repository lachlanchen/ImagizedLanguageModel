"""Separate optical state error from visual attention-mass error in a frozen translator."""

from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ilm.visual_lm.glyph_measure import expand_states, weighted_cross_attention
from ilm.visual_lm.optical_state_reader import OpticalStateReader
from scripts.check_glyph_measure import exact_assignments, until_stop
from scripts.train_optical_state_reader import ROOT, PUB, RUN, MODELS, TEACHERS, library, material, pack, read, sha, dump
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def compare(t, pairs, train):
    name,rev=TEACHERS[t]
    path=snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=rev,local_files_only=True)
    native=MarianMTModel.from_pretrained(path,local_files_only=True,attn_implementation="eager").eval().to("cuda")
    tok=MarianTokenizer.from_pretrained(path,local_files_only=True)
    lang=("en","zh")[t]
    reader=OpticalStateReader().eval().to("cuda")
    reader.load_state_dict(load_file(str(MODELS/lang/"weights.safetensors")))
    features=material("cuda")[0]
    counter=Counter()
    for j in range(0,len(train),256):
        for seq in tok([r["text"][1-t] for r in train[j:j+256]]).input_ids:
            counter.update(seq)
    accum=defaultdict(Counter)
    vocab=defaultdict(Counter)
    sequences=[]
    for start in range(0,len(pairs),12):
        batch=pairs[start:start+12]
        texts=[r["text"][1-t] for r in batch]
        inputs=tok(texts,padding=True,return_tensors="pt").to("cuda")
        labels=tok(text_target=[r["text"][t] for r in batch],padding=True,return_tensors="pt").input_ids.to("cuda")
        valid_y=labels.ne(native.config.pad_token_id)
        hidden=native.model.encoder(**inputs).last_hidden_state
        ids=exact_assignments(tok,texts,inputs)
        target,valid,weight=expand_states(hidden,ids)
        prediction, grid, log_mass=reader.states(features,pack([r["cells"][1-t] for r in batch],"cuda"))
        if not torch.equal(valid,grid):
            raise ValueError("Different visual source lattice")
        squared=((prediction-target)/reader.spread).square().mean(-1)
        for i, seq in enumerate(ids):
            local=inputs.input_ids[i, inputs.attention_mask[i].bool()].tolist()
            local_mse=squared[i,:len(seq)].cpu()
            for j,piece in enumerate(local):
                if piece==tok.eos_token_id: continue
                frequency=counter[piece]
                band="0" if frequency==0 else "1-9" if frequency<10 else "10-99" if frequency<100 else "100+"
                indices=[k for k,v in enumerate(seq) if v==j]
                vocab[band]["pieces"]+=1
                vocab[band]["sse"]+=float(local_mse[indices].mean())
        original=None
        written={}
        for arm, field, m in (("original",target,weight),("mass",target,log_mass),
                              ("field",prediction,weight),("both",prediction,log_mass)):
            with weighted_cross_attention(native,m):
                enc=BaseModelOutput(last_hidden_state=field)
                logits=native(encoder_outputs=enc,attention_mask=valid,labels=labels,use_cache=False).logits[valid_y]
                now=logits.float().log_softmax(-1)
                if original is None:
                    original=now
                truth=labels[valid_y].unsqueeze(-1)
                delta=(now-original).gather(-1,truth).abs()
                kl=(original.exp()*(original-now)).sum(-1)
                out=native.generate(encoder_outputs=enc,attention_mask=valid,max_new_tokens=128,
                                    num_beams=1,do_sample=False,forced_eos_token_id=None)
                written[arm]=[until_stop(s,native.config.eos_token_id) for s in out.cpu()]
            a=accum[arm]
            a["positions"]+=len(now)
            a["kl_sum"]+=kl.double().sum().item()
            a["ref_abs_sum"]+=delta.double().sum().item()
            a["max_ref_logp"] = max(a["max_ref_logp"], delta.max().item())
            a["top_changes"]+=int((now.argmax(-1)!=original.argmax(-1)).sum())
            a["equal_free"]+=sum(x==y for x,y in zip(written["original"],written[arm],strict=True))
            del logits, now
        sequences.extend({m:v[i] for m,v in written.items()} for i in range(len(batch)))
    for a in accum.values():
        a["kl"]=a["kl_sum"]/a["positions"]
        a["mean_ref_logp"]=a["ref_abs_sum"]/a["positions"]
    for b in vocab.values(): b["normalized_mse"]=b["sse"]/b["pieces"]
    return dict(conditions=accum,frequency=vocab),sequences


def main():
    configure_cuda_cli()
    dev, train=library("dev"),library("train")
    results, raw={},{}
    with gpu_owner():
        for t,lang in enumerate(("en","zh")):
            results[lang],raw[lang]=compare(t,dev,train)
    file=RUN/"factor-sequences.json"
    dump(file,raw)
    out=dict(program=sha(Path(__file__)),protocol=sha(ROOT/"references/optical_reader_factor_protocol.md"),
             training=sha(PUB/"training.json"),outputs=sha(file),directions=results)
    dump(PUB/"factor.json",out)
    print({l:{k:(round(v["kl"],4),v["equal_free"]) for k,v in d["conditions"].items()} for l,d in results.items()})


if __name__=="__main__":
    main()
