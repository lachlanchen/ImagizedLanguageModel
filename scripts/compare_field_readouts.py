"""Compare independently written glyphs with pretrained readouts on identical pictured passages."""
from pathlib import Path
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_field_to_glyph import requests, PUB, RUN, ROOT, read, dump, sha
from scripts.evaluate_optical_state_reader import as_answer
from scripts.evaluate_readout_objective import restore_reader, summarize, GEN
from scripts.evaluate_visual_library import blind_votes
from scripts.readout_work_interval import work_interval
from scripts.train_readout_objective import teacher_model, gpu_owner, configure_cuda_cli
from scripts.write_neural_glyphs import grid_bits


@torch.inference_mode()
def probes(rows,original):
    predictions=[{k:a[k] for k in ("prediction","regression","qwen","reference","foil")} for a in original]
    for t in (0,1):
        teacher,tok=teacher_model(t)
        nets={k:restore_reader(t,k) for k in ("regression","prediction")}
        ordered=[i for i,r in enumerate(rows) if r["t"]==t]
        for j in range(0,len(ordered),12):
            ids=ordered[j:j+12]
            cells=[grid_bits(RUN/"png"/f"{i:04d}"/"input.png",len(rows[i]["input"])) for i in ids]
            width=max(map(len,cells))
            handles=np.zeros((len(cells),width),np.int64)
            tiles=[np.zeros((1,128),np.uint8)]
            start=1
            for b,ink in enumerate(cells):
                n=len(ink)
                handles[b,:n]=np.arange(start,start+n)
                start+=n
                tiles.append(np.packbits(ink.reshape(n,1024),axis=1))
            f=torch.tensor(observations({"tiles":np.concatenate(tiles)},"ink"),device="cuda")
            src=torch.from_numpy(handles).cuda()
            texts=tok([rows[i]["source"] for i in ids],padding=True,return_tensors="pt").to("cuda")
            seq={"native":teacher.generate(**texts,**GEN).cpu()}
            for k,reader in nets.items():
                field,mask,mass=reader.states(f,src)
                with weighted_cross_attention(teacher,mass):
                    seq["probe_"+k]=teacher.generate(encoder_outputs=BaseModelOutput(last_hidden_state=field),
                                                     attention_mask=mask,**GEN).cpu()
            for idx,i in enumerate(ids):
                predictions[i].update({k:as_answer(y[idx],tok) for k,y in seq.items()})
    return predictions


def main():
    configure_cuda_cli()
    evaluation=read(PUB/"evaluation.json")
    prior=RUN/"guided.json"
    if sha(prior)!=evaluation["guided"]: raise ValueError("Prior answers")
    rows=requests()
    raw,votes=RUN/"readout-answers.json",RUN/"readout-votes.json"
    binding=dict(prior=sha(prior),program=sha(Path(__file__)),protocol=sha(ROOT/"references/field_transfer_readout_comparison.md"))
    with gpu_owner():
        torch.set_num_threads(2)
        if not raw.exists(): dump(raw,dict(binding=binding,answers=probes(rows,read(prior)["answers"])))
        d=read(raw)
        if d["binding"]!=binding: raise ValueError("Changed readout provenance")
        torch.cuda.empty_cache()
        if not votes.exists(): dump(votes,dict(raw=sha(raw),votes=blind_votes(rows,d["answers"])))
        v=read(votes)
        if v["raw"]!=sha(raw): raise ValueError("Unmatched readout grades")
    s=summarize(rows,d["answers"],v["votes"])
    s["hybrid_over_independent"]=work_interval(s["works"],"probe_prediction","prediction")
    s["hybrid_treatment"]=work_interval(s["works"],"probe_prediction","probe_regression")
    s["stops"]={m:sum(r[m]["stop"] for r in d["answers"]) for m in d["answers"][0]}
    dump(PUB/"readouts.json",dict(binding=binding,raw=sha(raw),votes=sha(votes),**s))
    print(s["total"],s["hybrid_over_independent"],flush=True)


if __name__=="__main__":
    main()
