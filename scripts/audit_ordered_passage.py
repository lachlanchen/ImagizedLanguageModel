"""Audit the ordered decoder on archived pixels and package an independently replayable pair."""
from pathlib import Path
import shutil
import numpy as np
import torch
from safetensors.torch import load_model
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.audit_passage_instructions import EVIDENCE
from scripts.compare_passage_instructions import OUT,TEST
from scripts.infer_pictured_entry import page,bitmap
from scripts import test_official_passage_composition as ext
from scripts.train_paired_passage import ROOT,WORK
from scripts.prepare_pictured_entry_answers import load,write_json,sha

BUNDLE=ROOT/"models/pictured_passage_zh"


def main():
    ext.WRAPPERS=TEST
    ps,qs,aliases,pairs,groups,prefixes,join,tiles,_=ext.prepare()
    folder=OUT/"ordered_decoder"
    result=load(folder/"result.json")
    q,n=len(qs),len(pairs)
    file=folder/"ordered.npz"
    assert sha(file)==result["official"]["archive_sha256"]
    with np.load(file) as z:
        pixels,off,stops=z["pixels"],z["offsets"],z["stopped"]
    def output(i):
        return pixels[off[i]:off[i+1]].tobytes()
    assert len(stops)==q+n*len(TEST) and int(off[-1])==len(pixels)
    assert sum(bool(stops[i]) and output(i) in ans for i,ans in enumerate(aliases))==result["official"]["single_exact"]
    sep=tiles[join].tobytes()
    series=[]
    for j,wrapper in enumerate(TEST):
        base=q+n*j
        ok=np.array([bool(stops[base+i]) and any(output(base+i)==x+sep+y for x in aliases[a] for y in aliases[b])
                     for i,(_,a,b) in enumerate(pairs)])
        both=ok.reshape(-1,2).all(1)
        assert (int(ok.sum()),int(both.sum()))==(result["official"][wrapper]["exact"],result["official"][wrapper]["both_orders"])
        series.append(round(100*int(both.sum())/(n//2),4))
        if j==1:
            example=2*int(np.flatnonzero(both)[0])
            failure=int(np.flatnonzero(~ok)[0])
    BUNDLE.mkdir(exist_ok=False)
    weights=BUNDLE/"varied_planner.safetensors"
    shutil.copyfile(OUT/"varied.safetensors",weights)
    shutil.copyfile(WORK/"passage_composer.safetensors",ROOT/"models/pictured_idiom_zh/passage_composer.safetensors")
    assert sha(weights)==load(OUT/"comparison.json")["varied"]["weights_sha256"]
    net=OrderedPassageComposer().eval()
    load_model(net,str(weights))
    torch.set_num_threads(2)
    receipts={}
    for label,i in (("first",example),("swapped",example+1),("missed",failure)):
        g,a,b=pairs[i]
        dest=BUNDLE/"examples"/label
        dest.mkdir(parents=True)
        source=tiles[ps[g]].tobytes()
        prompt=tiles[np.concatenate((prefixes[1],qs[a],join,qs[b]))].tobytes()
        for key,ink in (("passage",source),("question",prompt)):
            page(ink).save(dest/(key+".png"))
        arguments=[np.frombuffer(bitmap(dest/(key+".png"),len(ink)//128),np.uint8).reshape(-1,128)
                   for key,ink in (("passage",source),("question",prompt))]
        answer,stop=net.answer(*arguments)
        k=q+n+i
        assert answer==output(k) and stop==stops[k]
        page(answer).save(dest/"answer.png")
        ref=min(x+sep+y for x in aliases[a] for y in aliases[b])
        page(ref).save(dest/"one_accepted_reference.png")
        receipt=dict(row=i,passage_cells=len(source)//128,question_cells=len(prompt)//128,output_cells=len(answer)//128,
                     stopped=bool(stop),exact=any(answer==x+sep+y for x in aliases[a] for y in aliases[b]) and stop,
                     answer_png_sha256=sha(dest/"answer.png"))
        write_json(dest/"example.json",receipt)
        receipts[label]=receipt
    f=load(EVIDENCE/"figure.json")
    f["table"].append(["Ordered decoder, varied instructions",f'{result["internal_pairs"]} / 6,086',
                       f'{result["internal_both"]} / 3,043',f'{result["single"]} / 1,788'])
    f["series"].append(series)
    f["externalNote"]="On the shorter Chinese instruction, the same weights rose from 0 to 505 of 4,800 both-correct pairs when the decoder enforced contiguous questions."
    write_json(EVIDENCE/"figure.json",f)
    write_json(EVIDENCE/"ordered_decoder.json",result)
    write_json(EVIDENCE/"ordered_audit.json",dict(official_images_regraded=q+n*len(TEST),cpu_png_examples=receipts,
                                                  deployed_sha256=sha(weights)))
    print(receipts,flush=True)


if __name__=="__main__":
    main()
