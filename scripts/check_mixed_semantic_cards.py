"""Apply the frozen release to all distinct development cards with alternating scripts."""
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from scripts.train_semantic_listener import load_listener
from scripts.evaluate_semantic_listener import metrics,realization_patterns
from scripts.train_semantic_page import WORK,PUB,MODEL,read,archive,load,palette,gpu_owner,dump,snapshot,consumed


def run():
    meta=read(PUB/"corpus.json");schema=read(WORK/"schema.json")
    corpus=archive(WORK/"dev.npz",meta["archives"]["dev"])
    _,index=np.unique(corpus["slots"],axis=0,return_index=True)
    rows=corpus["slots"][np.sort(index)]
    meanings=schema["concepts"]
    gold=[{tuple(meanings[i]) for i in row if i} for row in rows]
    bank=archive(WORK/"bank.npz",meta["bank"])
    chars=read(MODEL/"readback.json")
    patterns=realization_patterns()
    with gpu_owner(),torch.inference_mode():
        reader=load("ink","cuda");listener=load_listener("cuda");out=palette("cuda")
        ink=torch.as_tensor(observations(bank,"ink"),device="cuda")
        outputs={"beam":[],"readback":[]}
        start=monotonic();energy=snapshot()
        for j in range(0,len(rows),32):
            facts=rows[j:j+32]
            language=(np.arange(facts.shape[1])%2==0).astype(int)
            # Run on the physically observed cell stream, with a mix inside each card.
            surface=ink[bank["surface"][language[None,:],facts]]
            mem=reader.encode_rows(surface.flatten(0,1)).reshape(len(facts),facts.shape[1],-1)
            mask=torch.as_tensor(facts!=0,device="cuda")
            for name,(ids,ns) in zip(outputs,beam_write(reader,mem,mask,out,listener)):
                outputs[name].extend("".join(chars[i] for i in r[:n]) for r,n in zip(ids.cpu().tolist(),ns.tolist()))
        cost=dict(seconds=monotonic()-start,energy=consumed(energy,snapshot()))
    dump(WORK/"mixed-dev-lines.json",outputs)
    dump(PUB/"mixed-dev.json",dict(scores={k:metrics(lines,gold,patterns) for k,lines in outputs.items()},cost=cost,
                                   partition="official development, previously unused for fitting or model choice",
                                   schema=meta["schema"],weights=read(PUB/"ink-fit.json")["weights"]))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    run()
