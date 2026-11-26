"""Fit matched pictured dictionary readers with repeated or varied instruction wording."""
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file, load_file
import torch
from torch.nn import functional as F
from ilm.visual_lm.origin_instruction import TRAIN, question
from scripts.train_origin_value_reader import OUT, ROOT, RECEIPT, rows, layout, dump, read, sha
from scripts.prepare_region_writer import Ink
from scripts.write_origin_values import model
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner, atomic_save

MODEL=ROOT/"models/origin_instructions"
PUB=ROOT/"references/evidence/origin_instructions_v1"
PROTOCOL=ROOT/"references/origin_instruction_transfer_protocol.md"


def load(arm,device):
    net=model("matched",device)
    dest=MODEL/arm
    path=dest/"reader.safetensors"
    assert sha(path)==read(dest/"fit.json")["weights"]
    net.load_state_dict(load_file(str(path),device=device))
    return net.eval()


def train():
    ds=rows("train")
    tiles=OUT/"tiles.npy"
    assert sha(tiles)==read(RECEIPT)["files"]["tiles.npy"]
    ink=Ink(np.load(tiles))
    original,labels,targets=layout(ds,"cuda")
    variants=[]
    for epoch in range(8):
        questions=[]
        for i,r in enumerate(ds):
            v=(epoch+i)%8
            s=r["supervision"]
            q=ink.encode(question(r,TRAIN[s["english"]][v-1])[0]) if v else r["question"]
            questions.append(torch.tensor(q))
        variants.append(torch.nn.utils.rnn.pad_sequence(questions,batch_first=True).cuda())
    assert not ink.bad,ink.bad
    bank=torch.from_numpy(np.unpackbits(ink.array(),axis=1).reshape(-1,32,32)).cuda()
    paths=(Path(__file__),PROTOCOL,ROOT/"ilm/visual_lm/origin_instruction.py",ROOT/"models/origin_value_reader/matched/fit.json",RECEIPT)
    binding={str(p.relative_to(ROOT)):sha(p) for p in paths}
    PUB.mkdir(exist_ok=True)
    for arm in ("repeat","varied"):
        dest=MODEL/arm
        dest.mkdir(exist_ok=True,parents=True)
        path=dest/"reader.safetensors"
        if (dest/"fit.json").exists():
            saved=read(dest/"fit.json")
            assert saved["binding"]==binding and sha(path)==saved["weights"]
            continue
        torch.manual_seed(28107)
        net=model("matched","cuda").train()
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        start=monotonic()
        torch.cuda.reset_peak_memory_stats()
        history=[]
        for epoch in range(8):
            q=original if arm=="repeat" else variants[epoch]
            order=np.random.default_rng(91048+epoch).permutation(len(ds))
            total=0.
            for j in range(0,len(ds),64):
                ix=order[j:j+64]
                opt.zero_grad(set_to_none=True)
                logits=net(bank,q[ix],labels[ix],targets[ix])
                loss=F.cross_entropy(logits.flatten(0,1),targets[ix].flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                total+=float(loss.detach())*len(ix)
            history.append(total/len(ds))
        seconds=monotonic()-start
        peak=torch.cuda.max_memory_allocated()
        assert seconds<1800 and peak<6*2**30
        atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},f),path)
        dump(dest/"fit.json",dict(weights=sha(path),binding=binding,history=history,exposures=len(ds)*8,
             updates=8*((len(ds)+63)//64),seconds=seconds,peak=peak,params=sum(p.numel() for p in net.parameters())))
        print(arm,history,flush=True)


if __name__=="__main__":
    configure_cuda_cli()
    with gpu_owner():
        torch.set_num_threads(2)
        train()
