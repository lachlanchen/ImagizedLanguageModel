"""Infer bidirectional source positions for the frozen parallel image corpus."""
from itertools import product
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.optical_lexicon import expand_table
from ilm.visual_lm.ordered_alignment import movement
from scripts.prepare_covered_phrases import WORK,PUB,base,prior
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def viterbi(emission,transition,lengths,steps):
    """Trace the most probable complete alignment. Padded columns never acquire mass."""
    batch,width=transition.shape[:2]
    device=emission.device
    index=torch.arange(batch,device=device)
    valid=torch.arange(width,device=device)[None,:]<lengths[:,None]
    state=(emission[:,0]/lengths[:,None]).log().masked_fill(~valid,-torch.inf)
    log_move=transition.log()
    back=[]
    for j in range(1,emission.size(1)):
        best,prev=(state.unsqueeze(2)+log_move).max(1)
        proposed=(best+emission[:,j].log()).masked_fill(~valid,-torch.inf)
        state=torch.where((j<steps)[:,None],proposed,state)
        back.append(prev.byte())
    pos=state.argmax(1)
    out=torch.full((batch,emission.size(1)),-1,device=device,dtype=torch.int8)
    for j in reversed(range(emission.size(1))):
        live=j<steps
        out[:,j]=torch.where(live,pos,-1).to(torch.int8)
        if j:
            pos=torch.where(live,back[j-1][index,pos].long(),pos)
    return out


def check():
    e=np.array([[[.2,.4,.3],[.7,.2,.5],[.1,.8,.3]],[[.5,.2,.1],[.6,.9,.1],[.9,.2,.3]]])
    sizes=[3,2]
    trans=np.zeros((2,3,3))
    for i,n in enumerate(sizes):
        trans[i,:n,:n]=movement(n,[.2,.1,.1,.5,.1])
    forward=viterbi(torch.tensor(e,device="cuda"),torch.tensor(trans,device="cuda"),
                    torch.tensor(sizes,device="cuda"),torch.tensor(sizes,device="cuda")).cpu().numpy()
    for i,n in enumerate(sizes):
        def score(path):
            return np.prod([e[i,j,k]*(trans[i,path[j-1],k] if j else 1/n) for j,k in enumerate(path)])
        expected=max(product(range(n),repeat=n),key=score)
        assert list(expected)==list(forward[i,:n]) and all(forward[i,n:]==-1)


def align(t,seq,corpus):
    lang=("en","zh")[t]
    fit=base.read(prior.MODEL/lang/"fit.json")
    outfile=WORK/f"links-{t}.npy"
    weight_file=base.MODEL/lang/"probabilities.npz"
    assert base.sha(weight_file)==fit["inputs"]["emissions"]
    receipt=PUB/f"links-{t}.json"
    if receipt.exists():
        r=base.read(receipt)
        assert r["hash"]==base.sha(outfile) and r["fit"]==base.sha(prior.MODEL/lang/"fit.json")
        return
    with np.load(weight_file) as z:
        table=torch.from_numpy(expand_table(z["ids"],z["mass"],z["tail"],z["prior"])).cuda()
    lengths=np.stack([np.count_nonzero(seq[d],axis=1) for d in (1-t,t)],1)
    src,tgt=[torch.as_tensor(np.array(seq[d],dtype=np.int32),device="cuda") for d in (1-t,t)]
    sizes=torch.tensor(lengths,device="cuda")
    bucket=lengths[:,0]//8*32+lengths[:,1]//8
    order=np.argsort(bucket,kind="stable")
    result=np.full(seq[t].shape,-1,np.int8)
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for group in np.split(order,np.flatnonzero(np.diff(bucket[order]))+1):
        width,steps=map(int,lengths[group].max(0))
        kernels=np.zeros((width+1,width,width),np.float32)
        for n in np.unique(lengths[group,0]):
            kernels[n,:n,:n]=movement(int(n),fit["mixing"])
        kernels=torch.from_numpy(kernels).cuda()
        for j in range(0,len(group),512):
            ids=group[j:j+512]
            length,nsteps=sizes[ids].unbind(1)
            a=src[ids,:width].long()
            b=tgt[ids,:steps].clamp(min=1).long()
            local=table[a[:,None,:],b[:,:,None]]*length[:,None,None]
            null=table[0,b]
            emission=(local+null[:,:,None])/(length[:,None,None]+1)
            path=viterbi(emission,kernels[length],length,nsteps)
            attached=local.gather(2,path.long().clamp_min(0).unsqueeze(2)).squeeze(2)
            path=path.masked_fill(attached<=null,-1)
            result[ids,:steps]=path.cpu().numpy()
        if monotonic()-start>3600 or torch.cuda.max_memory_allocated()>12*2**30:
            raise RuntimeError("Registered alignment resource ceiling reached")
    np.save(outfile,result)
    base.dump(receipt,dict(pairs=corpus["pairs"],attached=int((result>=0).sum()),hash=base.sha(outfile),
              seconds=monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
              fit=base.sha(prior.MODEL/lang/"fit.json"),source=base.sha(Path(__file__))))
    print(lang,base.read(receipt),flush=True)


def main():
    configure_cuda_cli()
    corpus=base.read(PUB/"corpus.json")
    paths=[prior.WORK/f"strips-{t}.npy" for t in (0,1)]
    assert [base.sha(p) for p in paths]==corpus["strips"]
    seq=[np.load(p,mmap_mode="r") for p in paths]
    with gpu_owner():
        check()
        for t in (0,1):
            align(t,seq,corpus)


if __name__=="__main__":
    main()
