"""Fit the inverse reader exclusively on human writing in the training split."""
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from ilm.visual_lm.semantic_listener import SemanticListener
from scripts.train_semantic_page import WORK,PUB,MODEL,read,archive,palette,sha,dump,gpu_owner,snapshot,consumed


def load_listener(device,report_dir=PUB,model_dir=MODEL):
    info=read(report_dir/"listener-fit.json")
    path=model_dir/"listener.safetensors"
    if sha(path)!=info["weights"]:
        raise ValueError("Changed listener weights")
    model=SemanticListener(info["classes"]).to(device)
    model.load_state_dict(load_file(str(path),device=str(device)))
    return model.eval()


def main():
    meta=read(PUB/"corpus.json")
    data=archive(WORK/"train.npz",meta["archives"]["train"])
    n_classes=read(PUB/"ink-fit.json")["concepts"]
    if (PUB/"listener-fit.json").exists():
        load_listener("cpu")
        return
    with gpu_owner():
        fit_listener(data,n_classes,palette("cuda"),model_dir=MODEL,report_dir=PUB,training_digest=meta["archives"]["train"])


def fit_listener(data,n_classes,glyphs,*,model_dir,report_dir,training_digest):
    torch.manual_seed(79511)
    rng=np.random.default_rng(79511)
    source=torch.as_tensor(data["slots"],device="cuda").long()
    truth=F.one_hot(source,n_classes).amax(1).float()
    model=SemanticListener(n_classes).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,fused=True)
    start=monotonic(); energy=snapshot()
    torch.cuda.reset_peak_memory_stats()
    for step in range(4000):
        selected=rng.integers(len(source),size=96)
        raw=data["y"][selected]
        raw=raw[:,:(raw>0).sum(1).max()]
        ids=torch.as_tensor(raw,device="cuda").long()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(glyphs[ids.clamp_min(0)],ids>0)
            loss=F.binary_cross_entropy_with_logits(logits[:,1:].float(),truth[selected,1:])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (step+1)%500==0:
            print(step+1,loss.item(),flush=True)
            if not torch.isfinite(loss) or monotonic()-start>600 or torch.cuda.max_memory_allocated()>5*2**30:
                raise RuntimeError("Listener exceeded registered ceiling")
    torch.cuda.synchronize()
    path=model_dir/"listener.safetensors"
    tmp=path.with_suffix(".tmp")
    save_file({k:v.cpu() for k,v in model.state_dict().items()},str(tmp));tmp.replace(path)
    dump(report_dir/"listener-fit.json",dict(weights=sha(path),seconds=monotonic()-start,energy=consumed(energy,snapshot()),
         peak=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
         classes=n_classes,updates=4000,examples=384000,training_archive=training_digest))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()
