"""Matched image-only fits for local heads and their overt relative antecedents."""
import math
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from ilm.visual_lm.natural_roles import tiles
from ilm.visual_lm.role_localization import RoleLocator
from scripts.prepare_visual_referents import MODEL, PUB, ROOT, WORK, dump, read, sha
from scripts.train_field_to_glyph import atomic_save, configure_cuda_cli, gpu_owner
from scripts.train_natural_pictured_roles import arrays

PARENT = ROOT/"models/visual_role_locator/context"
SEED = 602173


def fit(arm, rows, pixels, flags, palette):
    folder = MODEL/arm
    folder.mkdir(exist_ok=True,parents=True)
    path = folder/"weights.safetensors"
    binding = {str(p.relative_to(ROOT)):sha(p) for p in (
        PARENT/"weights.safetensors", PUB/"corpus.json", Path(__file__),
        ROOT/"ilm/visual_lm/role_localization.py", ROOT/"references/visual_referent_learning_protocol.md")}
    if (folder/"fit.json").exists():
        rec = read(folder/"fit.json")
        assert rec["binding"] == binding and rec["weights"] == sha(path)
        return
    assert sha(PARENT/"weights.safetensors") == read(PARENT/"fit.json")["weights"]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = RoleLocator().cuda()
    model.load_state_dict(load_file(str(PARENT/"weights.safetensors")))
    opt = torch.optim.AdamW(model.parameters(),lr=.00025,fused=True)
    labels = torch.tensor([[k for a,b in r[arm] for k in (a,b-1)] for r in rows],device="cuda")
    groups = [[i for i,r in enumerate(rows) if r["linked"]]]
    groups.extend([i for i,r in enumerate(rows) if not r["linked"] and r["voice"]==v]
                  for v in ("active","passive"))
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    history = []
    loss_total = 0.
    for step in range(1,1601):
        idx = rng.permutation(np.concatenate([rng.choice(g,n) for g,n in zip(groups,(24,32,8),strict=True)]))
        idx = torch.from_numpy(idx).cuda()
        ids = pixels[idx].long()
        opt.param_groups[0]["lr"] = .00025*min(1,step/60)*(.15+.85*(1+math.cos(math.pi*step/1600))/2)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            scores = model.read(model.eye(palette)[ids],flags[idx],ids.ne(0))
            loss = F.cross_entropy(scores.flatten(0,1),labels[idx].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        loss_total += loss.item()
        if step%200==0:
            history.append(loss_total/200)
            loss_total = 0.
            if monotonic()-start > 900 or torch.cuda.max_memory_allocated()>12*2**30:
                raise RuntimeError("Resource ceiling")
    atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},f),path)
    rec = dict(binding=binding,weights=sha(path),updates=1600,examples=1600*64,seed=SEED,
               losses=history,seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
               trainables=sum(p.numel() for p in model.parameters()))
    dump(folder/"fit.json",rec)
    dump(PUB/f"fit-{arm}.json",rec)
    print(arm,rec["seconds"],rec["losses"],flush=True)


def main():
    configure_cuda_cli()
    with gpu_owner():
        corpus = read(PUB/"corpus.json")
        for name,digest in (("train.json",corpus["splits"]["train"]),("codes.json",corpus["codes"]),
                            ("palette.npy",corpus["palette"])):
            assert sha(WORK/name)==digest
        rows = read(WORK/"train.json")
        ids,flags = arrays(rows,read(WORK/"codes.json"))
        x = torch.from_numpy(ids).cuda()
        marks = torch.from_numpy(flags==1).cuda()
        palette = tiles(torch.from_numpy(np.load(WORK/"palette.npy")).cuda())
        for arm in ("local","target"):
            fit(arm,rows,x,marks,palette)


if __name__=="__main__":
    main()
