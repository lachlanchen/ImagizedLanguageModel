"""Train matched physical and hidden source-ink writers on professional news supervision."""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file

from scripts.reserve_open_news_external import ROOT,PUB,WORK,RULE,PRIOR,ORIGINAL,base
from scripts.train_visual_training_breadth import restore as parent,corpus,MODEL as START
from scripts.train_glyph_draft_editor import atomic_save
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.train_ordered_translation import pack
from ilm.visual_lm.open_source_cells import OpenSourceCells
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.full_image_roles import pictured_input

MODEL=ROOT/'models/open_news_writer'
SEED=753901


def admitted():
    info=base.read(PUB/'selection.json')
    assert info['passed'] and min(info['counts'])>=30 and info['panel']==base.sha(WORK/'test.json')
    assert info['parent']==base.sha(PRIOR/'selection.json') and info['protocol']==base.sha(RULE)
    original=base.read(PRIOR/'selection.json')
    assert original['rows']==base.sha(ORIGINAL/'sentences.json') and original['protocol']==base.sha(ROOT/'references/open_news_training_protocol.md')
    return original


def bindings():
    files=[Path(__file__),RULE,PUB/'selection.json',PRIOR/'selection.json',START/'broad/fit.json',
           ROOT/'ilm/visual_lm/open_source_cells.py',ROOT/'ilm/visual_lm/glyph_draft_editor.py',
           ROOT/'scripts/reserve_open_news_external.py',ROOT/'scripts/prepare_open_news_training.py']
    return {str(p.relative_to(ROOT)):base.sha(p) for p in files}


def restore(arm,device):
    net=parent('broad',device)
    if arm=='parent':return net
    if arm not in ('hidden','ink'):raise ValueError(arm)
    p=MODEL/arm
    receipt=base.read(p/'fit.json')
    assert receipt['weights']==base.sha(p/'weights.safetensors') and receipt['bindings']==bindings()
    net.model.load_state_dict(load_file(str(p/'weights.safetensors')))
    net.model.eval()
    return net


def prepare(writer):
    rows=[r for r in base.read(ORIGINAL/'sentences.json') if not r['held']]
    face=base.Ink(writer.palette)
    encoded=[(face.encode(r['source']),face.encode(r['target'])) for r in rows]
    assert not face.bad
    palette=face.array()
    reader=OpenSourceCells(writer,0)
    extras={};dataset=[]
    for src,tgt in encoded:
        physical=palette[src]
        ids=[]
        for row in physical:
            key=row.tobytes()
            idx=reader.known.get(key)
            if idx is None:
                idx=extras.setdefault(key,len(writer.features)+len(extras))
            ids.append(idx)
        dataset.append((ids,tgt+[1],reader.phrase_cells(physical)))
    tiles=np.frombuffer(b''.join(extras),np.uint8).reshape(-1,128)
    extra=torch.from_numpy(observations({'tiles':tiles},'ink'))
    return dataset,extra,reader.mean


def fit(arm,dataset,extra,mean,replay):
    folder=MODEL/arm
    if (folder/'fit.json').exists():
        restore(arm,'cpu')
        return
    writer=parent('broad','cuda')
    net=writer.model.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0001,fused=True)
    historical,_,drafts,_=replay
    more=extra if arm=='ink' else mean.expand_as(extra)
    features=torch.cat((writer.features,more.cuda()))
    rng=np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    log=[]
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for ep in range(8):
        for group in optimizer.param_groups:group['lr']=.00010 if ep<4 else .00005
        order=rng.permutation(len(dataset))
        total=0;loss_sum=0.;steps=0;hashes=sha256()
        for j in range(0,len(order),16):
            ix=order[j:j+16]
            old=rng.integers(len(historical),size=16)
            hashes.update(ix.astype('<i4').tobytes());hashes.update(old.astype('<i4').tobytes())
            source=[dataset[i][0] for i in ix]
            target=[dataset[i][1] for i in ix]
            hints=[dataset[i][2] for i in ix]
            previous=historical[old].cpu().numpy()
            old_hints=drafts[0][old].cpu().numpy()
            for row,h in zip(previous,old_hints,strict=True):
                source.append(row[1][row[1]!=0].tolist())
                target.append(row[0][row[0]!=0].tolist()+[1])
                hints.append(h[h!=0].tolist() or [0])
            s=pack(source,'cuda');y=pack(target,'cuda');d=pack(hints,'cuda')
            with torch.no_grad():
                field=(255*writer.marker(features,s).sigmoid()*s.ne(0).unsqueeze(-1)).round().byte()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                law=net(features,pictured_input(s,field),y,writer.inv[0],0,d,writer.local[0])
            loss=F.nll_loss(law.transpose(1,2),writer.local[0][y],ignore_index=-100,reduction='sum')
            n=sum(map(len,target))
            (loss/n).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            total+=n;loss_sum+=loss.detach().item();steps+=1
            if monotonic()-start>1800 or torch.cuda.max_memory_allocated()>16*2**30:
                raise RuntimeError('Registered fit budget exceeded')
        log.append(dict(epoch=ep+1,steps=steps,positions=total,nll=loss_sum/total,schedule=hashes.hexdigest()))
        print(arm,ep+1,round(loss_sum/total,5),flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'weights.safetensors'
    atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},f),path)
    base.dump(folder/'fit.json',dict(arm=arm,weights=base.sha(path),bindings=bindings(),epochs=log,
              parameters=sum(p.numel() for p in net.parameters()),seconds=monotonic()-start,
              peak_bytes=torch.cuda.max_memory_allocated(),extra_shapes=len(extra)))


def main():
    admitted();configure_cuda_cli()
    with gpu_owner():
        dataset,extra,mean=prepare(parent('broad','cpu'))
        replay=corpus()
        for arm in ('hidden','ink'):fit(arm,dataset,extra,mean,replay)
        a,b=(base.read(MODEL/m/'fit.json')['epochs'] for m in ('hidden','ink'))
        assert all(x[k]==y[k] for x,y in zip(a,b,strict=True) for k in ('schedule','positions','steps'))


if __name__=='__main__':
    main()
