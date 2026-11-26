"""Train matched unrestricted optical writers with or without a fallible visual draft."""
from hashlib import sha256
from time import monotonic
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from ilm.visual_lm.glyph_draft_editor import GlyphDraftEditor
from ilm.visual_lm.full_image_roles import FullImageWriter, EventMarker, pictured_input
from scripts.prepare_glyph_draft_editor import SEED, WORK, MODEL, PUB, ROOT, RULE, base
from scripts.train_full_image_roles import loaded, batches, material, atomic_save, gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def bindings():
    paths = [Path(__file__), ROOT/'ilm/visual_lm/glyph_draft_editor.py', RULE, PUB/'selection.json', PUB/'drafts.json',
             ROOT/'models/full_image_roles/marked/fit.json', ROOT/'models/full_image_roles/marker/fit.json']
    return {str(p.relative_to(ROOT)):base.sha(p) for p in paths}


def restore(arm, device):
    folder = MODEL / arm
    rec = base.read(folder/'fit.json')
    weights = folder/'weights.safetensors'
    assert rec['weights']==base.sha(weights) and rec['bindings']==bindings()
    model = GlyphDraftEditor()
    model.load_state_dict(load_file(str(weights)))
    return model.to(device).eval()


def corpus():
    selection = base.read(PUB/'selection.json')
    assert selection['protocol']==base.sha(RULE)
    assert all(base.sha(WORK/p)==h for p,h in selection['hashes'].items())
    drafts = base.read(PUB/'drafts.json')
    assert drafts['selection']==base.sha(PUB/'selection.json')
    pairs = np.load(WORK/'pairs.npy')
    ragged = [[], []]
    for file, checksum in sorted(drafts['shards'].items()):
        assert base.sha(WORK/file)==checksum
        _, t, start = file.split('.')[0].split('-')
        rows = ragged[int(t)]
        assert len(rows)==int(start)
        with np.load(WORK/file) as z:
            ids, offsets = z['ids'], z['offsets']
            rows.extend(ids[i:j] for i,j in zip(offsets[:-1],offsets[1:],strict=True))
    widths = np.asarray([list(map(len,x)) for x in ragged]).T.copy()
    assert widths.shape==(len(pairs),2)
    arrays=[]
    for rows in ragged:
        dense=np.zeros((len(rows),max(1,max(map(len,rows)))),dtype=np.int32)
        for i,row in enumerate(rows):
            dense[i,:len(row)]=row
        arrays.append(torch.from_numpy(dense).cuda())
    return torch.from_numpy(pairs.astype(np.int32)).cuda(),np.count_nonzero(pairs,axis=-1),arrays,widths


def fit(arm, pairs, lengths, drafts, widths, fields, optical):
    out=MODEL/arm
    if (out/'fit.json').exists():
        restore(arm,'cpu')
        return
    torch.manual_seed(SEED)
    model=GlyphDraftEditor()
    missing, extra=model.load_state_dict(loaded(FullImageWriter(),'marked','cpu').state_dict(),strict=False)
    assert not extra and missing and all(name.startswith('draft_') for name in missing)
    model.cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)
    bank, inv, local=optical
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    history=[]
    for ep, lr in enumerate((.00015,.00010,.00007,.00004)):
        opt.param_groups[0]['lr']=lr
        loss_total=positions=0
        schedule=sha256()
        order=batches(lengths,ep)
        for idx in order:
            schedule.update(idx.astype('<i4').tobytes())
            lens=lengths[idx]
            ids=pairs[idx].long()
            mx=lens.max(0)
            n=int(lens.sum())+2*len(idx)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                losses=[]
                for t in (0,1):
                    s=1-t
                    src=pictured_input(ids[:,s,:mx[s]],fields[idx,s,:mx[s]])
                    tgt=F.pad(ids[:,t,:mx[t]],(0,1)).scatter_(1,torch.as_tensor(lens[:,t,None],device='cuda').long(),1)
                    draft=drafts[t][idx,:max(1,widths[idx,t].max())].long() if arm=='editor' else None
                    scores=model(bank,src,tgt,inv[t],t,draft,local[t])
                    losses.append(F.nll_loss(scores.transpose(1,2),local[t][tgt],ignore_index=-100,reduction='sum'))
                loss=sum(losses)/n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            positions+=n
            loss_total+=loss.item()*n
            if monotonic()-start>1800 or torch.cuda.max_memory_allocated()>15*2**30:
                raise RuntimeError('Registered writer resource ceiling exceeded')
        assert positions==lengths.sum()+2*len(lengths)
        history.append(dict(epoch=ep+1,positions=positions,updates=len(order),loss=loss_total/positions,schedule=schedule.hexdigest()))
        print(arm,history[-1],flush=True)
    out.mkdir(parents=True,exist_ok=True)
    weights=out/'weights.safetensors'
    atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},f),weights)
    base.dump(out/'fit.json',dict(epochs=history,seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
              gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),weights=base.sha(weights),bindings=bindings()))


def main():
    configure_cuda_cli()
    with gpu_owner():
        pairs,lengths,drafts,widths=corpus()
        marker=loaded(EventMarker(),'marker','cuda')
        optical=material('cuda')
        fields=torch.zeros((*pairs.shape,3),dtype=torch.uint8,device='cuda')
        with torch.inference_mode():
            for t in (0,1):
                assert (optical[2][t][drafts[t]][drafts[t].ne(0)]>0).all()
                for i in range(0,len(pairs),256):
                    ids=pairs[i:i+256,t].long()
                    fields[i:i+256,t]=(255*marker(optical[0],ids).sigmoid()*ids.ne(0).unsqueeze(-1)).round().byte()
        for arm in ('editor','matched'):
            fit(arm,pairs,lengths,drafts,widths,fields,optical)
        assert [e['schedule'] for e in base.read(MODEL/'editor/fit.json')['epochs']]==[e['schedule'] for e in base.read(MODEL/'matched/fit.json')['epochs']]


if __name__=='__main__':
    main()
