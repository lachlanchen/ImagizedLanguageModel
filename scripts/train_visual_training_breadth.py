"""Match glyph exposures while changing the diversity of bilingual supervision."""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from ilm.visual_lm.glyph_draft_editor import GlyphDraftEditor
from ilm.visual_lm.full_image_roles import pictured_input
from ilm.visual_lm.covered_phrases import character_fscore
from scripts.prepare_visual_training_breadth import ROOT, WORK, PUB, MODEL, RULE, SEED, COHORT, PASSES, base, previous
from scripts.train_glyph_draft_editor import batches, atomic_save, gpu_owner
from scripts.write_glyph_draft_editor import PictureEditor
from scripts.write_covered_phrases import CoveredWriter
from scripts.verse_gpu_runtime import configure_cuda_cli


def bindings():
    paths = [Path(__file__), RULE, PUB/'selection.json', PUB/'drafts.json', previous.MODEL/'editor/fit.json',
             ROOT/'ilm/visual_lm/glyph_draft_editor.py', ROOT/'scripts/prepare_visual_training_breadth.py']
    return {str(p.relative_to(ROOT)):base.sha(p) for p in paths}


def restore(arm, device):
    net = PictureEditor('editor', device)
    if arm != 'parent':
        if arm not in ('broad','narrow'):
            raise ValueError(f'Unknown writer {arm}')
        folder = MODEL / arm
        receipt = base.read(folder/'fit.json')
        assert receipt['weights']==base.sha(folder/'weights.safetensors') and receipt['bindings']==bindings()
        net.model.load_state_dict(load_file(str(folder/'weights.safetensors')))
    return net


def corpus():
    sel = base.read(PUB/'selection.json')
    assert sel['protocol']==base.sha(RULE) and all(base.sha(WORK/k)==v for k,v in sel['files'].items())
    manifest = base.read(PUB/'drafts.json')
    assert manifest['selection']==base.sha(PUB/'selection.json')
    pairs = np.load(WORK/'pairs.npy')
    lengths = np.count_nonzero(pairs,axis=-1)
    real = COHORT*PASSES
    assert len(pairs)==real+sel['roles'] and np.all(lengths[:real].reshape(PASSES,COHORT,2)==lengths[:COHORT])
    drafts, widths = [], []
    for t in (0,1):
        parts=[]
        for name,checksum in sorted(manifest['shards'].items()):
            if not name.startswith(f'draft-{t}-'):
                continue
            assert base.sha(WORK/name)==checksum and int(name[8:14])==len(parts)
            with np.load(WORK/name) as data:
                ids, offsets = data['ids'], data['offsets']
                parts.extend(ids[i:j] for i,j in zip(offsets[:-1],offsets[1:],strict=True))
        assert len(parts)==len(pairs)
        size=np.asarray(list(map(len,parts)))
        packed=np.zeros((len(size), max(size)),dtype=np.int32)
        for i,row in enumerate(parts):
            packed[i,:len(row)] = row
        drafts.append(torch.as_tensor(packed,device='cuda'))
        widths.append(size)
    return torch.as_tensor(pairs.astype(np.int32),device='cuda'), lengths, drafts, np.stack(widths,axis=1)


def development_inputs(net):
    records=base.read(base.WORK/'dev.json')
    result=[]
    for t in (0,1):
        engine=CoveredWriter(t)
        sources=[net.palette[r['cells'][1-t]] for r in records]
        ds=[engine.translate(s)[0] for s in sources]
        result.append((sources,ds,[r['cells'][t] for r in records]))
    return result


def development(net, inputs):
    net.model.eval()
    metrics=[]
    for t,(sources,drafts,targets) in enumerate(inputs):
        outputs=[]
        for i in range(0,len(sources),24):
            outputs.extend(net.translate(sources[i:i+24],t,drafts[i:i+24]))
        scores=[character_fscore(ref,net.addresses(out)) if stop else 0 for (out,stop),ref in zip(outputs,targets,strict=True)]
        metrics.append(dict(mean_glyph_f2=float(np.mean(scores)),stopped=sum(bool(s) for _,s in outputs),n=len(scores)))
    return metrics


def fit(arm, data, fields, validation):
    dest=MODEL/arm
    if (dest/'fit.json').exists():
        restore(arm,'cpu')
        return
    net=restore('parent','cuda')
    model=net.model
    pairs,lengths,drafts,widths=data
    fixed=np.arange(PASSES*COHORT,len(pairs))
    layout=np.concatenate((np.arange(COHORT),fixed))
    torch.manual_seed(SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    history=[]
    for ep,lr in enumerate((.00013,.00011,.00009,.00007,.00005,.00003)):
        model.train()
        optimizer.param_groups[0]['lr']=lr
        offset=ep*COHORT if arm=='broad' else 0
        selected=np.concatenate((np.arange(offset,offset+COHORT),fixed))
        assert np.array_equal(lengths[layout],lengths[selected])
        positions=updates=draft_cells=0
        total=0.
        schedule=sha256()
        for slot in batches(lengths[layout],ep):
            idx=selected[slot]
            lens=lengths[idx]
            mx=lens.max(axis=0)
            n=int(lens.sum())+2*len(idx)
            schedule.update(slot.astype('<i4').tobytes()+lens.astype('<i4').tobytes())
            ids=pairs[idx].long()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                terms=[]
                for t in (0,1):
                    s=1-t
                    source=pictured_input(ids[:,s,:mx[s]],fields[idx,s,:mx[s]])
                    target=F.pad(ids[:,t,:mx[t]],(0,1)).scatter_(1,torch.as_tensor(lens[:,t,None],device='cuda').long(),1)
                    draft=drafts[t][idx,:max(1,widths[idx,t].max())].long()
                    law=model(net.features,source,target,net.inv[t],t,draft,net.local[t])
                    terms.append(F.nll_loss(law.transpose(1,2),net.local[t][target],ignore_index=-100,reduction='sum'))
                loss=sum(terms)/n
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            updates+=1
            positions+=n
            draft_cells+=int(widths[idx].sum())
            total+=loss.item()*n
            if monotonic()-start>1800 or torch.cuda.max_memory_allocated()>16*2**30:
                raise RuntimeError('Resource ceiling exceeded')
        assert positions==lengths[layout].sum()+2*len(layout)
        metrics=development(net,validation)
        history.append(dict(epoch=ep+1,positions=positions,draft_cells=draft_cells,updates=updates,schedule=schedule.hexdigest(),
                            train_nll=total/positions,development=metrics,elapsed_s=monotonic()-start))
        print(arm,history[-1],flush=True)
    dest.mkdir(parents=True,exist_ok=True)
    weights=dest/'weights.safetensors'
    atomic_save(lambda p:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},p),weights)
    base.dump(dest/'fit.json',dict(weights=base.sha(weights),bindings=bindings(),epochs=history,seconds=monotonic()-start,
             parameters=sum(p.numel() for p in model.parameters()),peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name()))


def main():
    configure_cuda_cli()
    with gpu_owner():
        data=corpus()
        net=restore('parent','cuda')
        pairs=data[0]
        fields=torch.zeros((*pairs.shape,3),dtype=torch.uint8,device='cuda')
        with torch.inference_mode():
            for t in (0,1):
                for i in range(0,len(pairs),256):
                    ids=pairs[i:i+256,t].long()
                    fields[i:i+256,t]=(255*net.marker(net.features,ids).sigmoid()*ids.ne(0).unsqueeze(-1)).round().byte()
        validation=development_inputs(net)
        base.dump(PUB/'starting-development.json',development(net,validation))
        for arm in ('narrow','broad'):
            fit(arm,data,fields,validation)
        first,second=[base.read(MODEL/m/'fit.json')['epochs'] for m in ('narrow','broad')]
        assert all(all(a[k]==b[k] for k in ('updates','positions','schedule')) for a,b in zip(first,second,strict=True))


if __name__=='__main__':
    main()
