"""Archive independent proposals before counting any exact completions."""
import fcntl
import io
from pathlib import Path
import resource
import time
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch

from ilm.visual_lm.generated_idioms import answer_encoder,propose,select
from ilm.visual_lm.idiom_ink import PicturedIdiom,visual_input
from scripts.diagnose_visual_context_gradients import load,require,sha,write_new
from scripts.fit_pictured_idiom_alignment import ROOT,DATA,OUT as ALIGNED
from scripts.fit_pictured_idioms import guard
from scripts.train_visual_extractive_reader import save_page

OUT=ROOT/'artifacts/generated_idiom_choice_v1'
PROTOCOL=ROOT/'references/pictured_idiom_generated_choice_protocol.md'


def boot(device):
    state=torch.load(ALIGNED/'weights.pt',weights_only=True,map_location='cpu')
    net=PicturedIdiom(state['landmarks']).to(device).eval()
    net.load_state_dict(state)
    aux=answer_encoder().to(device).eval()
    aux.load_state_dict(torch.load(ALIGNED/'training_only_aux.pt',map_location='cpu',weights_only=True))
    return net,aux


@torch.inference_mode()
def generate(model,encoder,x,pixels,beam,start):
    rng=torch.Generator(device='cuda').manual_seed(421)
    features=visual_input(torch.as_tensor(np.unpackbits(pixels,axis=1),device='cuda'),model.landmarks)
    with ZipFile(OUT/'proposals.zip','x',compression=ZIP_DEFLATED,compresslevel=1) as archive:
        for i in range(0,len(x),64):
            if i%2048==0:
                guard(start);print('generated',i,flush=True)
                require(time.monotonic()-start<1500,'Inference deadline')
            j=min(i+64,len(x));f=features[torch.as_tensor(x[i:j].astype(np.int32),device='cuda')]
            pix,n,stop=propose(model,f,rng=rng)
            first=torch.from_numpy(np.unpackbits(beam['packed'][i:j],axis=-1)).cuda().bool()
            allpix=torch.cat((first[:,None],pix),dim=1)
            n=torch.cat((torch.as_tensor(beam['length'][i:j],device='cuda')[:,None],n),dim=1)
            stop=torch.cat((torch.as_tensor(beam['stopped'][i:j],device='cuda')[:,None],stop),dim=1)
            winner=select(model,encoder,f,allpix,n,stop)
            data=io.BytesIO()
            np.savez(data,pixels=np.packbits(allpix.cpu().numpy(),axis=-1),length=n.cpu().numpy().astype(np.uint8),
                     stopped=stop.cpu().numpy(),choice=winner.cpu().numpy().astype(np.uint8))
            archive.writestr(f'{i:05d}.npz',data.getvalue())


def grade(dev,px):
    selected=[];length=[];ended=[];oracle=[];random=[];old=[]
    with ZipFile(OUT/'proposals.zip') as z:
        for name in sorted(z.namelist()):
            i=int(name[:-4])
            with np.load(io.BytesIO(z.read(name))) as data:
                a,n,e,c=(data[k] for k in ('pixels','length','stopped','choice'))
            b=len(c);rows=np.arange(b);truth=px[dev['target'][i:i+b]]
            hit=(a[:,:,:4]==truth[:,None]).all((2,3)) & (n==4) & e
            selected.extend(a[rows,c]);length.extend(n[rows,c]);ended.extend(e[rows,c])
            oracle.extend(hit.any(1));random.extend(hit[:,1]);old.extend(hit[:,0])
    a=np.asarray(selected);n=np.asarray(length);e=np.asarray(ended)
    np.savez_compressed(OUT/'generated.npz',packed=a,length=n,stopped=e)
    y=dev['target'];truth=px[y];hit=(a[:,:4]==truth).all((1,2)) & (n==4) & e
    prev=np.asarray(old);any_hit=np.asarray(oracle)
    other=np.roll(np.arange(len(y)),-1);swap=(a[other,:4]==truth).all((1,2)) & (n[other]==4) & e[other]
    missing=~(y[:,:,None]==dev['context'][:,None,:]).any(-1)
    _,grp=np.unique(dev['group'],return_inverse=True);sizes=np.bincount(grp)
    delta=np.bincount(grp,weights=hit.astype(int)-prev.astype(int))
    rng=np.random.default_rng(421);ci=[]
    for _ in range(2000):
        sample=rng.integers(len(sizes),size=len(sizes));ci.append(delta[sample].sum()/sizes[sample].sum())
    new_words=set(map(tuple,y[hit & missing.any(1)]));old_words=set(map(tuple,y[prev & missing.any(1)]))
    with np.load(DATA/'train.npz') as f:known={b.tobytes() for b in px[np.unique(f['target'])]}
    ink_counts={'known':0,'unknown':0,'blank':0}
    for output,l in zip(a,n):
        for cell in output[:l]:
            kind='known' if cell.tobytes() in known else 'unknown' if cell.any() else 'blank';ink_counts[kind]+=1
    examples={}
    for name,mask in [('first',np.ones(len(y),bool)),('new_four_absent',hit & ~prev & missing.all(1)),
                      ('new_absent',hit & ~prev & missing.any(1)),('first_miss',~hit)]:
        if not mask.any():continue
        j=int(np.flatnonzero(mask)[0]);examples[name]=dict(index=j,tag=int(dev['tag'][j]))
        for suffix,seq in [('before',px[dev['context'][j,:24]]),('after',px[dev['context'][j,24:]]),
                           ('reference',truth[j]),('answer',a[j,:n[j]])]:
            if len(seq):save_page(OUT/(name+'_'+suffix+'.png'),seq)
    return dict(examples=len(y),complete_exact=int(hit.sum()),swapped_exact=int(swap.sum()),beam_exact=int(prev.sum()),
                first_random_exact=int(sum(random)),oracle_proposal_exact=int(any_hit.sum()),new_exact=int((hit & ~prev).sum()),
                lost_exact=int((prev & ~hit).sum()),unique_correct=len(set(map(tuple,y[hit]))),
                distinct_correct_with_absent=len(new_words),new_distinct_with_absent=len(new_words-old_words),
                old_distinct_lost=len(old_words-new_words),exact_with_all_four_absent=int((hit & missing.all(1)).sum()),
                stopped=int(e.sum()),lengths={str(l):int((n==l).sum()) for l in np.unique(n)},ink_counts=ink_counts,
                improvement_95=np.quantile(ci,[.025,.975]).tolist(),examples_by_type=examples,
                passed=bool(len(new_words)-len(old_words)>=30 and np.quantile(ci,.025)>0))


def main():
    start=time.monotonic();guard(start);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    report=load(ALIGNED/'result.json')
    for f in ('weights.pt','training_only_aux.pt','generated.npz'):
        require(sha(ALIGNED/f)==report['source_hashes'][str((ALIGNED/f).relative_to(ROOT))],'Changed checkpoint or beam control')
    OUT.mkdir(exist_ok=False)
    net,aux=boot('cuda')
    px=np.load(DATA/'pixels.npy')
    with np.load(DATA/'validation.npz') as f:dev={k:f[k] for k in f.files}
    with np.load(ALIGNED/'generated.npz') as f:beam={k:f[k] for k in f.files}
    g=time.monotonic();generate(net,aux,dev['context'],px,beam,start);elapsed=time.monotonic()-g
    result=grade(dev,px)
    result.update(generation_seconds=elapsed,seconds=time.monotonic()-start,peak_vram=torch.cuda.max_memory_allocated(),
                  host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_hashes={str(p.relative_to(ROOT)):sha(p)
                  for p in [Path(__file__).resolve(),ROOT/'ilm/visual_lm/generated_idioms.py',PROTOCOL,ALIGNED/'result.json',
                            OUT/'proposals.zip',OUT/'generated.npz']})
    guard(start);write_new(OUT/'result.json',result);print(result,flush=True)

if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
