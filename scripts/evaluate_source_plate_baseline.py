"""Measure source-grounded historical requests through the existing physical cloze model."""
import fcntl
import time

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from scripts.fit_pictured_idioms import guard
from scripts.diagnose_visual_context_gradients import load,require,sha,write_new
from scripts.prepare_source_plate_questions import OUT as DATA, ROOT
from scripts.train_visual_extractive_reader import save_page

OUT=ROOT/'artifacts/source_plate_baseline_v1'


@torch.inference_mode()
def main():
    start=time.monotonic()
    guard(start)
    OUT.mkdir()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    raw=np.load(DATA/'pixels.npy')
    with np.load(DATA/'development.npz') as f:
        x,y,lang,length,card = f['context'],f['target'],f['language'],f['length'],f['card']
    root=ROOT/'models/pictured_idiom_zh'
    state=load_file(str(root/'reader_writer.safetensors'))
    state.update(load_file(str(root/'distilled_writer.safetensors')))
    model=PicturedIdiom(state['landmarks']).cuda().eval()
    model.load_state_dict(state)
    features=visual_input(torch.as_tensor(np.unpackbits(raw,axis=1),device='cuda').bool(),model.landmarks)
    arms={}
    for mode in ('intact','blank_question','blank_plate'):
        inputs=x.copy()
        if mode=='blank_question':
            inputs[:,:24]=0
        if mode=='blank_plate':
            inputs[:,24:]=0
        streams,sizes,ended=[],[],[]
        for i in range(0,len(x),128):
            guard(start)
            cells,n,stop=model.generate(features[torch.as_tensor(inputs[i:i+128].astype(np.int32),device='cuda')])
            cells[np.arange(8)[None,:]>=n[:,None]]=0
            streams.append(cells);sizes.extend(n);ended.extend(stop)
        ink=np.concatenate(streams);n=np.asarray(sizes);stop=np.asarray(ended)
        archive=OUT/(mode+'.npz')
        np.savez_compressed(archive,packed=ink,length=n,stopped=stop)
        same=(ink[:,:2]==raw[y]).all(2)
        exact=stop & (n==length) & same[:,0] & ((length==1)|same[:,1])
        sub={f'{label}_{k}':int(exact[(lang==v)&(length==k)].sum()) for v,label in enumerate(['zh','en']) for k in [1,2]}
        # Each block is single stage A, single stage B, A/B and B/A.
        shaped=exact.reshape(-1,4)
        emitted=ink[np.arange(8)[None,:]<n[:,None]]
        refs={r.tobytes() for r in raw[np.unique(y[length==2])]}
        arms[mode]=dict(exact=int(exact.sum()),correct_length=int((stop & (n==length)).sum()),stopped=int(stop.sum()),
                       lengths={str(k):int((n==k).sum()) for k in np.unique(n)},by_language_length=sub,
                       order_pairs_both=int((shaped[:,2]&shaped[:,3]).sum()),
                       emitted_historic_target_shape=int(sum(r.tobytes() in refs for r in emitted)),emitted=len(emitted),
                       archive_sha256=sha(archive))
    demo=load(DATA/'yan/source.json')
    with np.load(DATA/'train.npz') as f:
        ids=f['context'][demo['index']]
    ink,n,stop=model.generate(features[torch.as_tensor(ids.astype(np.int32),device='cuda')][None])
    if n[0]:
        save_page(OUT/'yan_answer.png',ink[0,:n[0]])
    result=dict(examples=len(x),cards=len(np.unique(card)),group_denominator=len(x)//4,
                order_pair_denominator=len(x)//4,arms=arms,seconds=time.monotonic()-start,
                yan=dict(stage=demo['stages'],card=demo['card'],cells=int(n[0]),stopped=bool(stop[0])),
                source_report_sha256=sha(DATA/'report.json'),weights_sha256=sha(root/'distilled_writer.safetensors'))
    write_new(OUT/'result.json',result)
    print(result,flush=True)

if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
