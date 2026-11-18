"""Generate the two frozen-weight path controls for the source-boundary intervention."""
import fcntl
import time

import numpy as np
import torch

from scripts.evaluate_sentence_choice import (DEST, DIRECTORY, INTERACTION, ROOT, SPLITS, WORK, ConditionalInk,
    SentenceBoundary, generate, load, require, resources, roster, save_streams, sha, sites, write_new)


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    started = time.monotonic()
    path = DEST/'assignment.safetensors'
    require(sha(path)==load(DEST/'fit.json')['arms']['assignment']['sha256'],'Assignment checkpoint changed')
    model = SentenceBoundary.restore(path,sha(INTERACTION),'cuda')
    ink = ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    folder = WORK/'ending_control'
    folder.mkdir()
    results = {}
    for split in SPLITS:
        rows,bank=roster(split)
        targets=[bank[rows[i]['y'][j:k]] for i,c,j,k in sites(rows)]
        results[split]={}
        for active in ('drawing','ending'):
            output,stop,_,seconds,peak=generate(rows,'source',bank,started,stream=True,interaction=ink,
                                                sentence_boundary=model,residual_path=active)
            outfile=folder/(split+'_'+active+'.npz')
            save_streams(outfile,output,stop)
            with np.load(outfile) as stored:
                parts=np.split(stored['pixels'],stored['offsets'][1:-1])
                exact=[bool(s and np.array_equal(p,t)) for p,t,s in zip(parts,targets,stored['stopped'],strict=True)]
                first=[np.array_equal(p[0],t[0]) for p,t in zip(parts,targets,strict=True)]
                result=dict(sites=len(targets),exact=sum(exact),multi=sum(e for e,t in zip(exact,targets) if len(t)>1),
                            first_correct=sum(first),stopped=int(stored['stopped'].sum()),sha256=sha(outfile),
                            seconds=seconds,peak_mib=peak)
                results[split][active]=result
            print(split,active,result,flush=True)
            resources(started)
    write_new(folder/'results.json',results)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
