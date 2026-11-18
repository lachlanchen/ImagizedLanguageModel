"""Run a visual phrase writer with autonomous edit decisions and no reference prefix."""
import fcntl

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.first_emission import FirstEmission
from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.phrase_editing import write_sentence
from scripts.evaluate_sentence_choice import (DIRECTORY, INTERACTION, ROOT, ConditionalInk, LocalInk, PhraseInk,
    checkpoint, load, roster, sha, write_new)
from scripts.train_first_emission import WORK
from scripts.train_local_ink import OUT


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    net=checkpoint('source')
    pen=ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    update=FirstEmission.restore(WORK/'fit/preference.safetensors',sha(INTERACTION),'cuda')
    stream=PhraseInk(*(LocalInk.load(DIRECTORY/(n+'.npz')) for n in ('onset','later','end')))
    rows,bank=roster('reserved')
    images=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    q=visual_input(images[load(OUT/'preparation.json')['question_tiles'][0][0]],net.landmarks)[None]
    folder=WORK/'sentences'
    folder.mkdir()
    results={}
    for arm in ('copy','frozen','preference'):
        parts,trace=[],[]
        for row in rows:
            pictures,end,n=(bank[row['x']],'stop',0) if arm=='copy' else write_sentence(net,images[row['x']],q,stream,pen,
                                                                    update if arm=='preference' else None)
            raw=np.asarray(pictures,dtype=np.uint8).reshape(-1,128)
            target=bank[row['y']]
            dist=Levenshtein.distance(list(map(bytes,raw)),list(map(bytes,target)))
            trace.append(dict(end=end,reference_glyphs=len(target),emitted_glyphs=len(raw),distance=dist,
                              exact=bool(end=='stop' and dist==0),insertions=n))
            parts.append(raw)
        path=folder/(arm+'.npz')
        with path.open('xb') as file:
            np.savez_compressed(file,pixels=np.concatenate(parts),offsets=np.cumsum([0]+list(map(len,parts))))
        results[arm]=dict(rows=trace,exact=sum(z['exact'] for z in trace),stopped=sum(z['end']=='stop' for z in trace),
                          distance=sum(z['distance'] for z in trace),reference=sum(z['reference_glyphs'] for z in trace),
                          distinct=len({p.tobytes() for p in parts}),archive_sha256=sha(path))
        print(arm,{k:v for k,v in results[arm].items() if k!='rows'},flush=True)
    write_new(folder/'report.json',results)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
