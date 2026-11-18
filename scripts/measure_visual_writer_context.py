"""Ablate near or far ink in the held-out visual writer, retaining physical positions."""
import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_visual_writer_distillation import OUT, ROOT
from scripts import fit_pictured_idioms as first
from scripts.score_visual_writer_transfer import PUBLIC, matching_pairs, paired_interval

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PUBLIC, help='Directory containing the scored result')
    destination = parser.parse_args().output
    existing = load(destination/'result.json')
    require(existing['fresh']['transfer_pass'] and existing['development']['transfer_pass'], 'Unqualified distilled writer')
    fresh = OUT/'fresh'
    work = OUT/'distance'
    work.mkdir()
    start = time.monotonic()
    first.guard(start)
    state = torch.load(OUT/'distilled_targets/weights.pt',map_location='cpu',weights_only=True)
    model = PicturedIdiom(state['landmarks']).cuda().eval()
    model.load_state_dict(state)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    px = np.load(fresh/'pixels.npy')
    features = visual_input(torch.as_tensor(np.unpackbits(px,axis=1),dtype=torch.bool,device='cuda'),model.landmarks)
    require(not px[0].any(), 'Nonwhite source pad')
    with np.load(fresh/'contexts.npz') as f:
        x,y,groups,tags = f['context'],f['target'],f['group'],f['tag']
    truth = px[y]
    a,b = matching_pairs(x,y,groups)
    hits = {}
    result = {}
    for arm in ('intact','nearest_eight','outside_eight'):
        if arm=='intact':
            archive = fresh/'distilled_targets/generated.npz'
            require(sha(archive)==existing['fresh']['arms']['distilled_targets']['archive_sha256'],'Intact archive changed')
        else:
            altered = x.copy()
            if arm=='nearest_eight':
                altered[:, :20] = 0
                altered[:, 28:] = 0
            else:
                altered[:, 20:28] = 0
            streams, lengths, ends = [], [], []
            for pos in range(0,len(x),128):
                first.guard(start)
                inputs = torch.as_tensor(altered[pos:pos+128].astype(np.int32),device='cuda')
                ink,n,stop = model.generate(features[inputs])
                ink[np.arange(8)[None,:]>=n[:,None]] = 0
                streams.append(ink)
                lengths.extend(n)
                ends.extend(stop)
            archive = work/(arm+'.npz')
            np.savez_compressed(archive,packed=np.concatenate(streams),length=lengths,stopped=ends)
        with np.load(archive) as f:
            ink,n,stop = f['packed'],f['length'],f['stopped']
        hit = (ink[:,:4]==truth).all((1,2)) & stop & (n==4)
        hits[arm] = hit
        both = hit[a] & hit[b]
        result[arm] = dict(exact=int(hit.sum()), both=int(both.sum()), gained=int((hit & ~hits['intact']).sum()),
                           lost=int((~hit & hits['intact']).sum()), archive_sha256=sha(archive),
                           intact_minus_masked_pp_95=paired_interval(hits['intact'],hit,groups))
    result['intact_both_pairs'] = [
        dict(tags=tags[[i,j]].tolist(), rows=[int(i),int(j)], correct={k: h[[i,j]].tolist() for k,h in hits.items()})
        for i,j in zip(a,b) if hits['intact'][i] and hits['intact'][j]]
    result['n'] = len(x)
    result['seconds'] = time.monotonic()-start
    result['protocol_sha256'] = sha(ROOT/'references/visual_writer_context_distance_protocol.md')
    result['model_sha256'] = sha(OUT/'distilled_targets/weights.pt')
    result['program_sha256'] = sha(Path(__file__))
    write_new(destination/'context_distance.json',result)
    print(result,flush=True)

if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
