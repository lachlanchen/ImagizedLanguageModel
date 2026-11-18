"""Reproduce the frozen writer using its original numeric execution mode."""
import fcntl
import time

import numpy as np
import torch

from scripts.fit_mixed_ink_reader import (FRESH, OUT, ROOT, PicturedIdiom, draw, guard,
    load, load_initial, require, sha, verified_split, visual_input, write_new)


def run():
    start = time.monotonic()
    guard(start)
    report = load(OUT/'report.json')
    folder = OUT/'replay'
    folder.mkdir()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    data = verified_split(FRESH)
    bank = np.load(FRESH/'pixels.npy')
    state = load_initial()
    net = PicturedIdiom(state['landmarks']).cuda().eval().requires_grad_(False)
    net.load_state_dict(state)
    features = visual_input(torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda'),net.landmarks)
    pic,n,stop = draw(net,features,data['context'],start)
    np.savez_compressed(folder/'generated.npz',packed=pic,length=n,stopped=stop)
    old_path = FRESH/'released/generated.npz'
    prior = load(ROOT/'references/evidence/glyph_planner_20261118/evaluation.json')['arms']['released']
    require(sha(old_path)==prior['archive_sha256'],'Prior archive changed')
    with np.load(old_path) as old:
        same = (old['packed']==pic).all((1,2)) & (n==old['length']) & (stop==old['stopped'])
    x,y = data['context'],data['target']
    absent = ~(y[:,:,None]==x[:,None]).any((1,2))
    exact = (pic[:,:4]==bank[y]).all((1,2)) & (n==4) & stop
    eligible = (x[:,20:28]!=0).all(1)
    outcomes = {}
    for variant in ('familiar','unseen'):
        hits = []
        for role in range(3):
            path = OUT/f'{variant}_{role}.npz'
            require(sha(path)==report['output_sha256'][path.name],'Changed source decisions')
            with np.load(path) as a:
                require((a['actions'][:,0] == (role==0)).all(),'Observed gate differs')
                hits.append(exact if role==0 else a['exact'])
        outcomes[variant] = int((eligible & np.logical_and.reduce(hits)).sum())
    result = dict(identical=int(same.sum()),total=len(x),exact=int(exact.sum()),
                  all_absent_exact=int((absent & exact).sum()),all_three=outcomes,
                  archive_sha256=sha(folder/'generated.npz'),old_sha256=sha(old_path),
                  initial_report_sha256=sha(OUT/'report.json'),seconds=time.monotonic()-start)
    write_new(folder/'report.json',result)
    print(result)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
