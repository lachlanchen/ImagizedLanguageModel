"""Localize free phrase errors with exact reference histories and a perfect-length counterfactual."""
import fcntl
import time
import numpy as np
import torch
from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.local_ink import LocalInk, signatures
from scripts.evaluate_stream_ink import COHORTS, OUT, ROOT, RUN, checkpoint, guard, load, require, roster, sha, sites, write_new
from scripts.train_local_ink import fields, tensors


@torch.inference_mode()
def run():
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    model = checkpoint('source')
    tables = []
    for name,path in [('onset',RUN/'source.npz'),('later',RUN/'stream/source_later.npz'),('end',RUN/'stream/source_end.npz')]:
        h = load(RUN/'source_fit.json')['sha256'] if name=='onset' else load(RUN/'stream/source_fit.json')['tables'][name]['sha256']
        require(sha(path)==h,'Changed count source')
        tables.append(LocalInk.load(path))
    onset,later,end = tables
    for split in (*COHORTS,'fresh'):
        rows,bank = roster(split)
        ink = torch.from_numpy(np.unpackbits(bank,axis=1)).cuda().bool()
        bits = ink[:,model.landmarks]
        sig = signatures(bits.cpu().numpy())
        features = visual_input(ink,model.landmarks)
        q = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
        results = []
        for i in range(0,len(rows),24):
            block = [r for r in rows[i:i+24] if r['x']]
            loc = sites(block)
            if not loc:
                continue
            mem,_,hist,y,_ = tensors(block,features,bits,model,q,False)
            positions = [(b,c,t,k) for b,c,j,k in loc for t in range(j,k)]
            following = [(b,c,t+1,k) for b,c,t,k in positions]
            b,c,t,k = torch.tensor(positions,device='cuda').T
            boundary = model.boundary(mem,b,c)
            state = (boundary+hist[b,t]).tanh()
            truth = bits[y[b,t]]
            initial = np.array([t==j for b,c,j,k in loc for t in range(j,k)])
            first = torch.from_numpy(initial).cuda()
            ctx = fields(block,sig,positions)
            bias = np.empty((len(ctx),12,16),np.float32)
            bias[initial] = onset.bias(ctx[initial])
            bias[~initial] = later.bias(ctx[~initial])
            code = model.draw.choose(state,pattern_bias=torch.from_numpy(bias).cuda())
            # The supplied-code arm has the same later reference state whenever its first bitmap is correct.
            def draw(code):
                image = model.motor(code)>0
                image[:,model.landmarks] = code.bool()
                return (image==ink[y[b,t]]).all(1).cpu().numpy()
            correct = draw(code)
            supplied = draw(torch.where(first[:,None],truth,code))
            odds = end.bias(fields(block,sig,following))[:,0]
            after = (boundary+hist[b,t+1]).tanh()
            stop = model.end(after)[:,0]+torch.from_numpy(odds[:,1]-odds[:,0]).cuda()>=0
            actions = (stop==(t+1==k)).cpu().numpy()
            lengths = np.array([k-j for b,c,j,k in loc])
            idx = np.r_[0,np.cumsum(lengths)[:-1]]
            for flags in (correct,supplied,actions):
                require(len(flags)==lengths.sum(),'Incomplete history')
            results.extend(zip(correct[idx],np.logical_and.reduceat(correct,idx),np.logical_and.reduceat(supplied,idx),
                               np.logical_and.reduceat(actions,idx),lengths,strict=True))
        first,all_draw,given_draw,ends,length = np.asarray(results,dtype=np.int64).T
        allowed = length<=32
        folder = RUN/'stream'/split
        expected = load(folder/'report.json')['arms']
        audit = {}
        for arm,flags in [('source',all_draw),('supplied',given_draw)]:
            perfect_length = flags.astype(bool)&allowed
            exact = perfect_length & ends.astype(bool)
            require(int(exact.sum())==expected[arm]['exact'],'Reference-history exact count differs')
            audit[arm] = dict(exact=int(exact.sum()),perfect_length=int(perfect_length.sum()),
                              stopping_only_errors=int((perfect_length&~exact).sum()),
                              perfect_length_multi=int((perfect_length&(length>1)).sum()))
        require(first.sum()==expected['source']['first_glyph_exact'],'First-glyph replay differs')
        with np.load(folder/'source.npz') as z:
            known = {bytes(pic) for pic in bank}
            drawn = z['pixels']
            audit['known_emitted_bitmaps'] = [sum(bytes(pic) in known for pic in drawn),len(drawn)]
            freq = {}
            for a,b in zip(z['offsets'],z['offsets'][1:]):
                key = drawn[a:b].tobytes()
                freq[key] = freq.get(key,0)+1
            audit['distinct_phrases'] = len(freq)
            audit['most_frequent_phrase_occurrences'] = max(freq.values())
        audit.update(sites=len(length),over_cap=int((~allowed).sum()),wrong_first=int((first==0).sum()))
        write_new(folder/'length_audit.json',audit)
        print(split,audit,flush=True)
        guard(start)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
