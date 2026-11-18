"""Test distinct visual answers sharing both nearest pictured neighbours."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.diagnose_visual_context_gradients import sha,write_new
from scripts.fit_pictured_idioms import ROOT,DATA

def main():
    with np.load(DATA/'validation.npz') as d:x,y,g=d['context'],d['target'],d['group']
    groups=defaultdict(list)
    for i,c in enumerate(x):groups[c[23:25].tobytes()].append(i)
    pairs=[]
    for group in groups.values():
        eligible=list(group)
        while eligible:
            i=eligible.pop(0)
            for pos,j in enumerate(eligible):
                if g[i]!=g[j] and not np.array_equal(y[i],y[j]):
                    pairs.append((i,eligible.pop(pos)));break
    a,b=np.asarray(pairs).T;px=np.load(DATA/'pixels.npy');truth=px[y];arms={}
    for name,folder in [('ink','pictured_idioms_v1'),('alignment','pictured_idiom_alignment_v1'),('selection','generated_idiom_choice_v1')]:
        archive=ROOT/'artifacts'/folder/'generated.npz'
        with np.load(archive) as f:ink,n,s=f['packed'],f['length'],f['stopped']
        hits=(ink[:,:4]==truth).all((1,2)) & (n==4) & s
        both=hits[a]&hits[b];diff=(n[a]!=n[b]) | (s[a]!=s[b]) | (ink[a]!=ink[b]).any((1,2))
        arms[name]=dict(changed=int(diff.sum()),both_correct=int(both.sum()),first_both=[int(k) for k in pairs[np.flatnonzero(both)[0]]] if both.any() else None,
                        single_correct=int((hits[a]^hits[b]).sum()),archive_sha256=sha(archive))
    result=dict(pairs=len(pairs),compared_arms=arms,nearest_two_only_both_correct_ceiling=0)
    write_new(ROOT/'artifacts/generated_idiom_choice_v1/context_pairs.json',result);print(result)


if __name__=='__main__':
    main()
