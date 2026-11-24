"""Check the released likelihood against inference and check absence of future-cell leakage."""
import numpy as np
import torch
from ilm.visual_lm.book_predict import Predictor
from ilm.visual_lm.prefix_counts import history_windows
from scripts.prepare_book_prefix import corpus,MODEL

def main():
    torch.set_num_threads(2)
    meta,data = corpus()
    ids = data['stream'][next(r[4] for r in meta['records'] if r[0]=='train' and r[5]-r[4]>64):][:65]
    model = Predictor(MODEL,'ink-hybrid',data,'cpu')
    history = history_windows(ids,[(0,len(ids))])
    node,cf = model.count.lookup(history[1:])
    cp = model.count.target_prob(node,ids[1:])
    x = torch.from_numpy(ids[:-1].astype(np.int64)).unsqueeze(0)
    y = torch.from_numpy(ids[1:].astype(np.int64)).unsqueeze(0)
    with torch.inference_mode():
        loss = model.model.loss(model.feats[x],x,y,torch.tensor(cf)[None],torch.tensor(cp)[None])
        mass = model.probabilities(ids[None,:-1],5)
        expected = -mass.gather(-1,y[:,5:,None]).log().mean()
        torch.testing.assert_close(loss,expected)
        short = model.probabilities(ids[None,:16],15)
        torch.testing.assert_close(short[0,0],mass[0,10],atol=1e-6,rtol=2e-5)
    # A held-only image in a prompt can be copied; another held-only image is unreachable.
    outsider = meta['train_shapes'] + 1
    short = np.array([[ids[0],outsider]],np.uint16)
    p = model.probabilities(short,1)[0,0]
    assert 0 < p[outsider] < 1 and p[outsider+1] == 0 and p[0] == 0
    print(f'Likelihood agreement {float(loss):.7f}, prefix causality, new-image support verified.')

if __name__=='__main__':
    main()
