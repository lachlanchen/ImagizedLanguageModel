"""Losslessly pack the source-conditioned image statistics and verify every count after reload."""
from pathlib import Path
import argparse
import numpy as np

from ilm.visual_lm.local_ink import LocalInk
from scripts.train_local_ink import RUN, load, require, sha, write_new
from scripts.infer_visual_phrase import MODELS


def export(out):
    out.mkdir(parents=True)
    result = {}
    for name,path in [('onset',RUN/'source.npz'),('later',RUN/'stream/source_later.npz'),('end',RUN/'stream/source_end.npz')]:
        expected = load(RUN/'source_fit.json')['sha256'] if name=='onset' else load(RUN/'stream/source_fit.json')['tables'][name]['sha256']
        require(sha(path)==expected,'Input count file differs')
        model = LocalInk.load(path)
        compact = LocalInk(model.root,{k:t.compact() for k,t in model.tables.items()},model.strength)
        dst = out/(name+'.npz')
        compact.save(dst)
        restored = LocalInk.load(dst)
        require(np.array_equal(model.root,restored.root) and np.array_equal(model.strength,restored.strength),'Changed root or strength')
        for k,t in model.tables.items():
            other = restored.tables[k]
            require(np.array_equal(t.keys,other.keys),'Changed context identities')
            for idx in np.array_split(np.arange(len(t.keys)),64):
                require(np.array_equal(other.read(idx),t.values[idx]),'Count changed during compression')
        result[name] = dict(sha256=sha(dst),source_sha256=expected,bytes=sum(a.nbytes for a in compact.arrays().values()),
                            compressed_bytes=dst.stat().st_size)
    write_new(out/'manifest.json',dict(format='ilm-local-ink-v1',phrase_update_sha256=sha(MODELS/'phrase_update.safetensors'),tables=result))
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=MODELS/'local_ink')
    export(p.parse_args().output)
