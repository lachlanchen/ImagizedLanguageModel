"""A confidence-gated relation predictor over local sequences of observed glyph images."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np

BUCKETS=32768
THRESHOLD=.8


@lru_cache(maxsize=16384)
def glyph_hash(cell):
    return int.from_bytes(hashlib.blake2s(cell,digest_size=4).digest(),"little")


def features(question,subject):
    cells=[question[i:i+128] for i in range(0,len(question),128)]
    if subject:
        title=[subject[i:i+128] for i in range(0,len(subject),128)]
        # Consume whole cells. A byte replacement could match across physical cell boundaries.
        i=0
        while i<=len(cells)-len(title):
            if cells[i:i+len(title)]==title:cells[i:i+len(title)]=[b""];i+=1
            else:i+=1
    hashes=[glyph_hash(k) for k in cells]
    indices=set()
    for i in range(len(hashes)):
        code=17
        for h in hashes[i:i+3]:
            code=(code*65599+h) & 0xffffffff
            indices.add(code % BUCKETS)
    return np.asarray(sorted(indices),dtype=np.int32)


class PicturedRelation:
    def __init__(self,path):
        with np.load(Path(path),allow_pickle=False) as state:
            self.weights=state["weights"]
            self.bias=state["bias"]

    def choose(self,question,subject,base):
        scores=self.weights[:,features(question,subject)].sum(1)+self.bias
        mass=np.exp(scores-scores.max())
        role=int(scores.argmax())
        confidence=float(mass[role]/mass.sum())
        return (role if confidence>=THRESHOLD else base),confidence
