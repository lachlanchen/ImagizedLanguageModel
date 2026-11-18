"""Autoregressive binary-image language baseline with conditional physical patch counts."""
from pathlib import Path

import numpy as np

SIDE = 32
PATCH = 8
START = 1 << 16  # Outside the 16-bit visual signature space.
RADIX = START + 1
ESCAPE = 1 / 65536


def tile_features(images):
    """Return 4-by-4 coarse ink signatures and sixteen literal 8-by-8 patches."""
    ink = np.asarray(images)
    if ink.ndim != 3 or ink.shape[1:] != (SIDE, SIDE) or ink.dtype != np.uint8 or (ink > 1).any():
        raise ValueError('Supply 32-by-32 uint8 glyphs with one for ink')
    patches = ink.reshape(-1, 4, PATCH, 4, PATCH).transpose(0, 1, 3, 2, 4).reshape(-1, 16, 64)
    mass = patches.sum(-1)
    signature = np.packbits(mass*16 > mass.sum(-1, keepdims=True), axis=1).view('>u2').ravel()
    values = np.packbits(patches, axis=-1).view('>u8').reshape(-1, 16)
    return signature.astype(np.uint64), values.astype(np.uint64)


def assemble(bits):
    parts = np.unpackbits(np.asarray(bits, dtype='>u8').view(np.uint8)).reshape(4, 4, PATCH, PATCH)
    return parts.transpose(0, 2, 1, 3).reshape(SIDE, SIDE).copy()


class CountTable:
    def __init__(self, keys, counts, width):
        self.keys, self.counts, self.width = keys, counts, width

    @classmethod
    def fit(cls, conditions, values, width):
        keys = np.asarray(conditions, np.uint64)*np.uint64(width) + np.asarray(values, np.uint64)
        unique, counts = np.unique(keys, return_counts=True)
        return cls(unique, counts.astype(np.uint32), width)

    def interpolate(self, condition, base):
        key = int(condition)*self.width
        lo, hi = np.searchsorted(self.keys, np.asarray([key,key+self.width], dtype=np.uint64))
        if hi == lo:
            return base
        observed = self.counts[lo:hi]
        result = base * (hi-lo)
        result[(self.keys[lo:hi] - np.uint64(key)).astype(np.intp)] += observed
        return result / (float(observed.sum())+hi-lo)


def history_key(penultimate, last):
    return penultimate * RADIX + last


class PatchCountLanguage:
    """Learn suffix-conditioned ink parts, their spatial joins and a separate end event."""
    def __init__(self, parts, ending, part_radix):
        self.parts = parts
        self.ending = ending
        self.part_radix = part_radix

    @classmethod
    def fit(cls, input_tiles, passages, *, rotated=False):
        """Passage indices name rows of input_tiles during training only."""
        signatures, patches = tile_features(input_tiles)
        preceding, last, target, end_previous, end_last = [], [], [], [], []
        for passage in passages:
            ids = np.asarray(passage, dtype=np.intp)
            if ids.ndim != 1 or len(ids) < 1 or (ids < 0).any() or (ids >= len(signatures)).any():
                raise ValueError('Invalid visual training passage')
            seq = signatures[ids]
            before = np.r_[np.uint64(START), seq[:-1]]
            preceding.append(before[:-1])
            last.append(seq[:-1])
            successors = ids[1:]
            target.append(np.roll(successors, len(successors)//2) if rotated else successors)
            end_previous.append(before[-1])
            end_last.append(seq[-1])
        prev, tail, targets = (np.concatenate(a) for a in (preceding,last,target))
        if not len(targets):
            raise ValueError('Training requires at least one visual transition')
        pairs = history_key(prev, tail)
        stop_labels = np.r_[np.zeros(len(pairs),np.uint8), np.ones(len(end_last),np.uint8)]
        stop_tail = np.r_[tail, np.asarray(end_last,np.uint64)]
        stop_pairs = np.r_[pairs,history_key(np.asarray(end_previous,np.uint64),np.asarray(end_last,np.uint64))]
        ending = ((np.bincount(stop_labels, minlength=2)+1)/(len(stop_labels)+2),
                  CountTable.fit(stop_tail, stop_labels, 2), CountTable.fit(stop_pairs, stop_labels, 2))
        occurring, counts = np.unique(targets, return_counts=True)
        alphabets = [np.unique(patches[occurring,j]) for j in range(16)]
        radix = max(map(len,alphabets))+1
        if radix > (1 << 21):
            raise ValueError('Physical patch support exceeds the sparse table key capacity')
        # A zero neighbour means that side borders the outside of the glyph.
        ranks = np.column_stack([np.searchsorted(alpha, patches[:,j]) for j, alpha in enumerate(alphabets)])
        parts = []
        for j, alpha in enumerate(alphabets):
            values = ranks[targets,j]
            local = np.zeros(len(occurring), dtype=np.uint64)
            if j >= 4:
                local += (ranks[occurring,j-4].astype(np.uint64)+1)*radix
            if j % 4:
                local += ranks[occurring,j-1].astype(np.uint64)+1
            # Identical target glyphs share their spatial events. Weight by their real frequencies.
            shape_keys = local*np.uint64(len(alpha)) + ranks[occurring,j].astype(np.uint64)
            keys, inverse = np.unique(shape_keys, return_inverse=True)
            shape_counts = np.zeros(len(keys), dtype=np.uint32)
            np.add.at(shape_counts,inverse,counts.astype(np.uint32))
            prior = np.bincount(values,minlength=len(alpha)).astype(np.float64)/len(values)
            parts.append((alpha,prior,CountTable.fit(tail,values,len(alpha)),CountTable.fit(pairs,values,len(alpha)),
                          CountTable(keys,shape_counts,len(alpha))))
        return cls(parts,ending,radix)

    @staticmethod
    def history(images):
        sig, _ = tile_features(np.asarray(images)[-2:])
        if not len(sig):
            raise ValueError('A visible prompt is required')
        return int(sig[-2]) if len(sig)>1 else START, int(sig[-1])

    def stop_probability(self, history):
        a,b = history
        prior,short,long = self.ending
        return float(long.interpolate(history_key(a,b), short.interpolate(b,prior))[1])

    def patch_probabilities(self, j, history, spatial):
        _,prior,short,long,joins = self.parts[j]
        a,b = history
        sequence = long.interpolate(history_key(a,b), short.interpolate(b,prior))
        position = joins.interpolate(spatial,prior)
        mass = sequence * (position/prior)
        return mass / mass.sum()

    def draw(self, history, *, generator=None):
        """Generate tiles from scores over physical 8-by-8 bit patterns."""
        raw, neighbours = [], []
        for j, (alphabet, *_) in enumerate(self.parts):
            up = neighbours[j-4] if j >= 4 else 0
            left = neighbours[j-1] if j%4 else 0
            prob = self.patch_probabilities(j,history,up*self.part_radix+left)
            if generator is not None and generator.random() < ESCAPE:
                bits = int.from_bytes(generator.bytes(8),'big')
                rank = int(np.searchsorted(alphabet,np.uint64(bits)))
                seen = rank < len(alphabet) and int(alphabet[rank]) == bits
                neighbour = rank+1 if seen else 0
            else:
                rank = int(np.argmax(prob) if generator is None else generator.choice(len(prob),p=prob))
                bits, neighbour = int(alphabet[rank]), rank+1
            raw.append(bits)
            neighbours.append(neighbour)
        return assemble(raw)

    def generate(self, prompt, *, maximum_new=128, generator=None):
        if maximum_new <= 0:
            raise ValueError('A positive output cap is required')
        history = self.history(prompt)
        output = []
        for _ in range(maximum_new):
            p = self.stop_probability(history)
            if (p >= .5 if generator is None else generator.random() < p):
                break
            image = self.draw(history,generator=generator)
            output.append(image)
            new, _ = tile_features(image[None])
            history = history[1], int(new[0])
        return np.asarray(output,dtype=np.uint8).reshape(-1,32,32), len(output)<maximum_new

    def save(self, path):
        arrays = {'version':np.array(1),'radix':np.array(self.part_radix),'stop_prior':self.ending[0]}
        def add(name, table):
            arrays[name+'_keys'], arrays[name+'_counts'] = table.keys, table.counts
        for name, table in zip(('stop_short','stop_long'),self.ending[1:]):
            add(name,table)
        for j,(alpha,prior,short,long,joins) in enumerate(self.parts):
            arrays[f'alpha_{j}'],arrays[f'prior_{j}']=alpha,prior
            for name,table in [('short',short),('long',long),('joins',joins)]:
                add(f'{name}_{j}',table)
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream,**arrays)

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as z:
            if int(z['version'])!=1:
                raise ValueError('Unsupported patch count model')
            def table(name,width):
                return CountTable(z[name+'_keys'],z[name+'_counts'],width)
            ending=(z['stop_prior'],table('stop_short',2),table('stop_long',2))
            parts=[]
            for j in range(16):
                alpha=z[f'alpha_{j}']
                parts.append((alpha,z[f'prior_{j}'],*(table(f'{name}_{j}',len(alpha)) for name in ('short','long','joins'))))
            return cls(parts,ending,int(z['radix']))
