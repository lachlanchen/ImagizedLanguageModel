"""Smoothed local four-pixel marginals for causal glyph generation."""
from dataclasses import dataclass

import numpy as np

# Fields are the left and right source cells, then the older and newer prefix cells.
LEVELS = (((0,), (1,), (3,)), ((0,1), (0,3), (1,3), (2,3)), ((0,1,3),), ((0,1,2,3),))
MISSING = np.uint64(1 << 48)
GROUPS = 12
PATTERNS = 16


def signatures(bits):
    """A physical 48-bit signature, including zero for an actual blank cell."""
    packed = np.packbits(np.asarray(bits, dtype=bool), axis=-1)
    if bits.shape[-1] != 48:
        raise ValueError('Expected exactly 48 measured pixels per cell')
    return (packed.astype(np.uint64) << np.arange(40,-1,-8,dtype=np.uint64)).sum(-1,dtype=np.uint64)


def context_signature(context, columns):
    # Each subset has a separate table. Mix both halves of each 48-bit measurement.
    key = np.full(len(context), 14695981039346656037, dtype=np.uint64)
    for col in columns:
        value = context[:,col]
        for shift in (48,32,16,0):
            key ^= (value >> np.uint64(shift)) & np.uint64(65535)
            key *= np.uint64(1099511628211)
    return key


def cell_context(source, cursor, prefix):
    if not 0 <= cursor <= len(source):
        raise ValueError('Invalid boundary')
    return (source[cursor-1] if cursor else MISSING, source[cursor] if cursor<len(source) else MISSING,
            prefix[-2] if len(prefix)>=2 else MISSING, prefix[-1] if len(prefix) else MISSING)


@dataclass
class Counts:
    keys: np.ndarray
    values: np.ndarray

    wide_rows: np.ndarray | None = None
    wide_values: np.ndarray | None = None

    def read(self, indices):
        data = self.values[indices].astype(np.float32)
        if self.wide_rows is not None and len(self.wide_rows):
            wide = np.searchsorted(self.wide_rows,indices).clip(max=len(self.wide_rows)-1)
            match = self.wide_rows[wide]==indices
            data[match] = self.wide_values[wide[match]]
        return data

    def compact(self):
        if self.wide_rows is not None:
            return self
        wide = np.flatnonzero(self.values.max(axis=(1,2))>255).astype(np.int32)
        return Counts(self.keys,self.values.clip(max=255).astype(np.uint8),wide,self.values[wide])

    @classmethod
    def fit(cls, keys, target, patterns):
        distinct, inverse, frequency = np.unique(keys,return_inverse=True,return_counts=True)
        keep = frequency>=2
        valid = keep[inverse]
        index = keep.cumsum()[inverse[valid]]-1
        values = np.zeros((int(keep.sum()),target.shape[1],patterns),np.uint32)
        for j in range(target.shape[1]):
            joint = index*patterns+target[valid,j]
            values[:,j] = np.bincount(joint,minlength=len(values)*patterns).reshape(-1,patterns)
        return cls(distinct[keep],values)

    def find(self, keys):
        indices = np.searchsorted(self.keys,keys)
        if not len(self.keys):
            return indices,np.zeros(len(keys),bool)
        indices = indices.clip(max=len(self.keys)-1)
        return indices,self.keys[indices]==keys


class LocalInk:
    def __init__(self, root, tables, strength):
        self.root = root
        self.tables = tables
        self.strength = strength

    @classmethod
    def fit(cls, context, target, *, ablate=False, binary=False):
        groups,patterns = (1,2) if binary else (GROUPS,PATTERNS)
        if len(target)!=len(context) or not len(target) or target.shape!=(len(context),groups):
            raise ValueError('Missing visual count observations')
        global_counts = np.stack([np.bincount(target[:,j],minlength=patterns) for j in range(groups)])
        root = (global_counts+.5)/(len(target)+patterns*.5)
        tables = {}
        for level in LEVELS:
            for cols in level:
                if not (ablate and min(cols)<2):
                    tables[cols] = Counts.fit(context_signature(context,cols),target,patterns)
        return cls(root.astype(np.float32),tables,np.ones(groups,np.float32))

    def log_ratio(self, context):
        prior = np.broadcast_to(self.root,(len(context),*self.root.shape)).copy()
        for level in LEVELS:
            posterior = np.zeros_like(prior)
            votes = np.zeros(len(context),dtype=np.int32)
            for cols in level:
                if cols not in self.tables:
                    continue
                table = self.tables[cols]
                idx,seen = table.find(context_signature(context,cols))
                counts = table.read(idx[seen])
                total = counts[:,0,:].sum(1)[:,None,None]
                posterior[seen] += (counts+4*prior[seen])/(total+4)
                votes += seen
            seen = votes>0
            prior[seen] = posterior[seen]/votes[seen,None,None]
        return np.log(prior)-np.log(self.root)

    def bias(self, context):
        return self.log_ratio(context)*self.strength[None,:,None]

    def arrays(self):
        data = dict(root=self.root,strength=self.strength)
        for cols,table in self.tables.items():
            label = ''.join(map(str,cols))
            data['keys_'+label] = table.keys
            data['counts_'+label] = table.values
            if table.wide_rows is not None:
                data['wide_rows_'+label] = table.wide_rows
                data['wide_values_'+label] = table.wide_values
        return data

    def save(self, path):
        with path.open('xb') as stream:
            np.savez_compressed(stream,**self.arrays())

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            root,strength = data['root'],data['strength']
            if root.shape not in ((GROUPS,PATTERNS),(1,2)) or strength.shape!=(len(root),):
                raise ValueError('Incompatible local probability dimensions')
            tables = {}
            for level in LEVELS:
                for cols in level:
                    label = ''.join(map(str,cols))
                    if 'keys_'+label in data:
                        key,counts = data['keys_'+label],data['counts_'+label]
                        if key.dtype!=np.uint64 or counts.shape!=(len(key),*root.shape) or (key[1:]<=key[:-1]).any():
                            raise ValueError('Incompatible count table')
                        if counts.dtype==np.uint32:
                            tables[cols] = Counts(key,counts)
                        elif counts.dtype==np.uint8:
                            rows,wide = data['wide_rows_'+label],data['wide_values_'+label]
                            if rows.dtype!=np.int32 or wide.dtype!=np.uint32 or wide.shape!=(len(rows),*root.shape):
                                raise ValueError('Incompatible high-count rows')
                            if len(rows) and (rows[0]<0 or rows[-1]>=len(key) or (rows[1:]<=rows[:-1]).any()):
                                raise ValueError('Invalid high-count row indices')
                            tables[cols] = Counts(key,counts,rows,wide)
                        else:
                            raise ValueError('Unsupported count type')
            return cls(root,tables,strength)

class PhraseInk:
    """The three calibrated distributions used before and after the first emission."""
    def __init__(self, onset, later, ending):
        self.onset = onset
        self.later = later
        self.ending = ending

    def update(self, context, emitted_bits):
        context[:,2] = context[:,3]
        context[:,3] = signatures(emitted_bits)
        bias = self.ending.bias(context)[:,0]
        return bias[:,1]-bias[:,0]
