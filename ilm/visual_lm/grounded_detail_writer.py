"""Fit coarse and detail-conditioned writing from paired glyph pictures."""
from pathlib import Path

import numpy as np

from ilm.visual_lm.patch_count_language import tile_features
from ilm.visual_lm.visual_extractive_reader import packed_tiles

KEY = np.dtype([('layout', '<u2'), ('ink', '<u8')])
LEVELS = ('position', 'layout', 'center', 'surround', 'layout_center', 'layout_surround')


def parts(cells):
    images = np.unpackbits(packed_tiles(cells), axis=1).reshape(-1,32,32)
    grid = images.reshape(-1,8,4,8,4).transpose(0,1,3,2,4).reshape(-1,64,16)
    centers = np.packbits(grid, axis=-1).view('>u2').reshape(-1,64)
    padded = np.pad(images, ((0,0),(2,2),(2,2)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (8,8), axis=(1,2))[:,::4,::4]
    surround = np.packbits(windows.reshape(-1,64,64), axis=-1).view('>u8').reshape(-1,64)
    layout = tile_features(images)[0].astype(np.uint16)
    return layout, centers, surround


def condition(shape, ink, count):
    out = np.empty(count, dtype=KEY)
    out['layout'], out['ink'] = shape, ink
    return out


def conditions(layout, centers, surround, j):
    return (condition(0,0,len(layout)), condition(layout,0,len(layout)),
            condition(0,centers[:,j],len(layout)), condition(0,surround[:,j],len(layout)),
            condition(layout,centers[:,j],len(layout)), condition(layout,surround[:,j],len(layout)))


def fit_modes(keys, targets, weights):
    # Aggregate identical source condition / target patch events, then take their weighted mode.
    order = np.lexsort((targets, keys['ink'], keys['layout']))
    key, y = keys[order], targets[order]
    first = np.r_[0,np.flatnonzero((key[1:]!=key[:-1]) | (y[1:]!=y[:-1]))+1]
    mass = np.add.reduceat(weights[order], first)
    key, y = key[first], y[first]
    rank = np.lexsort((y, -mass, key['ink'], key['layout']))
    key, y = key[rank], y[rank]
    best = np.r_[True,key[1:]!=key[:-1]]
    return key[best], y[best].astype(np.uint16)


class GroundedDetailWriter:
    def __init__(self, tables):
        self.tables = tables

    @classmethod
    def fit(cls, source, target, weights):
        source, target = packed_tiles(source), packed_tiles(target)
        weights = np.asarray(weights, dtype=np.int64)
        if source.shape != target.shape or weights.shape != (len(source),) or np.any(weights<=0):
            raise ValueError('Supply one positive frequency per paired picture')
        layout, center, surround = parts(source)
        _, values, _ = parts(target)
        tables = [[fit_modes(keys, values[:,j], weights) for keys in conditions(layout,center,surround,j)]
                  for j in range(64)]
        return cls(tables)

    def paint(self, source, *, detail=True):
        layout, center, surround = parts(source)
        answer = np.empty((len(layout),64), dtype='>u2')
        for j, tables in enumerate(self.tables):
            requests = conditions(layout,center,surround,j)
            answer[:,j] = tables[0][1][0]
            for k in range(1, len(tables) if detail else 2):
                keys, values = tables[k]
                idx = np.minimum(np.searchsorted(keys, requests[k]), len(keys)-1)
                match = keys[idx] == requests[k]
                answer[match,j] = values[idx[match]]
        pixels = np.unpackbits(answer.view(np.uint8), axis=1).reshape(-1,8,8,4,4).transpose(0,1,3,2,4)
        return np.packbits(pixels.reshape(-1,1024),axis=1)

    def save(self, path):
        arrays={'version':np.array(1)}
        for j, tables in enumerate(self.tables):
            for k,(keys,values) in enumerate(tables):
                arrays[f'keys_{j}_{k}'], arrays[f'modes_{j}_{k}'] = keys, values
        with Path(path).open('xb') as f:
            np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data['version'])!=1:
                raise ValueError('Unsupported detail writer format')
            return cls([[(data[f'keys_{j}_{k}'],data[f'modes_{j}_{k}']) for k in range(len(LEVELS))]
                        for j in range(64)])


def answer(reader, writer, passage, question, *, detail=True):
    left, right = reader.bounds(passage, question)
    return writer.paint(passage[left:right], detail=detail), (left,right)
