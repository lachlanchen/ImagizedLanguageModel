"""Conditional Chinese glyph counts indexed by the last four physical images."""
from functools import lru_cache

import numpy as np
import torch


class BitmapHistory:
    def __init__(self, corpus, memory_tiles, condition):
        if condition not in ("ordered", "shuffled"):
            raise ValueError(condition)
        tiles = np.load(corpus / "tiles.npy", mmap_mode="r")
        self.base = len(tiles)
        self.tile_ids = {bytes(tile): i for i,tile in enumerate(tiles) if i}
        emission_ids = {bytes(tile): i for i,tile in enumerate(memory_tiles)}
        self.to_memory = np.array([emission_ids.get(bytes(tile),-1) for tile in tiles], np.int32)
        self.keys = np.load(corpus / condition / "keys.npy", mmap_mode="r")
        self.next_tile = np.load(corpus / condition / "next.npy", mmap_mode="r")
        self.max_history = 4

    @lru_cache(maxsize=8192)
    def counts(self, prefix):
        key = 0
        for ident in prefix:
            key = self.base*key + ident
        width = self.base**(self.max_history - len(prefix))
        lower, upper = self.keys.searchsorted([key*width, (key+1)*width])
        ids = self.to_memory[self.next_tile[lower:upper]]
        return np.unique(ids[ids>=0],return_counts=True)

    def probabilities(self, neural, history):
        p = np.asarray(neural,dtype=np.float64).copy()
        recent = []
        for k in range(1, min(self.max_history,len(history)//128)+1):
            stop = len(history) - (k-1)*128
            ident = self.tile_ids.get(history[stop-128:stop])
            if ident is None:
                break
            recent.append(ident)
            ids, counts = self.counts(tuple(recent))
            n = int(counts.sum())
            if n:
                strength = min(.8, n/(n+8+len(ids)))
                p *= 1-strength
                p[ids] += strength*counts/n
        return p

    def choose(self, logits, histories):
        neural = logits.softmax(-1).detach().cpu().numpy()
        chosen = [int(self.probabilities(p,h).argmax()) for p,h in zip(neural,histories,strict=True)]
        return torch.as_tensor(chosen,device=logits.device)
