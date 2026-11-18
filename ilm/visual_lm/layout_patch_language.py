"""A coarse physical layout shared by the sixteen patches of a generated glyph."""
from pathlib import Path

import numpy as np

from ilm.visual_lm.patch_count_language import CountTable, PatchCountLanguage, START, history_key, tile_features


def weighted_table(condition, values, weights, width):
    encoded = condition.astype(np.uint64) * np.uint64(width) + values.astype(np.uint64)
    keys, groups = np.unique(encoded, return_inverse=True)
    counts = np.zeros(len(keys), dtype=np.uint64)
    np.add.at(counts, groups, weights.astype(np.uint64))
    return CountTable(keys, counts, width)


def suffix_tables(first, last, outcomes, width):
    counts = np.bincount(outcomes, minlength=width).astype(np.float64)
    prior = (counts + 1) / (len(outcomes) + width)
    return prior, CountTable.fit(last, outcomes, width), CountTable.fit(history_key(first, last), outcomes, width)


class LayoutPatchLanguage(PatchCountLanguage):
    """A two-glyph suffix predicts regional ink; a conditional writer draws its detail."""

    def __init__(self, parts, ending, radix, layouts, layout_suffix):
        super().__init__(parts, ending, radix)
        self.layouts = layouts
        self.layout_suffix = layout_suffix

    @classmethod
    def fit(cls, tiles, passages, *, rotated=False):
        sig, bits = tile_features(tiles)
        first, last, targets, end_first, end_last = [], [], [], [], []
        for sequence in passages:
            ids = np.asarray(sequence)
            if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
                    or (ids < 0).any() or (ids >= len(tiles)).any()):
                raise ValueError('Each training passage must index visible glyph cells')
            history = sig[ids]
            previous = np.r_[np.uint64(START), history[:-1]]
            first.append(previous[:-1])
            last.append(history[:-1])
            following = ids[1:]
            targets.append(np.roll(following, len(following)//2) if rotated else following)
            end_first.append(previous[-1])
            end_last.append(history[-1])
        if not targets or not any(len(t) for t in targets):
            raise ValueError('Supply at least one visible transition')
        first, last, targets = map(np.concatenate, (first, last, targets))
        ending = suffix_tables(np.r_[first, end_first], np.r_[last, end_last],
                               np.r_[np.zeros(len(last), np.uint8), np.ones(len(end_last), np.uint8)], 2)
        layouts, classes = np.unique(sig[targets], return_inverse=True)
        layout_suffix = suffix_tables(first, last, classes, len(layouts))

        unique, weights = np.unique(targets, return_counts=True)
        shape = sig[unique]
        alphabets = [np.unique(bits[unique, j], return_inverse=True) for j in range(16)]
        radix = max(len(alpha) for alpha, _ in alphabets) + 1
        if (int(shape.max()) + 1) * radix**3 >= 2**64:
            raise ValueError('Layout and neighbour combinations exceed the sparse key width')
        parts = []
        for j, (alphabet, rank) in enumerate(alphabets):
            width = len(alphabet)
            neighbours = np.zeros(len(unique), dtype=np.uint64)
            if j >= 4:
                neighbours += (alphabets[j-4][1].astype(np.uint64) + 1) * radix
            if j % 4:
                neighbours += alphabets[j-1][1].astype(np.uint64) + 1
            # Aggregate identical events without expanding every image into sixteen training rows.
            prior = np.bincount(rank, weights=weights, minlength=width) / weights.sum()
            coarse = weighted_table(shape, rank, weights, width)
            adjacent = weighted_table(shape * radix**2 + neighbours, rank, weights, width)
            parts.append((alphabet, prior, coarse, adjacent))
        return cls(parts, ending, radix, layouts, layout_suffix)

    def layout_probabilities(self, history):
        first, last = history
        prior, short, long = self.layout_suffix
        return long.interpolate(history_key(first, last), short.interpolate(last, prior))

    def patch_probabilities(self, j, layout, spatial):
        _, prior, coarse, adjacent = self.parts[j]
        return adjacent.interpolate(int(layout) * self.part_radix**2 + spatial, coarse.interpolate(layout, prior))

    def draw_layout(self, layout, *, generator=None):
        """Decode one physical grid, using the common bit-packing and literal escape law."""
        if not 0 <= int(layout) < START:
            raise ValueError('A physical layout has sixteen bits')
        return super().draw(int(layout), generator=generator)

    def draw(self, history, *, generator=None):
        prob = self.layout_probabilities(history)
        k = np.argmax(prob) if generator is None else generator.choice(len(prob), p=prob)
        return self.draw_layout(self.layouts[k], generator=generator)

    def save(self, path):
        arrays = dict(version=np.array(2), radix=np.array(self.part_radix), layouts=self.layouts)
        for name, (prior, short, long) in [('end', self.ending), ('layout', self.layout_suffix)]:
            arrays[name+'_prior'] = prior
            for level, table in [('short', short), ('long', long)]:
                arrays[name+'_'+level+'_keys'] = table.keys
                arrays[name+'_'+level+'_counts'] = table.counts
        for j, (alphabet, prior, coarse, adjacent) in enumerate(self.parts):
            arrays[f'alpha_{j}'], arrays[f'prior_{j}'] = alphabet, prior
            for name, table in [('coarse', coarse), ('adjacent', adjacent)]:
                arrays[f'{name}_{j}_keys'], arrays[f'{name}_{j}_counts'] = table.keys, table.counts
        with Path(path).open('xb') as f:
            np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z['version']) != 2:
                raise ValueError('Unsupported shared-layout checkpoint')

            def table(name, width):
                return CountTable(z[name+'_keys'], z[name+'_counts'], width)

            def suffix(name):
                prior = z[name+'_prior']
                return prior, table(name+'_short', len(prior)), table(name+'_long', len(prior))

            parts = []
            for j in range(16):
                alpha = z[f'alpha_{j}']
                parts.append((alpha, z[f'prior_{j}'], table(f'coarse_{j}', len(alpha)),
                              table(f'adjacent_{j}', len(alpha))))
            return cls(parts, suffix('end'), int(z['radix']), z['layouts'], suffix('layout'))
