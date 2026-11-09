"""Physical-image corpus and bounded-window sampling for complete learning."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .native_image_continuation import NativeImagePrefix


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PixelLanguageCorpus:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.train = self.manifest['train']
        self.development = self.manifest['development']
        if len(self.train) != self.manifest['train_questions'] or len(self.train) % 32:
            raise ValueError('complete 32-question training windows required')
        if any(r['index'] != i or r['split'] != 'train' for i, r in enumerate(self.train)):
            raise ValueError('training order or partition differs')
        if len(self.development) != 24 or any(r['split'] != 'development' for r in self.development):
            raise ValueError('the fixed 24-question development partition is required')
        self._targets = {}

    def path(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe physical-image path')
        return self.root / path

    def array(self, relative, expected):
        path = self.path(relative)
        if digest(path) != expected:
            raise ValueError('owning image changed: ' + relative)
        with Image.open(path) as image:
            values = np.asarray(image.convert('L')).copy()
        if values.ndim != 2 or not np.isin(values, [0, 255]).all():
            raise ValueError('complete binary writing pixels required')
        return values

    def prompt(self, index, *, development=False, route='normal'):
        if development:
            if route not in ('normal', 'blank'):
                raise ValueError('normal or erased development input required')
            row = self.development[index]
            values = self.array(row[route], row[route + '_sha256'])
        else:
            if route != 'normal':
                raise ValueError('training reads the original normal prompt only')
            row = self.train[index]
            values = self.array(row['prompt'], row['prompt_sha256'])
        return torch.from_numpy(values).float().div(127.5).sub(1)[None, None]

    def targets(self, index):
        if index not in self._targets:
            row = self.train[index]
            values = self.array(row['answer'], row['answer_sha256'])
            n = row['answer_blocks']
            if values.shape != (32, n * 64) or not 1 <= n <= 128:
                raise ValueError('complete target blocks differ')
            pixels = torch.from_numpy(values).float().div(127.5).sub(1)
            self._targets[index] = pixels.reshape(32, n, 64).permute(1, 0, 2)[:, None].contiguous()
        return self._targets[index]

    def references(self, index, *, development=False):
        row = self.development[index] if development else self.train[index]
        paths = row['references'] if development else [dict(path=row['answer'], sha256=row['answer_sha256'])]
        return [torch.from_numpy(self.array(r['path'], r['sha256'])) for r in paths]

    def bindings(self):
        result = {'manifest.json': digest(self.root / 'manifest.json'), 'audit.json': digest(self.root / 'audit.json')}
        for row in self.train:
            for key in ('prompt', 'answer'):
                if digest(self.path(row[key])) != row[key + '_sha256']:
                    raise ValueError('prepared training image changed')
                result[row[key]] = row[key + '_sha256']
        for row in self.development:
            for key in ('normal', 'blank'):
                result[row[key]] = row[key + '_sha256']
            result.update({r['path']: r['sha256'] for r in row['references']})
        for name, value in result.items():
            if digest(self.path(name)) != value:
                raise ValueError('prepared source changed: ' + name)
        return result


class WindowSampler:
    """Fresh corpus permutations, bounded windows and fresh local repeats."""
    def __init__(self, size, *, seed, window=32, repeats=4):
        if size % window or window % 4 or repeats < 1:
            raise ValueError('complete batch-four windows required')
        self.size, self.window, self.repeats = size, window, repeats
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.local_order = torch.randperm(window, generator=self.generator)
        self.start = self.cursor = self.repeat = self.cycle = self.draws = 0

    @property
    def current(self):
        return self.order[self.start:self.start + self.window].tolist()

    def take(self):
        if self.cursor == self.window:
            self.cursor = 0
            self.repeat += 1
            if self.repeat == self.repeats:
                self.repeat = 0
                self.start += self.window
                if self.start == self.size:
                    self.start = 0
                    self.cycle += 1
                    self.order = torch.randperm(self.size, generator=self.generator)
            self.local_order = torch.randperm(self.window, generator=self.generator)
        selected = self.local_order[self.cursor:self.cursor + 4]
        result = self.order[self.start + selected].tolist()
        self.cursor += 4
        self.draws += 4
        return result

    def state_dict(self):
        return dict(size=self.size, window=self.window, repeats=self.repeats,
                    order=self.order.clone(), local_order=self.local_order.clone(),
                    start=self.start, cursor=self.cursor, repeat=self.repeat, cycle=self.cycle,
                    draws=self.draws, generator=self.generator.get_state())

    def load_state_dict(self, state):
        if (state['size'], state['window'], state['repeats']) != (self.size, self.window, self.repeats):
            raise ValueError('window sampler dimensions differ')
        if (not torch.equal(state['order'].sort().values, torch.arange(self.size))
                or not torch.equal(state['local_order'].sort().values, torch.arange(self.window))
                or not 0 <= state['start'] < self.size or state['start'] % self.window
                or not 0 <= state['cursor'] <= self.window or state['cursor'] % 4
                or not 0 <= state['repeat'] < self.repeats or state['cycle'] < 0):
            raise ValueError('invalid serialized permutation or cursor')
        expected = state['cycle'] * self.size * self.repeats + state['start'] * self.repeats + state['repeat'] * self.window + state['cursor']
        if state['draws'] != expected:
            raise ValueError('sampler exposure counters disagree')
        for key in ('order', 'local_order', 'start', 'cursor', 'repeat', 'cycle', 'draws'):
            setattr(self, key, state[key].clone() if isinstance(state[key], torch.Tensor) else state[key])
        self.generator.set_state(state['generator'])


def native_batch(pool, indices, device):
    """Assemble only owning full native keys and causal target histories."""
    selected = [pool[i] for i in indices]
    prefixes = [row[0] for row in selected]
    longest = max(p.length for p in prefixes)
    values = tuple(tuple(torch.cat([torch.nn.functional.pad(p.key_values[layer][kind], (0, 0, 0, longest - p.length))
        for p in prefixes]).to(device) for kind in range(2)) for layer in range(len(prefixes[0].key_values)))
    prefix = NativeImagePrefix(values, torch.cat([p.initial_state for p in prefixes]).to(device),
        torch.cat([p.rotary_offsets for p in prefixes]).to(device),
        valid_lengths=torch.tensor([p.length for p in prefixes], device=device))
    lengths = torch.tensor([len(row[1]) for row in selected], device=device)
    codes = torch.zeros(len(indices), int(lengths.max()), 2, 768, device=device)
    for b, (_, target) in enumerate(selected):
        codes[b, :len(target)] = target.to(device)
    return prefix, codes, lengths
