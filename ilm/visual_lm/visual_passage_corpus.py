"""Read unique training passage images as complete next-writing examples."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .visual_sequence_corpus import VisualSequenceCorpus


class VisualPassageCorpus(Dataset):
    def __init__(self, manifest, corpus_directory, expected_sha256=None):
        raw = Path(manifest).read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Passage selection changed')
        index = json.loads(raw)
        if index['schema'] != 'visual-training-passages-v1':
            raise ValueError('Unknown visual passage format')
        self.source = VisualSequenceCorpus(corpus_directory, split='train',
                                          expected_report_sha256=index['corpus_sha256'])
        self.rows = index['passages']

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        record = self.source[row['corpus_index']]
        start, end = row['start'], row['start'] + row['length']
        if not 0 <= start < end <= record['prompt_glyphs']:
            raise ValueError('Passage extends beyond its training prompt')
        ink = record['ink'][start:end]
        packed = np.packbits(ink.numpy(), bitorder='big')
        if hashlib.sha256(packed.tobytes()).hexdigest() != row['ink_sha256']:
            raise ValueError('Passage pixels changed')
        return {'ink': ink, 'prompt_glyphs': 0}
