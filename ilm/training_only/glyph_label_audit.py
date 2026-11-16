"""Lossless source-label accounting; no classifier, teacher or model execution."""
from collections import Counter
import hashlib

import numpy as np


def verify_record(packed, text, prompt_length, actual_prompt_length, renderer):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.shape != (len(text), 128):
        raise ValueError('Keep every original physical cell and source character')
    if not 0 <= prompt_length <= len(text) or actual_prompt_length != prompt_length:
        raise ValueError('Source/pixel prompt boundary differs')
    expected = np.frombuffer(b''.join(renderer.tile(c) for c in text), dtype=np.uint8).reshape(-1, 128)
    return int(np.count_nonzero(np.any(packed != expected, axis=1)))


def add_pairs(bank, packed, text):
    """Count every physical-mask/codepoint pair without overwriting aliases."""
    if packed.dtype != np.uint8 or packed.shape != (len(text), 128):
        raise ValueError('Label extent differs from the actual cell extent')
    pairs = np.empty(len(text), dtype=[('mask', 'V128'), ('codepoint', '<u4')])
    pairs['mask'] = np.ascontiguousarray(packed).view('V128').reshape(-1)
    pairs['codepoint'] = [ord(c) for c in text]
    unique, counts = np.unique(pairs, return_counts=True)
    for pair, count in zip(unique, counts):
        key = pair['mask'].tobytes()
        bank.setdefault(key, Counter())[int(pair['codepoint'])] += int(count)


def inventory_summary(bank):
    aliases = []; white_nonspace = []
    for mask, labels in sorted(bank.items()):
        row = dict(mask_sha256=hashlib.sha256(mask).hexdigest(),
                   labels=[dict(codepoint=cp, character=chr(cp), occurrences=count)
                           for cp, count in sorted(labels.items())])
        if any(mask) and len(labels) > 1:
            aliases.append(row)
        elif not any(mask):
            white_nonspace.extend(v for v in row['labels'] if not v['character'].isspace())
    return dict(physical_masks=len(bank), raw_codepoints=len({cp for labels in bank.values() for cp in labels}),
        glyph_occurrences=sum(sum(v.values()) for v in bank.values()),
        nonwhite_aliases=aliases, white_nonspace_labels=white_nonspace,
        context_free_nonwhite_labels_unique=not aliases,
        physically_white_labels_only_whitespace=not white_nonspace)
