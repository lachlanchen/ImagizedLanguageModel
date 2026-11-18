"""Measure which different training passages become identical after summing their ink."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def measure(directory):
    records = (directory / 'train.jsonl').read_bytes()
    bank_path = directory / 'pixels.npy'
    receipt = json.loads((directory / 'preparation.json').read_text())
    if (hashlib.sha256(records).hexdigest() != receipt['partitions']['train']['sha256']
            or hashlib.sha256(bank_path.read_bytes()).hexdigest() != receipt['pixels_sha256']):
        raise ValueError('Prepared training data differ from the experiment')
    tiles = np.unpackbits(np.load(bank_path), axis=1)
    groups = defaultdict(list)
    examples = [json.loads(line) for line in records.splitlines()]
    for index, row in enumerate(examples):
        # Preserve exact integer sums. A 96-cell source cannot overflow uint8.
        if not 0 < len(row['x']) <= 96:
            raise ValueError('Unexpected source length')
        planes = tiles[row['x']].sum(axis=0, dtype=np.uint8)
        groups[(len(row['x']), planes.tobytes())].append(index)

    indistinguishable = [group for group in groups.values() if len(group) > 1]
    translation_errors = 0
    copy_errors = 0
    for group in indistinguishable:
        sources = Counter(tuple(examples[i]['x']) for i in group)
        targets = Counter(tuple(examples[i]['y']) for i in group)
        copy_errors += len(group) - max(sources.values())
        translation_errors += len(group) - max(targets.values())
    return dict(pairs=len(examples), distinct_length_and_sum=len(groups),
                collision_groups=len(indistinguishable), collision_rows=sum(map(len, indistinguishable)),
                minimum_exact_copy_errors=copy_errors, minimum_exact_reference_errors=translation_errors,
                example_indices=indistinguishable[:5], train_sha256=hashlib.sha256(records).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = measure(args.directory)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(result)
