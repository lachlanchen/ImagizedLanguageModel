"""Freeze physically disjoint classical Chinese sentence blocks for the diversity comparison."""
import hashlib
import heapq
import json
import tarfile
from pathlib import Path

import numpy as np

from scripts.prepare_edit_contrast_chapters import WORK as PRIOR
from scripts.prepare_visual_edit_translation import BASE, OUT, Rasterizer, digest, edit_trace, pairs
from scripts.train_visual_edit_translation import ROOT, bank, load, require, sha, write_new

WORK = ROOT / 'artifacts/visual_edit_diversity_v1'
SHARDS = [f'shard_{i}' for i in range(4)]
COHORTS = ('discovery', 'replication')


def records(path):
    with path.open(encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def read_split(name, prep):
    path = WORK / f'{name}.jsonl'
    require(sha(path) == prep['splits'][name]['sha256'], f'Changed split {name}')
    yield from records(path)


def prepare():
    archive = BASE / 'archive.tar.gz'
    original = load(OUT / 'preparation.json')
    require(sha(archive) == original['archive_sha256'], 'Archive changed')
    renderer = Rasterizer()
    old = bank()
    training_groups, excluded_groups, texts, images = set(), set(), set(), set()

    def exclude(path, tiles, checksum, groups):
        require(sha(path) == checksum, 'Previous roster changed')
        for row in records(path):
            groups.add(row['g'])
            texts.add(row['source'])
            if row['x']:
                images.add(hashlib.sha256(tiles[row['x']].tobytes()).digest())
            else:
                try:
                    raw = b''.join(renderer.tile(c) for c in row['source'])
                except ValueError:
                    continue
                images.add(hashlib.sha256(raw).digest())

    for name in ('train', 'heldout', 'confirmation'):
        exclude(OUT / f'{name}.jsonl', old, original['partitions'][name]['sha256'],
                training_groups if name == 'train' else excluded_groups)
    for folder in (PRIOR, PRIOR / 'confirmation_roster'):
        spec = load(folder / 'preparation.json')
        require(sha(folder / 'pixels.npy') == spec['pixels_sha256'], 'Previous pixels changed')
        exclude(folder / 'fresh.jsonl', np.load(folder / 'pixels.npy'), spec['fresh_sha256'], excluded_groups)
    require(not training_groups & excluded_groups, 'Prior chapter overlap')

    pools = [[], [], []]
    capacities = (8192, 8192, 250000)
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = Path(member.name)
            if len(path.parts) < 4 or path.parts[1] != '双语数据' or path.name != 'bitext.txt':
                continue
            group = '/'.join(path.parts[2:-1])
            pool = 2 if group in training_groups else digest('diversity-chapter-v1/' + group)[0] % 2
            for index, (source, target) in enumerate(pairs(tar.extractfile(member).read())):
                if source in texts or len(source) > 96 or (pool == 2 and len(target) > 128):
                    continue
                priority = int.from_bytes(digest(f'diversity-row-v1/{group}/{index}')[:8], 'big')
                choice = (-priority, group, index, source, target)
                heap = pools[pool]
                if len(heap) < capacities[pool]:
                    heapq.heappush(heap, choice)
                elif choice > heap[0]:
                    heapq.heapreplace(heap, choice)

    ids = {bytes(tile): i for i, tile in enumerate(old)}
    stats = {}
    WORK.mkdir()
    for pool, heap in enumerate(pools):
        iterator = iter(sorted(heap, reverse=True))
        names = SHARDS if pool == 2 else [COHORTS[pool]]
        for name in names:
            path = WORK / f'{name}.jsonl'
            limit = 48000 if pool == 2 else 2048
            count = supported = source_cells = target_cells = 0
            groups = set()
            with path.open('x', encoding='utf8') as out:
                for _, group, index, source, target in iterator:
                    if source in texts:
                        continue
                    try:
                        source_tiles = [renderer.tile(c) for c in source]
                    except ValueError:
                        source_tiles = []
                    fingerprint = hashlib.sha256(b''.join(source_tiles)).digest()
                    if source_tiles and fingerprint in images:
                        continue
                    try:
                        target_tiles = [renderer.tile(c) for c in target]
                    except ValueError:
                        target_tiles = []
                    if pool == 2 and not (source_tiles and target_tiles):
                        continue
                    texts.add(source)
                    if source_tiles:
                        images.add(fingerprint)
                    x = y = []
                    if source_tiles and target_tiles:
                        x, y = ([ids.setdefault(tile, len(ids)) for tile in seq]
                                for seq in (source_tiles, target_tiles))
                    a, o = edit_trace(x, y) if x else ([], [])
                    row = dict(g=group, i=index, x=x, y=y, a=a, o=o, source=source, target=target)
                    out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                    count += 1
                    supported += bool(x)
                    groups.add(group)
                    source_cells += len(x)
                    target_cells += len(y)
                    if count == limit:
                        break
            require(count == limit, f'Insufficient distinct sources for {name}: {count}')
            stats[name] = dict(rows=count, pictured=supported, groups=len(groups), previous_eval_chapters=len(groups & excluded_groups), sha256=sha(path),
                               source_cells=source_cells, target_cells=target_cells)
            print(name, stats[name], flush=True)
    np.save(WORK / 'pixels.npy', np.frombuffer(b''.join(ids), np.uint8).reshape(-1, 128))
    write_new(WORK / 'preparation.json', dict(splits=stats, archive=original['archive_sha256'],
              pixels_sha256=sha(WORK / 'pixels.npy'), training_chapters=len(training_groups),
              excluded_evaluation_chapters=len(excluded_groups)))


if __name__ == '__main__':
    prepare()
