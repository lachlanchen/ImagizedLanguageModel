"""Reserve a new chapter-disjoint roster for a comparison of whole-glyph objectives."""
import argparse
import heapq
import json
import tarfile
from pathlib import Path

import numpy as np

from scripts.prepare_visual_edit_translation import BASE, OUT, Rasterizer, digest, edit_trace, pairs
from scripts.train_visual_edit_translation import ROOT, bank, inputs, load, require, sha, write_new

WORK = ROOT / 'artifacts/visual_edit_contrast_v1'


def main(confirmation=False):
    destination = WORK / "confirmation_roster" if confirmation else WORK
    tag = "contrast-confirmation-v1" if confirmation else "contrast-fresh-v1"
    archive = BASE / 'archive.tar.gz'
    require(sha(archive) == load(OUT / 'preparation.json')['archive_sha256'], 'Archive changed')
    groups, seen, texts = set(), set(), set()
    for split in ('train', 'heldout', 'confirmation'):
        for row in inputs(split):
            groups.add(row['g'])
            texts.add(row['source'])
            if row['x']:
                seen.add(tuple(row['x']))
    if confirmation:
        require(sha(WORK / 'fresh.jsonl') == load(WORK / 'preparation.json')['fresh_sha256'], 'First fresh roster changed')
        fresh_bank = np.load(WORK / 'pixels.npy')
        fresh_images = set()
        for line in (WORK/'fresh.jsonl').open():
            row = json.loads(line)
            groups.add(row['g'])
            texts.add(row['source'])
            if row['x']:
                fresh_images.add(fresh_bank[row['x']].tobytes())
    heap = []
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            path = Path(member.name)
            if len(path.parts) < 4 or path.parts[1] != '双语数据' or path.name != 'bitext.txt':
                continue
            group = '/'.join(path.parts[2:-1])
            if group in groups:
                continue
            for index, (source, target) in enumerate(pairs(tar.extractfile(member).read())):
                if len(source) > 96 or source in texts:
                    continue
                priority = int.from_bytes(digest(f'{tag}/{group}/{index}')[:8], 'big')
                entry = (-priority, group, index, source, target)
                if len(heap) < 8192:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    ids = {bytes(bitmap): i for i, bitmap in enumerate(bank())}
    renderer = Rasterizer()
    rows = []
    for _, group, index, source, target in sorted(heap, reverse=True):
        if source in texts:
            continue
        try:
            x, y = [[ids.setdefault(renderer.tile(c), len(ids)) for c in text] for text in (source, target)]
        except ValueError:
            x, y = [], []
        if x and tuple(x) in seen:
            continue
        if confirmation and x and b''.join(renderer.tile(c) for c in source) in fresh_images:
            continue
        seen.add(tuple(x))
        texts.add(source)
        actions, outputs = edit_trace(x, y) if x else ([], [])
        rows.append(dict(g=group, i=index, source=source, target=target, x=x, y=y, a=actions, o=outputs))
        if len(rows) == 2048:
            break
    require(len(rows) == 2048 and not groups.intersection(r['g'] for r in rows), 'Fresh chapter coverage')
    destination.mkdir()
    file = destination / 'fresh.jsonl'
    with file.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    np.save(destination / 'pixels.npy', np.frombuffer(b''.join(ids), np.uint8).reshape(-1, 128))
    report = dict(rows=len(rows), groups=len({r['g'] for r in rows}), pictured=sum(bool(r['x']) for r in rows),
                  excluded_chapters=len(groups), fresh_sha256=sha(file), pixels_sha256=sha(destination / 'pixels.npy'))
    write_new(destination / 'preparation.json', report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation', action='store_true')
    main(parser.parse_args().confirmation)
