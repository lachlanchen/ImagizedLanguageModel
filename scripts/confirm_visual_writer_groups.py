"""Evaluate the three fixed image writers on disjoint, previously unused source groups."""
from collections import Counter
import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from scripts import fit_pictured_idioms as first
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_pictured_idiom_alignment import local_counts
from scripts.fit_visual_writer_distillation import DATA, OUT, PRIOR, ROOT
from scripts.prepare_pictured_idioms import BIND, RAW, TAG, WIDTH, Rasterizer

FRESH = OUT / 'fresh'


def prepare(train, validation, initial_pixels):
    raster = Rasterizer()
    packed = {v.tobytes(): i for i, v in enumerate(initial_pixels)}
    require(len(packed) == len(initial_pixels), 'Nonunique physical tile IDs')
    seen = {v.tobytes() for split in (train, validation) for v in split['context']}
    old_groups = set(train['group'])
    rows, targets, groups, tags = [], [], [], []
    counts = Counter()
    source = RAW / 'train.parquet'
    require(sha(source) == BIND['train'], 'Source changed')

    def tiles(text):
        result = []
        for c in text:
            tile = raster.tile(c)
            result.append(packed.setdefault(tile, len(packed)))
        return result

    for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
        for record in batch.to_pylist():
            group = record['idx']
            bucket = hashlib.sha256(str(group).encode()).digest()[0]
            if not 32 <= bucket <= 35:
                continue
            require(group not in old_groups, 'Training group present')
            counts['source_groups'] += 1
            all_tags = [int(s) for passage in record['content'] for s in TAG.findall(passage)]
            answers = dict(zip(all_tags, record['answers']['text'], strict=True))
            for cid, gold in zip(record['answers']['candidate_id'], answers.values(), strict=True):
                require(record['candidates'][cid] == gold, 'Source answer disagrees')
            for passage in record['content']:
                for m in TAG.finditer(passage):
                    tag = int(m[1])
                    answer = answers[tag]
                    if len(answer) != 4 or not all('\u3400' <= c <= '\u9fff' for c in answer):
                        counts['non_four_han'] += 1
                        continue
                    left = TAG.sub('□', passage[:m.start()])[-WIDTH:]
                    right = TAG.sub('□', passage[m.end():])[:WIDTH]
                    try:
                        ids = [0] * (WIDTH-len(left)) + tiles(left+right) + [0] * (WIDTH-len(right))
                        target = tiles(answer)
                    except ValueError:
                        counts['unsupported'] += 1
                        continue
                    require(len(packed) < 65536, 'Packed tile index overflow')
                    x = np.array(ids, dtype=np.uint16)
                    if x.tobytes() in seen:
                        counts['overlap'] += 1
                        continue
                    seen.add(x.tobytes())
                    rows.append(x)
                    targets.append(target)
                    groups.append(group)
                    tags.append(tag)
    dev = dict(context=np.stack(rows), target=np.array(targets, np.uint16), group=np.array(groups, np.int32),
               tag=np.array(tags, np.int32))
    pixels = np.frombuffer(b''.join(packed), dtype=np.uint8).reshape(-1, 128)
    require(np.array_equal(initial_pixels, pixels[:len(initial_pixels)]), 'Original tile IDs changed')
    np.save(FRESH / 'pixels.npy', pixels)
    np.savez(FRESH / 'contexts.npz', **dev)
    write_new(FRESH / 'preparation.json', dict(counts=dict(counts), examples=len(rows), groups=len(set(groups)),
              source_sha256=sha(source), contexts_sha256=sha(FRESH/'contexts.npz'), pixels_sha256=sha(FRESH/'pixels.npy')))
    return dev, pixels


@torch.no_grad()
def main():
    start = time.monotonic()
    first.guard(start)
    require((OUT / 'execution.json').is_file(), 'Both final checkpoints required')
    FRESH.mkdir()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with np.load(DATA/'train.npz') as z:
        train = {k: z[k] for k in z.files}
    with np.load(DATA/'validation.npz') as z:
        val = {'context': z['context']}
    dev, pixels = prepare(train, val, np.load(DATA/'pixels.npy'))
    images = torch.as_tensor(np.unpackbits(pixels, axis=1), device='cuda').bool()
    for name, folder in [('published', PRIOR), ('original_targets', OUT/'original_targets'),
                         ('distilled_targets', OUT/'distilled_targets')]:
        first.OUT = FRESH / name
        first.OUT.mkdir()
        state = torch.load(folder/'weights.pt', weights_only=True, map_location='cpu')
        net = PicturedIdiom(state['landmarks']).cuda().eval()
        net.load_state_dict(state)
        features = visual_input(images, net.landmarks)
        stats = first.compare(net, features, images, train, dev, pixels, np.unique(train['target']), start)
        write_new(first.OUT/'result.json', stats)
        if name == 'published':
            aux = nn.Sequential(nn.Linear(448,384), nn.GELU(), nn.Linear(384,192)).cuda().eval()
            aux.load_state_dict(torch.load(PRIOR/'training_only_aux.pt', weights_only=True))
            options = torch.as_tensor(np.unique(train['target'],axis=0).astype(np.int32),device='cuda')
            keys = F.normalize(aux(features[options].flatten(1)))
            chosen = []
            for batch in np.array_split(dev['context'], (len(dev['context'])+255)//256):
                state = net.read(features[torch.as_tensor(batch.astype(np.int32),device='cuda')])
                chosen.append(options[(F.normalize(state)@keys.T).argmax(1)].cpu().numpy())
            np.savez_compressed(FRESH/'diagnostic_choices.npz', supplied_inventory_selection=np.concatenate(chosen),
                                local_exact=local_counts(train,dev))
        print(name,stats['examples'],stats['complete_exact'],flush=True)
    write_new(FRESH/'execution.json', dict(seconds=time.monotonic()-start, program_sha256=sha(Path(__file__)),
              protocol_sha256=sha(ROOT/'references/visual_writer_fresh_groups_protocol.md')))


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
