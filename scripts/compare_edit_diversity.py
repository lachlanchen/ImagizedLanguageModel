"""Compare independent sentence coverage at fixed passage exposures and update count."""
import argparse
import fcntl
import hashlib
import itertools
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE
from scripts.audit_edit_contrast_streams import recount
from scripts.compare_visual_edit_contrast import PARENT, restore
from scripts.evaluate_visual_edit_joint import teacher
from scripts.evaluate_visual_edit_translation import ci, evaluate, local_answer, local_program
from scripts.prepare_edit_diversity import COHORTS, SHARDS, WORK, read_split, records
from scripts.train_visual_edit_translation import OUT, ROOT, examples, guard, load, require, scored, sha, write_new

ARMS = {'repeat': [SHARDS[0]] * 4, 'diverse': SHARDS}
PARENT_HASH = '705f020c14a1d2b81f55c4b4fc2650cff94e9710b5ded2ffd7251c618a0e025e'


def fingerprint(net):
    digest = hashlib.sha256()
    for _, tensor in sorted(net.state_dict().items()):
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit(arm, prep, tiles):
    start = time.monotonic()
    guard(start)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(7209)
    rng = np.random.default_rng(7209)
    net = restore(PARENT)
    net.requires_grad_(False)
    for module in (net.step, net.action, net.draw):
        module.requires_grad_(True)
    trainable = {k: p for k, p in net.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=.0003, weight_decay=.01)
    ink = torch.as_tensor(np.unpackbits(tiles, axis=1), device='cuda').bool()
    features = visual_input(ink, net.landmarks)
    folder = WORK / arm
    folder.mkdir()
    history = []
    for shard in ARMS[arm]:
        cols, lengths, widths = examples(list(read_split(shard, prep)))
        batches = []
        for block in np.array_split(rng.permutation(len(lengths)), 32):
            block = block[np.argsort(lengths[block])]
            batches.extend(block[i:i+128] for i in range(0, len(block), 128))
        rng.shuffle(batches)
        total = 0.
        for index, ids in enumerate(batches):
            loss, _ = scored(net, features, ink, cols, torch.as_tensor(ids, device='cuda'),
                             int(widths[ids].max()), int(lengths[ids].max()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable.values(), 2., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            if index % 128 == 0:
                guard(start)
        row = dict(shard=shard, loss=total/len(lengths), updates=len(batches), presentations=len(lengths),
                   steps=int(lengths.sum()), source_cells=int(widths.sum()), state=fingerprint(net))
        require(len(lengths) == 60000 and len(batches) == 480, 'Unexpected training budget')
        history.append(row)
        del cols
        print(arm, row, flush=True)
    state = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    parent = load_file(str(PARENT))
    require(all(torch.equal(v, parent[k]) for k, v in state.items() if k not in trainable), 'Frozen weights changed')
    checkpoint = folder / 'editor.safetensors'
    save_file(state, str(checkpoint))
    write_new(folder/'fit.json', dict(history=history, parameters=sum(p.numel() for p in trainable.values()),
              total_parameters=sum(p.numel() for p in net.parameters()), parent=sha(PARENT),
              seconds=time.monotonic()-start, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
              checkpoint_sha256=sha(checkpoint), device=torch.cuda.get_device_name()))


def local_control(training, cohorts):
    """Keep sufficient counts only for keys on which this fixed evaluation queries the rule."""
    contexts, golds = set(), set()
    for row in itertools.chain.from_iterable(cohorts.values()):
        golds.add(tuple(row['y']))
        padded = [-1, *row['x'], -2, -3]
        contexts.update(zip(padded, padded[1:], padded[2:]))
    unigram = {key[1] for key in contexts}
    counts = [defaultdict(Counter), defaultdict(Counter)]
    observed = set()
    for row in training:
        gold = tuple(row['y'])
        if gold in golds:
            observed.add(gold)
        source = row['x']
        blocks = [[] for _ in range(len(source)+1)]
        cursor = 0
        for action, tile in zip(row['a'], row['o'], strict=True):
            if action <= INSERT:
                blocks[cursor].append(tile)
            if action in (COPY, REPLACE, DROP):
                cursor += 1
        padded = [-1, *source, -2, -3]
        for block, key in zip(blocks, zip(padded, padded[1:], padded[2:]), strict=True):
            output = tuple(block)
            if key[1] in unigram:
                counts[0][key[1]][output] += 1
            if key in contexts:
                counts[1][key][output] += 1
    program = [{k: (v.most_common(1)[0][0], sum(v.values())) for k, v in table.items()} for table in counts]
    return program, [{'y': g} for g in observed]


def score(arm, prep, tiles, cohorts):
    start = time.monotonic()
    guard(start)
    original = OUT / 'train.jsonl'
    require(sha(original) == load(OUT/'preparation.json')['partitions']['train']['sha256'], 'Changed parent corpus')
    test = list(itertools.islice(records(original), 160))
    control, _ = local_control(test, cohorts)
    full = local_program(test)
    require(all(local_answer(r['x'], control) == local_answer(r['x'], full)
                for rows in cohorts.values() for r in rows), 'Restricted-count control differs from original')
    new_data = (read_split(s, prep) for s in dict.fromkeys(ARMS[arm]))
    program, seen = local_control(itertools.chain(records(original), *new_data), cohorts)
    folder = WORK / arm
    path = folder/'editor.safetensors'
    require(sha(path) == load(folder/'fit.json')['checkpoint_sha256'], 'Weights changed')
    net = restore(path).eval()
    mapping = {bytes(v): i for i, v in enumerate(tiles)}
    for name, rows in cohorts.items():
        write_new(folder / f'{name}_teacher.json', teacher(net, tiles, records=rows))
        evaluate(name, net, tiles, program, seen, start, folder/name, records=rows)
        write_new(folder / f'{name}_recount.json', recount(folder/name, rows, tiles, mapping))
    guard(start)


def compare():
    report = {}
    for name in COHORTS:
        paired = load(WORK/'diverse'/name/'rows.json')
        repeat = load(WORK/'repeat'/name/'rows.json')
        for row, control in zip(paired, repeat, strict=True):
            require(all(row[k] == control[k] for k in ('index', 'group', 'length', 'copy')), 'Pair alignment')
            row['repeat'] = control['translation']
        interval = ci(paired, 'repeat')
        scored_report = load(WORK/'diverse'/name/'report.json')
        report[name] = dict(improvement_vs_repeat_pp=interval,
                            pass_all=bool(interval[0] > 0 and scored_report['pass']))
    first = [load(WORK/arm/'fit.json')['history'][0]['state'] for arm in ARMS]
    report['same_first_epoch'] = first[0] == first[1]
    report['qualified'] = all(report[name]['pass_all'] for name in COHORTS)
    write_new(WORK/'comparison.json', report)
    print(report, flush=True)


def run(phase):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    require('4090' in torch.cuda.get_device_name() and sha(PARENT) == PARENT_HASH, 'Device or parent changed')
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'pixels.npy') == prep['pixels_sha256'], 'Image bank changed')
    tiles = np.load(WORK/'pixels.npy')
    if phase == 'fit':
        for arm in ARMS:
            fit(arm, prep, tiles)
    else:
        cohorts = {name: list(read_split(name, prep)) for name in COHORTS}
        require(not {r['g'] for r in cohorts[COHORTS[0]]} & {r['g'] for r in cohorts[COHORTS[1]]}, 'Chapter collision')
        for arm in ARMS:
            score(arm, prep, tiles, cohorts)
        compare()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'evaluate'))
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(parser.parse_args().phase)
