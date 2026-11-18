"""Train matched sentence editors and score their actual output images on untouched chapters."""
import argparse
import fcntl
import hashlib
import json
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from scripts.evaluate_visual_edit_joint import teacher
from scripts.evaluate_visual_edit_translation import ci, evaluate, local_program
from scripts.prepare_edit_contrast_chapters import WORK
from scripts.train_visual_edit_translation import ROOT, examples, guard, inputs, load, require, scored, sha, write_new

PARENT = ROOT / 'models/visual_edit_zh/editor.safetensors'
ARMS = ('likelihood', 'contrast')


def restore(path):
    state = load_file(str(path))
    model = FertilityEditor(state['landmarks'], len(state['fertility.2.bias']) - 1).cuda()
    model.load_state_dict(state)
    return model


def contrast_loss(head, states, target, log_frequency=None):
    selected = torch.randperm(len(states), device=states.device)[:256]
    truth = target[selected]
    positions = torch.arange(len(truth), device=states.device)
    offsets = torch.arange(8, device=states.device)
    candidates = truth[(positions[:, None] + offsets) % len(truth)]
    queries = states[selected, None].expand(-1, len(offsets), -1)
    bits = candidates.flatten(0, 1)
    logits = head.teacher(queries.flatten(0, 1), bits)
    logp = logits.log_softmax(-1).gather(-1, nibbles(bits).unsqueeze(-1)).squeeze(-1)
    scores = logp.sum(1).reshape(len(truth), len(offsets))
    if log_frequency is not None:
        scores = scores - log_frequency(candidates)
    duplicates = (candidates == truth[:, None]).all(-1)
    duplicates[:, 0] = False
    return -scores.masked_fill(duplicates, -torch.inf).log_softmax(1)[:, 0].mean()


def fit(arm, tiles, columns, lengths, source_lengths, *, log_frequency=None):
    start = time.monotonic()
    guard(start)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(7209)
    rng = np.random.default_rng(7209)
    net = restore(PARENT)
    net.requires_grad_(False)
    for module in (net.step, net.action, net.draw):
        module.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.0003, weight_decay=.01)
    ink = torch.as_tensor(np.unpackbits(tiles, axis=1), device='cuda').bool()
    features = visual_input(ink, net.landmarks)
    history, schedule = [], hashlib.sha256()
    factor = float(arm != 'likelihood')

    def auxiliary(head, states, target):
        return factor * contrast_loss(head, states, target, log_frequency)

    for epoch in range(4):
        batches = []
        for block in np.array_split(rng.permutation(len(lengths)), 32):
            block = block[np.argsort(lengths[block])]
            batches.extend(block[i:i + 128] for i in range(0, len(block), 128))
        rng.shuffle(batches)
        aggregate = 0.
        for index, rows in enumerate(batches):
            schedule.update(rows.astype('<i4').tobytes())
            loss, _ = scored(net, features, ink, columns, torch.as_tensor(rows, device='cuda'),
                             int(source_lengths[rows].max()), int(lengths[rows].max()),
                             draw_auxiliary=auxiliary)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2., error_if_nonfinite=True)
            optimizer.step()
            aggregate += float(loss.detach()) * len(rows)
            if index % 128 == 0:
                guard(start)
        history.append(dict(epoch=epoch + 1, loss=aggregate / len(lengths), updates=len(batches)))
        print(arm, history[-1], flush=True)
    parent = load_file(str(PARENT))
    state = {key: value.detach().cpu().contiguous() for key, value in net.state_dict().items()}
    trainable = {name for name, p in net.named_parameters() if p.requires_grad}
    require(all(torch.equal(value, parent[key]) for key, value in state.items() if key not in trainable),
            'Frozen visual renderer, encoder or insertion count changed')
    folder = WORK / arm
    folder.mkdir()
    checkpoint = folder / 'editor.safetensors'
    save_file(state, str(checkpoint))
    write_new(folder / 'fit.json', dict(parent=sha(PARENT), parameters=sum(p.numel() for p in params),
              history=history, schedule=schedule.hexdigest(), seconds=time.monotonic() - start,
              peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20, checkpoint_sha256=sha(checkpoint)))


def score(arm, tiles, train, program, fresh):
    start = time.monotonic()
    folder = WORK / arm
    if arm == 'parent':
        folder.mkdir()
        checkpoint = PARENT
    else:
        checkpoint = folder / 'editor.safetensors'
        require(sha(checkpoint) == load(folder / 'fit.json')['checkpoint_sha256'], 'Fit weights changed')
    net = restore(checkpoint).eval()
    for split in (('fresh',) if arm == 'parent' else ('heldout', 'fresh')):
        records = fresh if split == 'fresh' else inputs('heldout')
        write_new(folder / (split + '_teacher.json'), teacher(net, tiles, records=records))
        evaluate(split, net, tiles, program, train, start, folder / split, records=records)
    write_new(folder / 'evaluation.json', dict(seconds=time.monotonic()-start))


def compare():
    result = {}
    for name in ('heldout', 'fresh'):
        rows = load(WORK / 'contrast' / name / 'rows.json')
        controls = load(WORK / 'likelihood' / name / 'rows.json')
        for row, control in zip(rows, controls, strict=True):
            require(all(row[k] == control[k] for k in ('index','group','length','copy','local')), 'Row mismatch')
            row['matched'] = control['translation']
        report = load(WORK / 'contrast' / name / 'report.json')
        interval = ci(rows, 'matched')
        result[name] = dict(matched_ci_pp=interval, local_ci_pp=report['improvement_ci_pp']['local'],
                            original_qualification=report['pass'],
                            qualified=bool(report['pass'] and interval[0] > 0))
    result['schedule_match'] = load(WORK/'contrast/fit.json')['schedule'] == load(WORK/'likelihood/fit.json')['schedule']
    require(result['schedule_match'], 'Training batch orders differ')
    write_new(WORK / 'comparison.json', result)
    print(result, flush=True)


def run(phase):
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    require('4090' in torch.cuda.get_device_name(), 'Expected a 4090-class GPU')
    require(sha(PARENT) == '705f020c14a1d2b81f55c4b4fc2650cff94e9710b5ded2ffd7251c618a0e025e', 'Parent changed')
    prep = load(WORK / 'preparation.json')
    require(sha(WORK/'pixels.npy') == prep['pixels_sha256'] and sha(WORK/'fresh.jsonl') == prep['fresh_sha256'],
            'New roster or pixels changed')
    train = inputs('train')
    tiles = np.load(WORK/'pixels.npy')
    if phase == 'fit':
        data, lengths, source_lengths = examples(train)
        for arm in ARMS:
            fit(arm, tiles, data, lengths, source_lengths)
    else:
        program = local_program(train)
        fresh = [json.loads(line) for line in (WORK/'fresh.jsonl').read_text().splitlines()]
        for arm in ('parent', *ARMS):
            score(arm, tiles, train, program, fresh)
        compare()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['fit','evaluate'])
    args = parser.parse_args()
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.phase)
