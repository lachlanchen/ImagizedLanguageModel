"""Isolate source order and test context-specific choice between real pictured glyphs."""
import fcntl
import json
import time
from collections import defaultdict

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, START
from scripts.compare_visual_edit_contrast import ARMS, WORK, restore
from scripts.train_visual_edit_translation import OUT, ROOT, guard, load, padded, require, sha, write_new


def path_score(head, states, bits):
    logits = head.teacher(states, bits)
    return logits.log_softmax(-1).gather(-1, nibbles(bits).unsqueeze(-1)).squeeze(-1).sum(-1)


def match(keys, targets, chapters):
    available = defaultdict(list)
    matched = []
    for j, key in enumerate(keys):
        pool = available[tuple(key)]
        i = next((k for k, row in enumerate(pool)
                  if targets[row] != targets[j] and chapters[row] != chapters[j]), None)
        if i is None:
            pool.append(j)
        else:
            matched.append((pool.pop(i), j))
    return np.asarray(matched, dtype=np.int64).reshape(-1, 2)


@torch.inference_mode()
def measure(arm, records, tiles):
    start = time.monotonic()
    folder = WORK / arm
    net = restore(folder / 'editor.safetensors').eval()
    ink = torch.as_tensor(np.unpackbits(tiles, axis=1), device='cuda').bool()
    features = visual_input(ink, net.landmarks)
    prompt = load(OUT / 'preparation.json')['question_tiles'][0][0]
    question = torch.tensor(prompt, device='cuda')[None]
    totals = defaultdict(lambda: np.zeros((2, 3)))
    states, targets, keys, chapters, exacts = [], [], [], [], []
    for first in range(0, len(records), 32):
        guard(start)
        samples = records[first:first+32]
        x, lengths = padded([r['x'] for r in samples])
        a, steps = padded([r['a'] for r in samples])
        y, _ = padded([r['o'] for r in samples])
        x, lengths, a, steps, y = [torch.as_tensor(v, device='cuda').long() for v in (x,lengths,a,steps,y)]
        source_pos = torch.arange(x.shape[1], device='cuda')[None]
        valid = source_pos < lengths[:, None]
        pos = torch.arange(a.shape[1], device='cuda')[None]
        active = pos < steps[:, None]
        advances = ((a == COPY) | (a == REPLACE) | (a == DROP)) & active
        cursor = torch.nn.functional.pad(advances.cumsum(1)[:, :-1], (1,0))
        last = torch.where((a <= INSERT) & active, pos+1, 0).cummax(1).values
        prev = torch.nn.functional.pad(y,(1,0)).gather(1,torch.nn.functional.pad(last[:, :-1],(1,0)))
        prev_action = torch.nn.functional.pad(a[:, :-1],(1,0),value=START)
        draw = ((a == INSERT) | (a == REPLACE)) & active
        row_ids = torch.where(draw)[0]
        if not len(row_ids):
            continue
        target = y[draw]
        gold = ink[target]
        source_ids = torch.nn.functional.pad(x,(0,1),value=-1).gather(1,cursor)
        source_ids = source_ids.masked_fill(cursor == lengths[:,None], -1)
        keys.extend(torch.stack((source_ids[draw],prev[draw],a[draw],prev_action[draw]), dim=-1).cpu().tolist())
        chapters.extend(samples[i]['g'] for i in row_ids.tolist())
        targets.extend(target.tolist())
        q = question.expand(len(x),-1)
        for reverse in (False, True):
            source = x.gather(1, (lengths[:,None]-1-source_pos).clamp_min(0)) if reverse else x
            positions = torch.where(cursor == lengths[:,None],cursor,lengths[:,None]-1-cursor) if reverse else cursor
            if reverse:
                changed = torch.nn.functional.pad(source,(0,1),value=-1).gather(1,positions)
                require(torch.equal(changed[draw & (cursor < lengths[:,None])],
                                    source_ids[draw & (cursor < lengths[:,None])]), 'Source glyph changed')
            memory, initial = net.encode(features[source],valid,features[q],torch.ones_like(q,dtype=torch.bool))
            h = net.teacher(memory,initial,positions,features[prev],prev_action)[draw]
            bits = gold[:,net.landmarks].float()
            logp = path_score(net.draw,h,bits)
            prediction = net.motor(net.draw.choose(h)) > 0
            exact = (prediction == gold).all(-1)
            if not reverse:
                states.append(h.cpu())
                exacts.extend(exact.tolist())
            values = torch.stack((torch.ones_like(logp),exact.float(),-logp),dim=-1).double()
            sums = values.new_zeros(len(x),3).index_add_(0,row_ids,values).cpu().numpy()
            for r, v in zip(samples,sums):
                totals[r['g']][int(reverse)] += v

    summary = np.stack(list(totals.values()))
    ordered, reversed_order = summary.sum(axis=0)
    original = load(folder/'fresh_teacher.json')
    require((int(ordered[0]),int(ordered[1])) == (original['source_draw_positions'],original['emitted_glyph_correct']),
            'Original physical history result failed to replay')
    draws = np.random.default_rng(414).integers(len(summary),size=(2000,len(summary)))
    paired = summary[draws].sum(axis=1)
    diff = (paired[:,0,1] - paired[:,1,1])/paired[:,0,0]
    pairs = match(keys,targets,chapters)
    all_states = torch.cat(states)
    all_targets = torch.as_tensor(targets)
    correct = np.asarray(exacts, bool)
    wins = ties = 0
    for batch in np.array_split(pairs, max(1, (len(pairs)+255)//256)):
        h = all_states[batch].cuda()
        bits = ink[all_targets[batch].cuda()][:,:,net.landmarks].float()
        own = path_score(net.draw,h.flatten(0,1),bits.flatten(0,1)).reshape(-1,2).sum(1)
        swapped = path_score(net.draw,h.flatten(0,1),bits.flip(1).flatten(0,1)).reshape(-1,2).sum(1)
        wins += int((own > swapped + 1e-5).sum())
        ties += int(((own-swapped).abs() <= 1e-5).sum())
    result = dict(draws=int(ordered[0]),original_exact=int(ordered[1]),reversed_exact=int(reversed_order[1]),
                  original_nll=ordered[2]/ordered[0],reversed_nll=reversed_order[2]/reversed_order[0],
                  order_accuracy_gain_pp=(100*np.quantile(diff,[.025,.975])).tolist(),
                  matched_pairs=len(pairs),correct_assignment=wins,tied_assignment=ties,
                  both_drawings_correct=int(correct[pairs].all(1).sum()),either_drawing_correct=int(correct[pairs].any(1).sum()),
                  seconds=time.monotonic()-start)
    write_new(folder/'source_order.json',result)
    print(arm,result,flush=True)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not load(WORK/'comparison.json')['fresh']['qualified'], 'Diagnostic requires the registered failure')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        prep = load(WORK/'preparation.json')
        require(sha(WORK/'fresh.jsonl') == prep['fresh_sha256'] and sha(WORK/'pixels.npy') == prep['pixels_sha256'],
                'Fresh roster changed')
        rows = [json.loads(line) for line in (WORK/'fresh.jsonl').read_text().splitlines()]
        for model in ARMS:
            require(sha(WORK/model/'editor.safetensors') == load(WORK/model/'fit.json')['checkpoint_sha256'], 'Checkpoint changed')
            measure(model,[r for r in rows if r['x']],np.load(WORK/'pixels.npy'))
