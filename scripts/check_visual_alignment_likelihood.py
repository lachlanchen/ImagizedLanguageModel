"""Check alignment probabilities and neural gradients on real short Chinese image pairs."""
import math

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.visual_alignment import alignment_log_likelihood
from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, START, STOP
from scripts.compare_visual_edit_contrast import PARENT
from scripts.prepare_edit_diversity import WORK, read_split
from scripts.train_visual_edit_translation import OUT, ROOT, load, require, sha, write_new


def path_sum(insert, drop, replace, copy, same, stop):
    """Reference enumeration explicitly distinguishes copying and drawing identical ink."""
    n, m = replace.shape
    paths = []

    def follow(i, j, weight):
        if i == n and j == m:
            paths.append(weight + stop)
            return
        if j < m:
            follow(i, j+1, weight + insert[i, j])
        if i < n:
            follow(i+1, j, weight + drop[i, j])
        if i < n and j < m:
            follow(i+1, j+1, weight + replace[i, j])
            if same[i, j]:
                follow(i+1, j+1, weight + copy[i, j])
    follow(0, 0, stop.new_zeros(()))
    return torch.stack(paths).logsumexp(0), len(paths)


def check(net, ink, q, row):
    x, y = ink[row['x']], ink[row['y']]
    n, m = len(x), len(y)
    fx, fy = [visual_input(v, net.landmarks).double() for v in (x, y)]
    memory, initial = net.encode(fx[None], torch.ones(1, n, dtype=torch.bool),
                                 q[None], torch.ones(1, len(q), dtype=torch.bool))
    # A single neural state per output prefix. Past edit decisions are unobserved.
    inputs = net.decoder_inputs(torch.zeros(1, m, initial.shape[-1], dtype=torch.double),
                                initial[:, None].expand(-1, m, -1), fy[None],
                                torch.full((1, m), START))
    prefix, _ = net.step(inputs, initial[None])
    history = torch.cat((initial[:, None], prefix), dim=1)[0]
    state = torch.tanh(memory[0, :, None] + history[None])
    eof = torch.arange(n+1)[:, None] == n
    valid = torch.ones(n+1, m+1, 5, dtype=torch.bool)
    valid[..., STOP] = eof
    for k in (COPY, REPLACE, DROP):
        valid[..., k] = ~eof
    action = net.action(state).masked_fill(~valid, -torch.inf).log_softmax(-1)
    bits = y[:, net.landmarks].double()[None].expand(n+1, -1, -1).reshape((n+1)*m, -1)
    draw = net.draw.teacher(state[:, :-1].reshape((n+1)*m, -1), bits)
    log_glyph = draw.log_softmax(-1).gather(-1, nibbles(bits).unsqueeze(-1)).sum(1).reshape(n+1, m)
    insert = action[:, :-1, INSERT] + log_glyph
    drop = action[:-1, :, DROP]
    replace = action[:-1, :-1, REPLACE] + log_glyph[:-1]
    copy = action[:-1, :-1, COPY]
    # The observed drawing is its physical code; copy must match that observation too.
    same = (x[:, None, net.landmarks] == y[None, :, net.landmarks]).all(-1)
    require(torch.equal(same, (x[:, None] == y[None]).all(-1)), 'Ambiguous physical code in selected pair')
    diagonal = torch.logaddexp(replace, copy.masked_fill(~same, -torch.inf))
    stop = action[-1, -1, STOP]
    fast = alignment_log_likelihood(insert, drop, diagonal, stop)
    brute, count = path_sum(insert, drop, replace, copy, same, stop)
    parameters = [net.action.weight, net.draw.output.weight, net.step.weight_ih_l0, net.shape.weight]
    left = torch.autograd.grad(fast, parameters, retain_graph=True)
    right = torch.autograd.grad(brute, parameters)
    torch.testing.assert_close(fast, brute, atol=1e-9, rtol=1e-10)
    errors = []
    for g, ref in zip(left, right, strict=True):
        torch.testing.assert_close(g, ref, atol=2e-9, rtol=1e-8)
        errors.append(float((g-ref).abs().max()))
    require(math.isfinite(float(fast.detach())) and float(fast.detach()) <= 0, 'Invalid target probability')
    return dict(chapter=row['g'], index=row['i'], source=row['source'], target=row['target'],
                paths=count, log_probability=float(fast.detach()),
                log_probability_gap=float((fast-brute).abs().detach()), max_gradient_gap=max(errors))


def run():
    torch.set_num_threads(2)
    weights = load_file(str(PARENT))
    net = FertilityEditor(weights['landmarks'], len(weights['fertility.2.bias'])-1)
    net.load_state_dict(weights)
    net.double().eval()
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'pixels.npy') == prep['pixels_sha256'], 'Input bank changed')
    ink = torch.from_numpy(np.unpackbits(np.load(WORK/'pixels.npy'), axis=1)).bool()
    q = visual_input(ink[load(OUT/'preparation.json')['question_tiles'][0][0]], net.landmarks).double()
    examples = []
    seen = set()
    for row in read_split('shard_0', prep):
        n, m = len(row['x']), len(row['y'])
        if not (2 <= n <= 4 and 2 <= m <= 5):
            continue
        kind = (m>n) - (m<n)
        if kind not in seen and row['x'] != row['y']:
            examples.append(check(net, ink, q, row))
            seen.add(kind)
        if len(seen) == 3:
            break
    require(len(examples) == 3, 'Expected natural contraction, expansion and equal length edits')
    path = ROOT/'references/evidence/visual_alignment_enumeration_20261118.json'
    write_new(path, dict(parent_sha256=sha(PARENT), source='first eligible training pair of each length relation',
                         examples=examples, optimized=False, dtype='float64', device='cpu'))
    print(examples)


if __name__ == '__main__':
    run()
