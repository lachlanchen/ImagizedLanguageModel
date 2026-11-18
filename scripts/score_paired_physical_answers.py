"""Compare the likelihood of two given physical answers in both matched contexts."""
import fcntl
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, STOP, PicturedIdiom, bit_image, nibbles, visual_input
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_local_teacher_writer import DATA, GUIDED, OUT, load_split
from scripts.fit_pictured_idioms import guard
from scripts.score_visual_writer_transfer import matching_pairs

DEST = OUT / 'paired_trajectories'
TOLERANCE = .0001


def physical_score(net, context, bits, feedback):
    """Teacher-force one specified path, reading the rendered image at each boundary."""
    symbols = nibbles(bits).flatten(1)
    label = F.pad(symbols, (0, 1), value=STOP)
    previous = F.pad(symbols, (1, 0), value=17)
    step = torch.arange(label.shape[1], device=context.device)
    last = F.pad(feedback, (0, 0, 1, 0))[:, step // GROUPS]
    seq = net.inputs(previous, (step % GROUPS).expand_as(label), last, context[:, None].expand(-1, len(step), -1))
    hidden, _ = net.causal(seq, context[None].contiguous())
    score = net.next_pattern(hidden).float()
    score[:, step % GROUPS != 0, STOP] = -torch.inf
    return score.log_softmax(-1).gather(-1, label[..., None]).squeeze(-1).sum(1)


def stepwise_score(net, context, bits):
    """Independent, single-step recurrence with physical feedback computed inside the loop."""
    codes = nibbles(bits)
    previous = torch.full((len(bits),), 17, dtype=torch.long, device=bits.device)
    feedback = torch.zeros_like(bits[:, 0])
    hidden = context[None].contiguous()
    total = context.new_zeros(len(bits))
    for t in range(codes.size(1) * GROUPS + 1):
        glyph, offset = divmod(t, GROUPS)
        inputs = net.inputs(previous, torch.full_like(previous, offset), feedback, context)
        out, hidden = net.causal(inputs[:, None], hidden)
        logits = net.next_pattern(out[:, 0])
        if offset:
            logits[:, STOP] = -torch.inf
        selected = torch.full_like(previous, STOP) if glyph == codes.size(1) else codes[:, glyph, offset]
        total += logits.log_softmax(-1).gather(1, selected[:, None])[:, 0]
        previous = selected
        if offset == GROUPS - 1:
            pic = net.motor(bit_image(codes[:, glyph])) > 0
            feedback = pic[:, net.landmarks].float()
    return total


def count(scores):
    # Columns are L(x,a), L(x,b), L(y,a), L(y,b).
    left = scores[:, 0] - scores[:, 1]
    right = scores[:, 3] - scores[:, 2]
    margin = left + right
    return dict(pairs=len(scores), individual_correct=int((left > TOLERANCE).sum() + (right > TOLERANCE).sum()),
                both_correct=int(((left > TOLERANCE) & (right > TOLERANCE)).sum()),
                neither_correct=int(((left < -TOLERANCE) & (right < -TOLERANCE)).sum()),
                assignment_correct=int((margin > TOLERANCE).sum()), assignment_ties=int((abs(margin) <= TOLERANCE).sum()),
                margin_quantiles=np.quantile(margin, [.1, .5, .9]).tolist())


@torch.inference_mode()
def score(net, split, data, px, name, start, *, destination=DEST):
    x, y, group = (data[k] for k in ('context', 'target', 'group'))
    i, j = matching_pairs(x, y, group)
    require((x[i, 23:25] == x[j, 23:25]).all() and (y[i] != y[j]).any(1).all() and (group[i] != group[j]).all(),
            'Pairs must share the adjacent cells and differ in answer and source')
    ink = torch.as_tensor(np.unpackbits(px, axis=1), device='cuda').bool()
    features = visual_input(ink, net.landmarks)
    bits = ink[:, net.landmarks].float()
    rendered = torch.cat([net.motor(bits[k:k+128]) > 0 for k in range(0, len(bits), 128)])
    feedback = rendered[:, net.landmarks].float()
    physical = (ink == rendered).all(1).cpu().numpy()
    scores = []
    max_error = 0.
    for offset in range(0, len(i), 64):
        left, right = i[offset:offset+64], j[offset:offset+64]
        prompts = np.concatenate([left, right])
        context = net.read(features[torch.as_tensor(x[prompts].astype(np.int32), device='cuda')])
        cx, cy = context.chunk(2)
        combined = torch.cat((cx, cx, cy, cy))
        ids = torch.as_tensor(y[np.concatenate((left, right, left, right))].astype(np.int32), device='cuda')
        value = physical_score(net, combined, bits[ids], feedback[ids])
        if offset == 0:
            other = stepwise_score(net, combined[:16], bits[ids[:16]])
            max_error = float((other - value[:16]).abs().max())
            require(max_error < .001, 'Sequential path probability differs')
        scores.append(torch.stack(value.chunk(4), dim=1).cpu().numpy())
        if offset % 1024 == 0:
            guard(start)
    scores = np.vstack(scores).astype(np.float64)
    renderable = physical[y[i]].all(1) & physical[y[j]].all(1)
    seen = set()
    distinct = []
    for k, (left, right) in enumerate(zip(group[i], group[j])):
        if left not in seen and right not in seen:
            distinct.append(k)
            seen.update((left, right))
    path = destination / (split + '_' + name + '.npz')
    np.savez_compressed(path, i=i, j=j, scores=scores, renderable=renderable, distinct_source_pairs=distinct)
    return dict(all=count(scores), renderable=count(scores[renderable]), unique_sources=count(scores[distinct]),
                replay_max_nats=max_error, sha256=sha(path))


def main():
    start = time.monotonic()
    guard(start)
    require((OUT/'evaluation/result.json').is_file(), 'Autonomous results required')
    DEST.mkdir()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    report = {}
    for name, parent in [('guided', GUIDED/'distilled_targets'), ('aligned', OUT/'aligned')]:
        weights = parent/'weights.pt'
        require(sha(weights) == load(parent/'fit.json')['weights_sha256'], 'Checkpoint changed')
        state = torch.load(weights, weights_only=True, map_location='cpu')
        net = PicturedIdiom(state['landmarks']).cuda().eval()
        net.load_state_dict(state)
        report[name] = {}
        for split, folder, file in [('validation', DATA, 'validation.npz'), ('previous', GUIDED/'fresh', 'contexts.npz'),
                                    ('new', OUT/'confirmation', 'contexts.npz')]:
            result = score(net, split, load_split(folder/file), np.load(folder/'pixels.npy'), name, start)
            report[name][split] = result
            print(name, split, result['all'], flush=True)
    report['resources'] = dict(seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_new(DEST/'result.json', report)


if __name__ == '__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
