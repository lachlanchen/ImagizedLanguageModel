"""Transfer a frozen picture reader to the original physical writer on training contexts."""
import fcntl
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, STOP, PicturedIdiom, nibbles, visual_input
from scripts import fit_pictured_idioms as first
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new

ROOT, DATA = first.ROOT, first.DATA
PRIOR = ROOT / 'artifacts/pictured_idiom_alignment_v1'
OUT = ROOT / 'artifacts/visual_writer_distillation_v1'
PROTOCOL = ROOT / 'references/visual_writer_distillation_protocol.md'
EPOCHS, BATCH = 24, 192
WRITER = ('ink_group.', 'group_position.', 'feedback.', 'causal.', 'next_pattern.')


def writer_logits(model, context, target_bits):
    codes = nibbles(target_bits).flatten(1)
    labels = F.pad(codes, (0, 1), value=STOP)
    previous = F.pad(codes, (1, 0), value=17)
    steps = torch.arange(previous.shape[1], device=context.device)
    prior = F.pad(target_bits, (0, 0, 1, 0))[:, steps // GROUPS]
    seq = model.inputs(previous, (steps % GROUPS).expand_as(previous), prior,
                       context[:, None].expand(-1, len(steps), -1))
    states, _ = model.causal(seq, context[None].contiguous())
    return model.next_pattern(states), labels


@torch.no_grad()
def prepare_teacher(model, aux, features, inputs, targets):
    options, inverse = torch.unique(targets, dim=0, return_inverse=True)
    keys = F.normalize(aux(features[options].flatten(1)))
    rng = torch.Generator(device=inputs.device).manual_seed(714)
    schedules, states = [], []
    exact, changed, top_mass = 0, 0, 0.0
    for start in range(0, len(inputs), 256):
        subset = slice(start, start + 256)
        context = model.read(features[inputs[subset]])
        probability = (F.normalize(context) @ keys.T / 0.02).softmax(-1)
        weight, best = probability.max(1)
        draw = torch.multinomial(probability, EPOCHS, replacement=True, generator=rng)
        gold = inverse[subset, None].expand_as(draw)
        keep = torch.rand(draw.shape, device=draw.device, generator=rng) < 0.25
        choice = torch.where(keep, gold, draw)
        schedules.append(choice.int())
        states.append(context)
        changed += int((choice != gold).sum())
        exact += int((best == gold[:, 0]).sum())
        top_mass += weight.sum().item()
    return torch.cat(states), options, torch.cat(schedules), {
        'examples': len(inputs), 'options': len(options), 'top1_training_correct': exact,
        'mean_top1_probability': top_mass / len(inputs), 'changed_targets': changed,
        'total_targets': len(inputs) * EPOCHS,
    }


def fit_arm(name, initial, features, inputs, targets, contexts, options, schedule, bits, start):
    model = PicturedIdiom(initial['landmarks']).cuda().eval()
    model.load_state_dict(initial)
    for key, value in model.named_parameters():
        value.requires_grad_(key.startswith(WRITER))
    trainable = [value for value in model.parameters() if value.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=0.0006, weight_decay=0.01)
    rng = np.random.default_rng(613)
    mask = torch.arange(GROUPS * 4 + 1, device=inputs.device) % GROUPS != 0
    test_rows = torch.arange(17, device=inputs.device)
    with torch.no_grad():
        full, original = model.teacher_logits(features[inputs[test_rows]], bits[targets[test_rows]])
        cached, copy = writer_logits(model, contexts[test_rows], bits[targets[test_rows]])
        require(torch.equal(original, copy) and torch.allclose(full, cached, atol=2e-4, rtol=2e-4),
                'Cached context disagrees with original teacher path')
    path = OUT / name
    path.mkdir()
    torch.cuda.synchronize()
    begin = time.monotonic()
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(inputs))
        total = 0.0
        for pos in range(0, len(order), BATCH):
            if pos % 3072 == 0:
                first.guard(start)
            rows = order[pos:pos + BATCH]
            answer = targets[rows] if name == 'original_targets' else options[schedule[rows, epoch]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores, labels = writer_logits(model, contexts[rows], bits[answer])
                stop = scores[:, :, STOP:].masked_fill(mask[None, :, None], -torch.inf)
                scores = torch.cat([scores[:, :, :STOP], stop], dim=-1)
                loss = F.cross_entropy(scores.flatten(0, 1).float(), labels.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            opt.step()
            total += loss.detach().item() * len(rows)
        history.append(total / len(inputs))
        print(name, epoch + 1, round(history[-1], 5), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - begin
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    require(all(torch.equal(v, initial[k]) for k, v in state.items() if not k.startswith(WRITER)),
            'Frozen tensors changed')
    torch.save(state, path / 'weights.pt')
    write_new(path / 'fit.json', dict(epochs=EPOCHS, losses=history, fit_seconds=elapsed,
                                      trained_parameters=sum(t.numel() for t in trainable),
                                      frozen_equal=True, weights_sha256=sha(path / 'weights.pt')))


def main():
    start = time.monotonic()
    first.guard(start)
    require(torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(), 'Expected 4090')
    torch.set_num_threads(2)
    torch.manual_seed(613)
    # The frozen reader made its archived inventory decisions in FP32 on the high setting.
    torch.set_float32_matmul_precision('high')
    prior = load(ROOT / 'references/evidence/pictured_idiom_alignment_20261118/result.json')
    prep = load(ROOT / 'references/evidence/pictured_idioms_20261118/preparation.json')
    for name in ('weights.pt', 'training_only_aux.pt', 'diagnostic_choices.npz', 'generated.npz'):
        path = PRIOR / name
        require(sha(path) == prior['source_hashes'][str(path.relative_to(ROOT))], 'Prior result changed')
    require(sha(DATA / 'pixels.npy') == prep['pixels_sha256'], 'Pixel table changed')
    for split, entry in prep['splits'].items():
        require(sha(DATA / (split + '.npz')) == entry['sha256'], 'Corpus changed')
    OUT.mkdir()
    initial = torch.load(PRIOR / 'weights.pt', weights_only=True, map_location='cpu')
    reader = PicturedIdiom(initial['landmarks']).cuda().eval()
    reader.load_state_dict(initial)
    aux = nn.Sequential(nn.Linear(448, 384), nn.GELU(), nn.Linear(384, 192)).cuda().eval()
    aux.load_state_dict(torch.load(PRIOR / 'training_only_aux.pt', weights_only=True))
    pixels = np.load(DATA / 'pixels.npy')
    ink = torch.as_tensor(np.unpackbits(pixels, axis=1), device='cuda', dtype=torch.bool)
    features = visual_input(ink, reader.landmarks)
    bits = ink[:, reader.landmarks].float()
    with np.load(DATA / 'train.npz') as f:
        train = {key: f[key] for key in f.files}
    x = torch.as_tensor(train['context'].astype(np.int32), device='cuda')
    y = torch.as_tensor(train['target'].astype(np.int32), device='cuda')
    begin = time.monotonic()
    contexts, options, schedule, teacher = prepare_teacher(reader, aux, features, x, y)
    torch.cuda.synchronize()
    teacher['seconds'] = time.monotonic() - begin
    write_new(OUT / 'teacher.json', teacher)
    for name in ('original_targets', 'distilled_targets'):
        fit_arm(name, initial, features, x, y, contexts, options, schedule, bits, start)
    # Both final fits are fixed before opening validation targets.
    with np.load(DATA / 'validation.npz') as f:
        dev = {key: f[key] for key in f.files}
    for name in ('original_targets', 'distilled_targets'):
        first.OUT = OUT / name
        net = PicturedIdiom(initial['landmarks']).cuda()
        net.load_state_dict(torch.load(first.OUT / 'weights.pt', weights_only=True))
        result = first.compare(net, features, ink, train, dev, pixels, np.unique(train['target']), start)
        write_new(first.OUT / 'result.json', result)
        print(name, 'complete exact', result['complete_exact'], flush=True)
    first.guard(start)
    write_new(OUT / 'execution.json', dict(seconds=time.monotonic()-start, device=torch.cuda.get_device_name(),
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              protocol_sha256=sha(PROTOCOL), program_sha256=sha(Path(__file__))))


if __name__ == '__main__':
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
