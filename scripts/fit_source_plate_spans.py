"""Supervise stage-name localization from training ink and score changed evidence."""
import fcntl
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.source_plate import SourcePlateReader, StageSpanReader
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_source_plate_coverage import new_phrasing
from scripts.fit_source_plate_reader import DATA, DISTILL, OUT as PRIOR, ROOT, assess, guard, load_base, table, tensor
from scripts.prepare_source_plate_questions import Rasterizer, STAGES

OUT = ROOT / 'artifacts/source_plate_spans_v1'


def train_spans(reader, features, px, questions, start):
    raster = Rasterizer()
    index = {p.tobytes(): i for i, p in enumerate(px)}
    aliases = [name for stage in STAGES.values() for name in stage]
    names = np.zeros((len(aliases), reader.max_width), np.int32)
    for i, name in enumerate(aliases):
        names[i, :len(name)] = [index[raster.tile(c)] for c in name]
        require((np.lib.stride_tricks.sliding_window_view(questions[:4096], len(name), axis=1)
                 == names[i, :len(name)]).all(-1).any(), 'Alias missing from training windows')
    keys = features[tensor(names[::2, :3])]
    lengths = np.array(list(map(len, aliases)))
    vocabulary = np.unique(questions)
    rng = np.random.default_rng(718)
    opt = torch.optim.AdamW(reader.parameters(), lr=.001)
    history = []
    for step in range(1500):
        # Include intact names, physical near misses and natural training-question substrings.
        which = rng.integers(len(names), size=512)
        batch = names[which].copy()
        changed = np.arange(128, 320)
        pos = (rng.random(len(changed)) * np.minimum(lengths[which[changed]] + 1, reader.max_width)).astype(int)
        batch[changed, pos] = rng.choice(vocabulary, len(changed))
        count = len(batch) - 320
        widths = rng.integers(1, reader.max_width + 1, size=count)
        begins = (rng.random(count) * (25 - widths)).astype(int)
        offsets = np.arange(reader.max_width)
        positions = begins[:, None] + offsets
        batch[320:] = np.where(offsets < widths[:, None],
                               questions[rng.integers(len(questions), size=count)[:, None], positions.clip(max=23)], 0)
        exact = (batch[:, None] == names[None]).all(-1).reshape(len(batch), 4, 2).any(-1)
        scores = reader.embed_spans(features[tensor(batch)]) @ reader.embed_headers(keys).T
        loss = F.binary_cross_entropy_with_logits(16 * (scores - .5), tensor(exact).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            guard(start)
            history.append(float(loss.detach()))
    return history


def third_phrasing(contexts, rows, language, lengths, px):
    lookup = {r.tobytes(): i for i, r in enumerate(px)}
    raster = Rasterizer()
    cards = load(DATA / 'development_cards.json')
    contexts = contexts.copy()
    for i, row in enumerate(rows):
        card = cards[i // 16]
        char = card[row[0]]['char']
        words = [STAGES[card[r]['stage']][language[i]] for r in row[:lengths[i]]]
        prompt = char + '? ' + ' > '.join(words) + '.' if language[i] else '「' + char + '」─' + '與'.join(words) + '。'
        require(len(prompt) <= 24, 'Prompt length')
        contexts[i, :24] = 0
        for j, c in enumerate(prompt, 24 - len(prompt)):
            contexts[i, j] = lookup.setdefault(raster.tile(c), len(lookup))
    return contexts, np.frombuffer(b''.join(lookup), np.uint8).reshape(-1, 128)


def score(ink, n, stop, route, target, lang, length):
    same = (ink[:, :2] == target).all(2)
    hit = stop & (length == n) & same[:, 0] & ((length == 1) | same[:, 1])
    block = hit.reshape(-1, 4)
    return dict(exact=int(hit.sum()), routed=int(route.sum()), length_correct=int((stop & (length == n)).sum()),
                both_reversed=int((block[:, 2] & block[:, 3]).sum()),
                groups={f'{name}_{k}': int(hit[(lang == l) & (length == k)].sum())
                        for l, name in enumerate(('zh', 'en')) for k in (1, 2)})


def main():
    start = time.monotonic()
    guard(start)
    require(not load(ROOT / 'artifacts/source_plate_coverage_v1/result.json')['qualified'], 'Unexpected predecessor')
    OUT.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(718)
    torch.set_float32_matmul_precision('high')
    writer = load_base()
    plate = SourcePlateReader().cuda().eval()
    plate.load_state_dict(load_file(str(PRIOR / 'source_plate.safetensors')))
    plate.requires_grad_(False)
    px = np.load(DATA / 'pixels.npy')
    report = load(DATA / 'report.json')
    require(sha(DATA / 'pixels.npy') == report['pixels_sha256'] and all(
        sha(DATA / (n + '.npz')) == r['examples_sha256'] for n, r in report['splits'].items()), 'Changed corpus')
    features = visual_input(table(DATA / 'pixels.npy'), writer.landmarks)
    model = StageSpanReader().cuda()
    with np.load(DATA / 'train.npz') as z:
        questions = z['context'][:, :24]
    begin = time.monotonic()
    history = train_spans(model, features, px, questions, start)
    elapsed = time.monotonic() - begin
    weights = OUT / 'stage_spans.safetensors'
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, str(weights))
    model = StageSpanReader().cuda().eval()
    model.load_state_dict(load_file(str(weights)))
    with np.load(DATA / 'development.npz') as z:
        ctx, y, lang, length, rows = (z[k] for k in ('context', 'target', 'language', 'length', 'row'))
    third, more = third_phrasing(ctx, rows, lang, length, px)
    rotated = third.copy()
    rotated[:, 28:44:5] = np.roll(third[:, 28:44:5], 1, axis=1)
    rotated_y = np.take_along_axis(rotated[:, 28:44:5], rows.clip(min=0), axis=1)
    paraphrase, alt = new_phrasing(ctx, rows, lang, length, px)
    arms = {}
    for name, ids, bank, target in [('intact', ctx, px, y), ('paraphrase', paraphrase, alt, y),
                                    ('third', third, more, y), ('relocated', rotated, more, rotated_y),
                                    ('blank_question', ctx, px, y), ('blank_plate', ctx, px, y)]:
        ink, n, stop, route = assess(writer, plate, tensor(np.unpackbits(bank, axis=1)).bool(), ids, name, start, model)
        file = OUT / (name + '.npz')
        np.savez_compressed(file, packed=ink, length=n, stopped=stop, copy=route)
        arms[name] = score(ink, n, stop, route, px[target], lang, length) | {'sha256': sha(file)}
        print(name, arms[name], flush=True)
    fresh = DISTILL / 'fresh'
    with np.load(fresh / 'contexts.npz') as z:
        mx, my = z['context'], z['target']
    ink, n, stop, route = assess(writer, plate, table(fresh / 'pixels.npy'), mx, 'intact', start, model)
    with np.load(fresh / 'distilled_targets/generated.npz') as z:
        preserved = (ink == z['packed']).all((1, 2)) & (n == z['length']) & (stop == z['stopped'])
    exact = stop & (n == 4) & (ink[:, :4] == np.load(fresh / 'pixels.npy')[my]).all((1, 2))
    np.savez_compressed(OUT / 'fresh_cloze.npz', packed=ink, length=n, stopped=stop, copy=route)
    threshold = len(ctx) / 4
    passed = all(min(arms[m]['groups'].values()) > .9 * threshold and arms[m]['both_reversed'] > .8 * threshold
                 for m in ('intact', 'paraphrase', 'third', 'relocated'))
    passed &= all(arms[m]['exact'] == 0 for m in ('blank_question', 'blank_plate')) and preserved.mean() >= .99
    result = dict(questions=len(ctx), arms=arms, qualified=bool(passed), fresh_cases=len(mx),
                  fresh_preserved=int(preserved.sum()), fresh_exact=int(exact.sum()), fresh_routes=int(route.sum()),
                  parameters=sum(p.numel() for p in model.parameters()), fit_seconds=elapsed, loss_samples=history,
                  total_seconds=time.monotonic() - start, peak_bytes=torch.cuda.max_memory_allocated(),
                  weights_sha256=sha(weights), source_reader_sha256=sha(PRIOR / 'source_plate.safetensors'))
    write_new(OUT / 'result.json', result)
    print('qualified', passed, flush=True)


if __name__ == '__main__':
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
