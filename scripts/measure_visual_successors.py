"""Fit a glyph-only successor discriminator and score held-out contextual assignments."""
from collections import Counter, defaultdict
import hashlib
import itertools
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from ilm.visual_lm.visual_successor import VisualSuccessor
from scripts.compare_visual_passage_registers import matching_sentences, physical_records
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_visual_word_order import FIT, REG, ROOT, RUN, credits
from scripts.prepare_visual_sequence_corpus import CMRC, Rasterizer

OUT = ROOT / 'artifacts/visual_successor_v1'
PROTOCOL = ROOT / 'references/visual_successor_protocol.md'
SOURCE = CMRC / 'cmrc2018_dev.json'
SOURCE_HASH = 'e9ff74231f05c230c6fa88b84441ee334d97234cbb610991cd94b82db00c7f1f'
HISTORY, CHUNK, SEED = 32, 64, 20261118


def guard(start):
    mem = next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    require(int(mem.split()[1]) / 1024**2 >= 24, 'Host memory reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 < 4, 'Process memory limit')
    require(time.monotonic() - start < 1200, 'CPU experiment deadline')


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def proposals(title, paragraphs, renderer, handles):
    for txt in sorted(paragraphs, key=digest):
        cells = []
        for c in txt:
            try:
                cells.append(chr(handles.get(renderer.tile(c), 0)))
            except ValueError:
                cells.append('\0')
        stream = ''.join(cells)
        groups = defaultdict(list)
        for i in range(HISTORY, len(txt)):
            if not all('\u4e00' <= c <= '\u9fff' for c in txt[i-1:i+1]):
                continue
            window = stream[i-HISTORY:i+1]
            if '\0' not in window and window[-1] not in window[:-1]:
                groups[window[-2]].append(i)
        positions = sorted((p for group in groups.values() for p in itertools.combinations(group, 2)),
                           key=lambda p: digest(f'{title}/{p[0]}/{p[1]}'))
        for i, j in positions:
            if j-i <= HISTORY:
                continue
            a, b = stream[i-HISTORY:i+1], stream[j-HISTORY:j+1]
            if a[-1] != b[-1] and a[-1] not in b[:-1] and b[-1] not in a[:-1]:
                yield a, b, dict(article_sha256=digest(title), paragraph_sha256=digest(txt), offsets=[i, j])


def cohort(handles, corpus):
    require(sha(SOURCE) == SOURCE_HASH, 'Evaluation source differs')
    articles = defaultdict(list)
    for a in load(SOURCE)['data']:
        articles[a['title']].extend(p['context'] for p in a['paragraphs'])
    titles = sorted(articles, key=digest)
    renderer = Rasterizer()
    streams = {t: proposals(t, articles[t], renderer, handles) for t in titles}
    pending = {t: next(streams[t], None) for t in titles}
    counts = Counter(titles=len(titles), without_pair=sum(p is None for p in pending.values()))
    accepted, used = {}, set()
    while pending:
        candidates = {t: p for t, p in pending.items() if p}
        patterns = {s[:-1] for a, b, _ in candidates.values() for s in (a, b)}
        hits = matching_sentences(patterns, corpus)
        pending = {}
        for t, (a, b, info) in candidates.items():
            histories = {a[:-1], b[:-1]}
            if len(histories) < 2 or histories & (hits | used):
                counts['overlap_rejections'] += 1
                pending[t] = next(streams[t], None)
            else:
                accepted[t] = (a, b, info)
                used.update(histories)
    rows = [accepted[t] for t in titles if t in accepted]
    counts['eligible'] = len(rows)
    return rows[:512], counts


def make_bank(handles):
    ordered = sorted(handles, key=handles.get)
    require([handles[t] for t in ordered] == list(range(1, len(ordered)+1)), 'Physical index alignment')
    packed = np.frombuffer(b''.join([bytes(128), *ordered]), dtype=np.uint8).reshape(-1, 128)
    return torch.from_numpy(np.unpackbits(packed, axis=1).reshape(-1, 32, 32))


def index_pairs(pairs, reverse):
    h, y = [], []
    for a, b, _ in pairs:
        for s in (a, b):
            hist = list(map(ord, s[:-1]))
            h.append(hist[-2::-1]+hist[-1:] if reverse else hist)
            y.append([ord(a[-1]), ord(b[-1])])
    return torch.tensor(h), torch.tensor(y)


@torch.inference_mode()
def evaluate(role, model, bank, pairs, start):
    scores = {}
    if role == 'native':
        features = torch.cat([model.encode_ink(b[:, None]) for b in bank.split(96)])
    else:
        features = model.shape(model.pool(bank))
    for reverse in (False, True):
        histories, targets = index_pairs(pairs, reverse)
        batches = []
        for h, y in zip(histories.split(32), targets.split(32)):
            guard(start)
            if role == 'native':
                state = model.history_codes(features[h])[:, -1]
                values = model.writer.log_prob(state.repeat_interleave(2, 0), features[y.flatten()]).reshape(-1, 2)
            else:
                q = model.queries(features[h])[:, -1:]
                values = 10 * (q @ model.keys(features[y]).transpose(1, 2)).squeeze(1)
            batches.append(values)
        all_scores = torch.cat(batches).reshape(-1, 2, 2).double().numpy()
        require(np.isfinite(all_scores).all(), 'Nonfinite assignment scores')
        scores['reversed' if reverse else 'normal'] = all_scores
    return scores


def train_batch(model, pooled, chunks, rotated, rng):
    """An output at position t sees only observed cells through position t-1."""
    ids = np.zeros((len(chunks), HISTORY + CHUNK), dtype=np.int64)
    valid = np.zeros((len(chunks), CHUNK), dtype=bool)
    label_indices = np.zeros((len(chunks), CHUNK), dtype=np.int64)
    for i, (history, targets) in enumerate(chunks):
        n = len(targets)
        ids[i, HISTORY-len(history):HISTORY] = history
        ids[i, HISTORY:HISTORY+n] = targets
        valid[i, :n] = True
        offset = rng.integers(1, n) if rotated and n > 1 else 0
        label_indices[i, :n] = (np.arange(n) + offset) % n
    ids, valid = torch.from_numpy(ids), torch.from_numpy(valid)
    targets = ids[:, HISTORY:]
    assigned = targets.gather(1, torch.from_numpy(label_indices))
    encoded = model.shape(pooled[ids])
    # Zero is reserved for missing input positions; real blank bitmaps have a positive handle.
    encoded = encoded.masked_fill((ids == 0).unsqueeze(-1), 0)
    q = model.queries(encoded[:, :-1])
    k = model.keys(encoded[:, HISTORY:])
    scores = 10 * (q @ k.transpose(1, 2))
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    # Remove padded query rows before logsumexp to avoid undefined gradients.
    rows = scores[valid]
    positives = (targets[:, None, :] == assigned[:, :, None])[valid]
    loss = (rows.logsumexp(-1) - rows.masked_fill(~positives, -torch.inf).logsumexp(-1)).mean()
    return loss, int(valid.sum()), int(((assigned != targets) & valid).sum())


def fit(model, bank, modern, rotated, start):
    chunks = []
    for passage in modern:
        ids = np.fromiter(map(ord, passage), dtype=np.int64)
        for i in range(1, len(ids), CHUNK):
            chunks.append((ids[max(0, i-HISTORY):i], ids[i:i+CHUNK]))
    order = np.random.default_rng(SEED).permutation(len(chunks))
    rng = np.random.default_rng(SEED+2)
    pooled = model.pool(bank)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    observations = changed = total_loss = updates = 0
    began = time.monotonic()
    for i in range(0, len(order), 16):
        guard(start)
        batch = [chunks[idx] for idx in order[i:i+16]]
        loss, n, swaps = train_batch(model, pooled, batch, rotated, rng)
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        observations += n
        changed += swaps
        total_loss += float(loss.detach()) * n
        updates += 1
    require(observations == 1047284, 'Incomplete training exposure')
    return dict(targets=observations, changed_labels=changed, updates=updates,
                mean_loss=total_loss / observations, seconds=time.monotonic()-began)


def trigram(modern, pairs):
    inputs = [index_pairs(pairs, reverse) for reverse in (False, True)]
    wanted = {tuple(h[-2:].tolist()) for histories, _ in inputs for h in histories}
    second, last, freq = defaultdict(Counter), defaultdict(Counter), Counter()
    for p in modern:
        ids = list(map(ord, p))
        freq.update(ids[1:])
        for i in range(1, len(ids)):
            last[ids[i-1]][ids[i]] += 1
            if i >= 2 and (ctx := (ids[i-2], ids[i-1])) in wanted:
                second[ctx][ids[i]] += 1

    def smooth(counts, backoff, y):
        if not counts:
            return backoff
        return (counts[y]+len(counts)*backoff)/(sum(counts.values())+len(counts))

    scores = {}
    denominator = sum(freq.values()) + len(freq) + 1
    for arm, (histories, targets) in zip(('normal', 'reversed'), inputs):
        values = []
        for history, yy in zip(histories.tolist(), targets.tolist()):
            ctx = tuple(history[-2:])
            row = []
            for y in yy:
                prob = smooth(last[ctx[-1]], (freq[y]+1)/denominator, y)
                row.append(np.log(smooth(second[ctx], prob, y)))
            values.append(row)
        scores[arm] = np.array(values).reshape(-1, 2, 2)
    return scores


def assignment(scores):
    a = scores[:, 0, 0] - scores[:, 0, 1]
    b = scores[:, 1, 1] - scores[:, 1, 0]
    return credits(a + b), dict(pairs=len(a), correct=int((a+b > 1e-6).sum()),
        ties=int((abs(a+b) <= 1e-6).sum()), accuracy=float(credits(a+b).mean()),
        individual_accuracy=float((credits(a).mean()+credits(b).mean())/2),
        both_strict=int(((a > 1e-6) & (b > 1e-6)).sum()))


def main():
    start = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only')
    guard(start)
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    handles, _, modern, corpus = physical_records(start)
    pairs, selection = cohort(handles, corpus)
    require(len(pairs) > 0, 'No eligible test articles')
    del corpus
    bank = make_bank(handles)
    h, y = index_pairs(pairs, False)
    roster = []
    for i, (_, _, info) in enumerate(pairs):
        histories, targets = h[2*i:2*i+2], y[2*i]
        require(histories[0, -1] == histories[1, -1] and targets[0] != targets[1]
                and not torch.isin(targets, histories).any(), 'Assignment or copy exclusion')
        roster.append(dict(**info, images_sha256=hashlib.sha256(bank[torch.cat((histories.flatten(), targets))].numpy().tobytes()).hexdigest()))
    write_new(OUT/'articles.json', roster)
    print(f'Frozen {len(roster)} article assignments', flush=True)
    scores = {'trigram': trigram(modern, pairs)}
    completed = load(FIT)
    require(sha(REG) == completed['registration_sha256'] and sha(RUN/'audit.json') == completed['audit_sha256'], 'Fit admission')
    native = ContinuousImageLanguage(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()), components=8).eval()
    arm, = [a for a in completed['arms'] if a['components'] == 8]
    snapshots = {s['update']: s for s in arm['snapshots']}
    for step in (0, 13785):
        snap = snapshots[step]
        path = ROOT/snap['file']
        require(sha(path) == snap['sha256'], 'Native weights differ')
        native.load_state_dict(torch.load(path, map_location='cpu', weights_only=True, mmap=True)['model'])
        scores[f'native_{step}'] = evaluate('native', native, bank, pairs, start)
    del native
    print('Native densities complete', flush=True)
    torch.manual_seed(SEED)
    model = VisualSuccessor()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    scores['initial'] = evaluate('reader', model.eval(), bank, pairs, start)
    fits = {}
    for name in ('aligned', 'rotated'):
        model.load_state_dict(initial)
        fits[name] = fit(model.train(), bank, modern, name == 'rotated', start)
        torch.save(model.state_dict(), OUT/f'{name}.pt')
        scores[name] = evaluate('reader', model.eval(), bank, pairs, start)
        print(f'{name} completed {fits[name]}', flush=True)
    arrays = {f'{name}_{condition}': values for name, arms in scores.items() for condition, values in arms.items()}
    np.savez_compressed(OUT/'scores.npz', **arrays)
    stats, credit = {}, {}
    for k, v in arrays.items():
        credit[k], stats[k] = assignment(v)
    draws = np.random.default_rng(SEED+1).integers(0, len(pairs), (10000, len(pairs)))
    ci = {}
    for a, others in [('aligned', ['chance','initial','rotated','trigram','reversed']),
                      ('native_13785', ['chance','native_0','trigram','reversed'])]:
        for b in others:
            ref = .5 if b == 'chance' else credit[a+'_reversed' if b == 'reversed' else b+'_normal']
            dif = credit[a+'_normal']-ref
            ci[f'{a}_minus_{b}'] = [float(dif.mean()), *np.quantile(dif[draws].mean(1), [.025,.975]).tolist()]
    guard(start)
    files = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    write_new(OUT/'result.json', dict(source_sha256=sha(SOURCE), fit_sha256=sha(FIT), protocol_sha256=sha(PROTOCOL),
        code_sha256={p:sha(ROOT/p) for p in ('ilm/visual_lm/visual_successor.py','scripts/measure_visual_successors.py')},
        files=files, selection=dict(selection), test_articles=len(roster), stats=stats, differences_and_intervals=ci,
        train_passages=len(modern), fits=fits, trainable_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.monotonic()-start, rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cpu_threads=2, torch_version=str(torch.__version__), autonomous_answers=0))
    print(ci, flush=True)


if __name__ == '__main__':
    main()
