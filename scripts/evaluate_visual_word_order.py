"""Mandarin minimal-pair scoring from physical glyphs at five saved checkpoints."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tarfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_visual_sequence_corpus import FONT, Rasterizer
from scripts.train_continuous_image_language import DATA, ROOT, RUN

ARCHIVE = ROOT / 'data/evaluation_only/zhoblimp_e8402762/ZhoBLiMP.tar.gz'
ARCHIVE_SHA = 'a0c97ff2dd55d5b67b1bafdb8c4204379c646951c69e43addb4ce781176f7428'
OUTPUT = ROOT / 'artifacts/visual_word_order_v1'
PROTOCOL = ROOT / 'references/visual_word_order_protocol.md'
FIT = ROOT / 'references/evidence/continuous_image_training_completed_20261118.json'
REG = ROOT / 'references/evidence/continuous_autoregressive_full_registration_20261118.json'
STEPS = (0, 3447, 6893, 10339, 13785)
EPSILON = 1e-6


def guard(start, entering=False):
    available = next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    require(int(available.split()[1]) / 1024**2 >= (48 if entering else 24), 'Host memory reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 < 4, 'CPU RSS limit')
    require(time.monotonic() - start <= 1200, 'CPU diagnostic deadline')


class PhysicalBigram:
    """Integer handles are local names for equality of all 1,024 pixels."""
    def __init__(self):
        self.images = {}
        self.transitions = defaultdict(Counter)
        self.unigram = Counter()
        self.records = []
        self.targets = 0

    def add_shard(self, arrays):
        ink = arrays['ink']
        require(ink.dtype == np.uint8 and ink.shape[1:] == (128,), 'Packed binary training tiles required')
        unique, inverse = np.unique(ink.view('V128').reshape(-1), return_inverse=True)
        handles = [self.images.setdefault(tile.tobytes(), len(self.images) + 1) for tile in unique]
        stream = np.asarray(handles)[inverse].tolist()
        offsets = arrays['record_glyph_offsets']
        for first, last, prompt in zip(offsets[:-1], offsets[1:], arrays['prompt_glyphs']):
            record = stream[first:last]
            self.records.append(''.join(map(chr, record)))
            suffix = record[max(1, int(prompt)) - 1:]
            for left, right in zip(suffix, suffix[1:] + [0]):
                self.transitions[left][right] += 1
                self.unigram[right] += 1
            self.targets += len(suffix) - 1

    def finish(self):
        self.searchable = '\0'.join(self.records)
        self.record_count = len(self.records)
        self.records.clear()
        self.counts = {key: sum(c.values()) for key, c in self.transitions.items()}
        self.denominator = sum(self.unigram.values()) + len(self.images) + 2

    def ids(self, tiles):
        return [self.images.get(tile, len(self.images) + 1) for tile in tiles]

    def contains(self, tiles):
        return ''.join(map(chr, self.ids(tiles))) in self.searchable

    def score(self, tiles):
        ids = self.ids(tiles)
        value = 0.
        for a, b in zip(ids, ids[1:] + [0]):
            counts = self.transitions.get(a)
            p = (self.unigram[b] + 1) / self.denominator
            if counts:
                p = (counts[b] + len(counts) * p) / (self.counts[a] + len(counts))
            value += math.log(p)
        return value


def training_baseline(start):
    report = load(DATA / 'report.json')
    require(sha(FONT) == report['font_sha256'], 'Training font differs')
    bound = load(REG)['input_sha256']
    require(sha(DATA / 'report.json') == bound[str((DATA / 'report.json').relative_to(ROOT))], 'Training corpus differs')
    baseline = PhysicalBigram()
    for shard in report['shards']:
        if shard['split'] == 'train':
            guard(start)
            name, = [p for p in shard['files'] if p.endswith('_pixels.npz')]
            path = DATA / name
            require(sha(path) == shard['files'][name] == bound[str(path.relative_to(ROOT))], 'Training pixels differ')
            with np.load(path, allow_pickle=False) as arrays:
                baseline.add_shard(arrays)
    baseline.finish()
    require((baseline.record_count, baseline.targets, baseline.unigram[0]) == (13785, 2571477, 13785), 'Training target accounting')
    return baseline


def selection(baseline):
    require(sha(ARCHIVE) == ARCHIVE_SHA, 'ZhoBLiMP archive differs')
    renderer = Rasterizer()
    pairs, seen, counts = [], set(), Counter()
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        members = sorted((m for m in archive if m.isfile() and m.name.endswith('.jsonl')), key=lambda m: m.name)
        for member in members:
            rows = [json.loads(line) for line in archive.extractfile(member)]
            rows.sort(key=lambda r: hashlib.sha256(f"{r['UID']}/{r['pairID']}".encode()).digest())
            kept = 0
            for r in rows:
                good, bad = r['sentence_good'], r['sentence_bad']
                if len(good) < 2 or good[:1] != bad[:1] or good == bad or sorted(good) != sorted(bad):
                    continue
                counts['eligible_text_rows'] += 1
                if kept == 24:
                    counts['beyond_quota'] += 1
                    continue
                one, two = [tuple(renderer.tile(c) for c in s) for s in (good, bad)]
                require(one[0] == two[0] and Counter(one) == Counter(two), 'Different observed glyphs')
                identity = tuple(sorted((one, two)))
                if one == two or identity in seen:
                    counts['identical_or_duplicate'] += 1
                    continue
                if baseline.contains(one) or baseline.contains(two):
                    counts['training_overlap'] += 1
                    continue
                pairs.append((r['UID'], r['pairID'], one, two))
                seen.add(identity)
                kept += 1
    require(len(pairs) > 0, 'No eligible physical pairs')
    return pairs, counts


@torch.inference_mode()
def score_codes(model, codes):
    """Batch of equal-length sentences. The first observed image is shared."""
    batch, length, width = codes.shape
    history = model.history_codes(codes)
    density = model.writer.log_prob(history[:, :-1].reshape(-1, history.shape[-1]),
                                    codes[:, 1:].reshape(-1, width)).reshape(batch, length - 1)
    logits = model.stop(history).squeeze(-1)
    ending = F.logsigmoid(-logits[:, :-1]).double().sum(-1) + F.logsigmoid(logits[:, -1]).double()
    return torch.stack((density.double().sum(-1), ending), dim=-1)


@torch.inference_mode()
def score_snapshot(model, pairs, start):
    tiles = sorted({tile for _, _, one, two in pairs for sentence in (one, two) for tile in sentence})
    packed = np.frombuffer(b''.join(tiles), dtype=np.uint8).reshape(-1, 128)
    ink = torch.from_numpy(np.unpackbits(packed, axis=1).reshape(-1, 1, 32, 32))
    features = torch.cat([model.encode_ink(batch) for batch in ink.split(96)])
    indices = {tile: i for i, tile in enumerate(tiles)}
    by_length = defaultdict(list)
    for i, (_, _, one, two) in enumerate(pairs):
        by_length[len(one)].append((i, [[indices[t] for t in s] for s in (one, two)]))
    scores = np.empty((len(pairs), 2, 2), dtype=np.float64)
    for length, group in by_length.items():
        for first in range(0, len(group), 8):
            guard(start)
            batch = group[first:first + 8]
            ids = torch.tensor([ids for _, ids in batch]).reshape(-1, length)
            scores[[i for i, _ in batch]] = score_codes(model, features[ids]).numpy().reshape(-1, 2, 2)
    require(np.isfinite(scores).all(), 'Nonfinite sentence score')
    return scores


def credits(margin):
    return np.where(margin > EPSILON, 1., np.where(margin < -EPSILON, 0., .5))


def summarize(pairs, margins):
    labels = np.asarray([r[0] for r in pairs])
    groups = sorted(set(labels))
    table, metrics, macro = {}, {}, {}
    for key, values in margins.items():
        scores = credits(values)
        means = [float(scores[labels == g].mean()) for g in groups]
        macro[key] = np.asarray(means)
        metrics[key] = dict(wins=int((scores == 1).sum()), ties=int((scores == .5).sum()),
                            losses=int((scores == 0).sum()), macro_accuracy=float(np.mean(means)))
        table[key] = dict(zip(groups, means))
    draws = np.random.default_rng(20261118).integers(0, len(groups), size=(10000, len(groups)))
    final = macro[str(STEPS[-1])]
    quantities = {'final_accuracy': final, 'final_minus_initial': final - macro['0'],
                  'final_minus_bigram': final - macro['bigram']}
    intervals = {k: np.quantile(v[draws].mean(1), [.025, .975]).tolist() for k, v in quantities.items()}
    return dict(metrics=metrics, by_paradigm=table, counts=dict(Counter(labels)), descriptive_intervals=intervals)


def main():
    start = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only')
    guard(start, entering=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    fit = load(FIT)
    require(sha(REG) == fit['registration_sha256'] and sha(RUN / 'audit.json') == fit['audit_sha256'], 'Fit admission')
    arm, = [a for a in fit['arms'] if a['components'] == 8]
    require(tuple(a['update'] for a in arm['snapshots']) == STEPS, 'All five fixed checkpoints required')
    OUTPUT.mkdir(exist_ok=False)
    baseline = training_baseline(start)
    pairs, counts = selection(baseline)
    bigram = np.array([[baseline.score(one), baseline.score(two)] for _, _, one, two in pairs])
    roster = [dict(paradigm=uid, pair_id=idx, good_sha256=hashlib.sha256(b''.join(a)).hexdigest(),
                   bad_sha256=hashlib.sha256(b''.join(b)).hexdigest(), bigram_scores=scores.tolist())
              for (uid, idx, a, b), scores in zip(pairs, bigram)]
    write_new(OUTPUT / 'pairs.json', roster)
    print(f'Selected {len(pairs)} pairs across {len(set(p[0] for p in pairs))} paradigms', flush=True)
    margins = {'bigram': bigram[:, 0] - bigram[:, 1]}
    model = ContinuousImageLanguage(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()), components=8).eval()
    for snapshot in arm['snapshots']:
        step = str(snapshot['update'])
        path = ROOT / snapshot['file']
        require(sha(path) == snapshot['sha256'], 'Snapshot differs')
        state = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        scores = score_snapshot(model, pairs, start)
        delta = scores[:, 0] - scores[:, 1]
        margins[step] = delta.sum(1)
        write_new(OUTPUT / f'scores_{step}.json', dict(checkpoint_sha256=snapshot['sha256'],
                  columns=['code_log_density', 'log_stop_path'], scores=scores.tolist()))
        print(f'Completed update {step}', flush=True)
    guard(start)
    files = {p.name: sha(p) for p in sorted(OUTPUT.glob('*.json'))}
    report = dict(**summarize(pairs, margins), exclusions=counts, pairs=len(pairs),
                  train_records=baseline.record_count, train_target_positions=baseline.targets,
                  train_visual_types=len(baseline.images), files=files, archive_sha256=ARCHIVE_SHA,
                  protocol_sha256=sha(PROTOCOL), script_sha256=sha(Path(__file__)), fit_sha256=sha(FIT),
                  rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  elapsed_seconds=time.monotonic() - start, torch_version=str(torch.__version__),
                  cpu_threads=2, model_updates=0, autonomous_streams_scored=0)
    write_new(OUTPUT / 'report.json', report)
    print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    main()
