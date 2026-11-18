"""Compare aligned visual and character-indexed learning on identical passages."""
import hashlib
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn

from ilm.visual_lm.visual_successor import VisualSuccessor
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.measure_visual_successors import (CMRC, HISTORY, OUT as VISUAL, ROOT, SEED, SOURCE,
    SOURCE_HASH, assignment, digest, fit, guard)
from scripts.prepare_visual_passages import MANIFEST
from scripts.prepare_visual_sequence_corpus import Rasterizer, context_hash

OUT = ROOT/'artifacts/visual_successor_cost_v1'


class CharacterSuccessor(VisualSuccessor):
    def __init__(self, vocabulary):
        super().__init__()
        self.shape = nn.Embedding(vocabulary, 96)

    @staticmethod
    def pool(indices):
        return indices


def recover(roster):
    manifest = load(MANIFEST)
    source = CMRC/'cmrc2018_train.json'
    require(sha(source) == manifest['source_sha256'] and sha(SOURCE) == SOURCE_HASH, 'Source mismatch')
    by_hash = {context_hash(p['context']): p['context'] for a in load(source)['data'] for p in a['paragraphs']}
    renderer = Rasterizer()
    modern = []
    for r in manifest['passages']:
        txt = by_hash[r['group']]
        require(len(txt) == r['length'] and hashlib.sha256(b''.join(renderer.tile(c) for c in txt)).hexdigest()
                == r['ink_sha256'], 'Source order or training pixels differ')
        modern.append(txt)
    vocabulary = {c: i for i, c in enumerate(sorted(set(''.join(modern))), start=2)}
    index = lambda s: [vocabulary.get(c, 1) for c in s]
    by_article = {(digest(a['title']), digest(p['context'])): p['context']
                  for a in load(SOURCE)['data'] for p in a['paragraphs']}
    histories, candidates = [], []
    for r in roster:
        txt = by_article[r['article_sha256'], r['paragraph_sha256']]
        i, j = r['offsets']
        a, b, yy = txt[i-HISTORY:i], txt[j-HISTORY:j], txt[i]+txt[j]
        raw = b''.join(renderer.tile(c) for c in a+b+yy)
        actual = hashlib.sha256(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).tobytes())
        require(actual.hexdigest() == r['images_sha256'], 'Evaluation pairing differs')
        histories.extend([index(a), index(b)])
        candidates.extend([index(yy), index(yy)])
    streams = [''.join(map(chr, index(s))) for s in modern]
    return streams, vocabulary, torch.tensor(histories), torch.tensor(candidates)


@torch.inference_mode()
def score(model, h, y):
    scores = []
    for x, yy in zip(h.split(64), y.split(64)):
        q = model.queries(model.shape(x))[:, -1:]
        k = model.keys(model.shape(yy))
        scores.append((10 * (q @ k.transpose(1, 2))).squeeze(1))
    return torch.cat(scores).double().numpy().reshape(-1, 2, 2)


def main():
    start = time.monotonic()
    guard(start)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only')
    visual = load(VISUAL/'result.json')
    require(all(sha(VISUAL/p) == v for p,v in visual['files'].items()), 'Visual run changed')
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    modern, vocab, h, y = recover(load(VISUAL/'articles.json'))
    prepared = time.monotonic()-start
    torch.manual_seed(SEED)
    model = CharacterSuccessor(len(vocab)+2)
    # Construction initializes the same query and key layers before replacing shape.
    torch.manual_seed(SEED)
    initial = VisualSuccessor()
    require(all(torch.equal(v, initial.state_dict()[k]) for k,v in model.state_dict().items()
                if not k.startswith('shape.')), 'Query or key initialization differs')
    initial_score = score(model.eval(), h, y)
    training = fit(model.train(), torch.arange(len(vocab)+2), modern, False, start)
    trained_score = score(model.eval(), h, y)
    torch.save(dict(model=model.state_dict(), vocabulary=vocab), OUT/'character.pt')
    np.savez_compressed(OUT/'scores.npz', initial=initial_score, final=trained_score)
    with np.load(VISUAL/'scores.npz', allow_pickle=False) as f:
        vis, _ = assignment(f['aligned_normal'])
    ini, initial_metrics = assignment(initial_score)
    char, final_metrics = assignment(trained_score)
    draws = np.random.default_rng(SEED+1).integers(0, len(vis), (10000, len(vis)))
    intervals = {}
    for k, dif in [('visual_minus_character', vis-char), ('character_minus_initial', char-ini)]:
        intervals[k] = [float(dif.mean()), *np.quantile(dif[draws].mean(1), [.025,.975]).tolist()]
    guard(start)
    write_new(OUT/'result.json', dict(training=training, initial=initial_metrics, final=final_metrics,
        intervals=intervals, measured_visual_fit_seconds=visual['fits']['aligned']['seconds'],
        runtime_ratio_character_over_visual=training['seconds']/visual['fits']['aligned']['seconds'],
        visual_result_sha256=sha(VISUAL/'result.json'), trainable_parameters=sum(p.numel() for p in model.parameters()),
        visual_parameters=visual['trainable_parameters'], preparation_seconds=prepared,
        total_seconds=time.monotonic()-start, cpu_threads=2, rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        vocabulary=len(vocab), unknown_test_target_occurrences=int((y[::2] == 1).sum()),
        unknown_history_occurrences=int((h == 1).sum()), files={p.name:sha(p) for p in OUT.iterdir()},
        script_sha256=sha(Path(__file__)), protocol_sha256=sha(ROOT/'references/visual_successor_cost_protocol.md'),
        matched_general_language_efficiency_measured=False))
    print(training, final_metrics, intervals, flush=True)


if __name__ == '__main__':
    main()
