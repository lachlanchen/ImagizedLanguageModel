"""Score actual unrestricted next images and two free answers from the frozen writer."""
import hashlib
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.context_contrast_writer import draw_glyph, generate
from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.infer_continuous_image_language import image_cells
from scripts.measure_visual_successors import ROOT, SOURCE, SOURCE_HASH, digest, guard
from scripts.prepare_visual_sequence_corpus import Rasterizer

OUT = ROOT/'artifacts/context_contrast_drawing_v1'
BASE = ROOT/'references/evidence/visual_successor_20261118'
YAN = ROOT/'references/evidence/native_yan_cold_20261118'
PROTOCOL = ROOT/'references/context_contrast_drawing_protocol.md'
CHECKPOINT = ROOT/'artifacts/continuous_autoregressive_full_v1/k8/model_0013785.pt'


def unpack(renderer, text):
    packed = np.frombuffer(b''.join(renderer.tile(c) for c in text), dtype=np.uint8)
    return np.unpackbits(packed).reshape(-1, 1, 32, 32)


def recover():
    prior = load(BASE/'visual.json')
    roster_file = ROOT/'artifacts/visual_successor_v1/articles.json'
    require(sha(SOURCE) == SOURCE_HASH == prior['source_sha256'] and sha(roster_file) == prior['files']['articles.json'],
            'Original corpus or selected articles changed')
    articles = {(digest(a['title']), digest(p['context'])): p['context'] for a in load(SOURCE)['data'] for p in a['paragraphs']}
    renderer = Rasterizer()
    histories, targets = [], []
    for row in load(roster_file):
        text = articles[row['article_sha256'], row['paragraph_sha256']]
        i, j = row['offsets']
        h1, h2 = (unpack(renderer, text[k-32:k]) for k in (i, j))
        yy = unpack(renderer, text[i]+text[j])
        require(hashlib.sha256(h1.tobytes()+h2.tobytes()+yy.tobytes()).hexdigest() == row['images_sha256'],
                'Original physical pair changed')
        histories.extend([h1, h2])
        targets.extend(yy)
    require(len(histories) == 1024, 'Expected both contexts from every article')
    return torch.from_numpy(np.stack(histories)), np.stack(targets)


@torch.inference_mode()
def draw_all(model, histories, start):
    # Deduplication memoizes only physical prompt inputs for the frozen encoder.
    shape = histories.shape
    flat = np.packbits(histories.numpy().reshape(-1, 1024), axis=-1)
    unique, inverse = np.unique(flat.view('V128').reshape(-1), return_inverse=True)
    cells = torch.from_numpy(np.unpackbits(unique.view(np.uint8)).reshape(-1, 1, 32, 32))
    codes = torch.cat([model.encode_ink(b) for b in cells.split(96)])
    handles = torch.from_numpy(inverse.reshape(shape[:2]))
    arms = ('stock','density','contrast')
    outputs = {arm: [] for arm in arms}
    emitted, decisions = [], {arm: [] for arm in arms[1:]}
    for h in handles.split(16):
        guard(start)
        states = model.history_codes(codes[h])[:, -1]
        active = model.stop(states).flatten() < 0
        emitted.append(active.numpy())
        for arm in arms:
            images = torch.zeros(len(h), 1, 32, 32, dtype=torch.uint8)
            if active.any():
                if arm == 'stock':
                    images[active], _ = model.draw(states[active])
                else:
                    suffix = model.history_codes(codes[h[active, -1:]])[:, -1] if arm == 'contrast' else None
                    images[active], tr = draw_glyph(model, states[active], suffix)
                    decisions[arm].append(torch.stack([tr['eligible'], tr['selected'], tr['density_choice']], -1).byte().numpy())
            outputs[arm].append(images.numpy())
    return ({k: np.concatenate(v) for k,v in outputs.items()}, np.concatenate(emitted),
            {k: np.concatenate(v) if v else np.empty((0,model.writer.groups,3), np.uint8) for k,v in decisions.items()})


def ink_f1(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    denominator = a.sum(1)+b.sum(1)
    return np.divide(2*(a*b).sum(1), denominator, out=np.ones(len(a)), where=denominator>0)


def metrics(images, emitted, targets, repertoire):
    own = emitted * ink_f1(images, targets)
    crossed = emitted * ink_f1(images, targets.reshape(-1, 2, 1, 32, 32)[:, ::-1].reshape(targets.shape))
    hits = (images == targets).reshape(len(images), -1).all(1) & emitted
    packed = [r.tobytes() for r in np.packbits(images[emitted].reshape(int(emitted.sum()), 1024), axis=1)]
    paired = emitted.reshape(-1,2).all(1)
    changes = (images[::2] != images[1::2]).reshape(-1, 1024).sum(1)[paired]
    margin = (own-crossed).reshape(-1,2).sum(1)
    return dict(emitted=int(emitted.sum()), stopped=int((~emitted).sum()), distinct_bitmaps=len(set(packed)),
                nonempty=int(images[emitted].reshape(-1,1024).any(1).sum()), exact_source=int(hits.sum()),
                exact_training_bitmaps=sum(t in repertoire for t in packed), both_exact=int(hits.reshape(-1,2).all(1).sum()),
                mean_source_f1=float(own.mean()), mean_exchanged_f1=float(crossed.mean()), mean_assignment_margin=float(margin.mean()),
                both_generated=int(paired.sum()), identical_between_contexts=int((changes==0).sum()),
                mean_changed_pixels_between_contexts=float(changes.mean()) if len(changes) else None), margin


def page(path, cells):
    tiles = cells.reshape(-1,32,32)
    cols = min(32, max(1, len(tiles)))
    rows = max(1, (len(tiles)+cols-1)//cols)
    grid = np.zeros((rows*cols,32,32),dtype=np.uint8)
    grid[:len(tiles)] = tiles
    Image.fromarray(255-255*grid.reshape(rows,cols,32,32).transpose(0,2,1,3).reshape(rows*32,cols*32)).save(path)


def summarize_decisions(a):
    return dict(groups=len(a)*a.shape[1], eligible_count_histogram=np.bincount(a[:,:,0].flatten(), minlength=9).tolist(),
                differs_from_density_on_own_prefix=int((a[:,:,1]!=a[:,:,2]).sum()))


@torch.inference_mode()
def main():
    start = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU only')
    guard(start)
    require(sha(CHECKPOINT)==load(YAN/'result.json')['checkpoint_sha256'], 'Native final weights changed')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    model = ContinuousImageLanguage(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()), components=8).eval()
    model.load_state_dict(torch.load(CHECKPOINT, mmap=True, weights_only=True, map_location='cpu')['model'])
    histories, targets = recover()
    outputs, emitted, traces = draw_all(model, histories, start)
    np.savez_compressed(OUT/'drawing.npz', emitted=emitted, targets=targets, **outputs)
    np.savez_compressed(OUT/'choices.npz', **traces)
    # The training repertoire enters only this post-generation comparison.
    bank = ROOT/'artifacts/native_yan_code_cycle_v1/control.npz'
    require(sha(bank)==load(YAN/'code_cycle.json')['files_sha256'][bank.name], 'Repertoire differs')
    with np.load(bank, allow_pickle=False) as z:
        repertoire = {r.tobytes() for r in z['packed']}
    stats, margins = {}, {}
    for arm, pixels in outputs.items():
        stats[arm], margins[arm] = metrics(pixels, emitted, targets, repertoire)
    delta = margins['contrast']-margins['density']
    draws=np.random.default_rng(20261120).integers(0,len(delta),(10000,len(delta)))
    interval=np.quantile(delta[draws].mean(1),[.025,.975])
    print('Complete paired image draws', stats, flush=True)
    # The archived, externally evaluated question is separate from the blinded cohort.
    prompt=torch.from_numpy(image_cells(YAN/'prompt.png'))
    require(hashlib.sha256(prompt.numpy().tobytes()).hexdigest()==load(YAN/'result.json')['routes']['pictured']['receipt']['prompt_ink_sha256'],
            'Pictured input differs')
    yan={}
    for arm in ('density','contrast'):
        guard(start)
        ink, steps, stopped=generate(model,prompt,contrast=arm=='contrast',maximum_new=128)
        raw=ink.numpy()
        page(OUT/(arm+'_yan.png'),raw)
        np.savez_compressed(OUT/(arm+'_yan.npz'),ink=raw)
        decisions=np.empty((0,model.writer.groups,3),dtype=np.uint8) if not steps else torch.cat([
            torch.stack([t['eligible'],t['selected'],t['density_choice']],-1) for t in steps]).byte().numpy()
        yan[arm]=dict(glyphs=len(raw),stopped=stopped,unique=len({t.tobytes() for t in raw}),
                      nonempty=int(raw.reshape(len(raw),1024).any(1).sum()),**summarize_decisions(decisions))
    guard(start)
    write_new(OUT/'result.json',dict(checkpoint_sha256=sha(CHECKPOINT),protocol_sha256=sha(PROTOCOL),
        code_sha256={n:sha(ROOT/n) for n in ('ilm/visual_lm/context_contrast_writer.py','scripts/evaluate_context_contrast_drawing.py')},
        files={p.name:sha(p) for p in OUT.iterdir()},articles=len(delta),stats=stats,
        decisions={k:summarize_decisions(v) for k,v in traces.items()},yan=yan,
        contrast_minus_density_assignment_margin=[float(delta.mean()),*interval.tolist()],
        physical_improvement=bool(interval[0]>0 and stats['contrast']['exact_source']>stats['density']['exact_source']),
        evaluation_seconds=time.monotonic()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,cpu_threads=2,
        fitted_updates=0,blinded_streams_opened=0))
    print('Yan results', yan, flush=True)


if __name__=='__main__':
    main()
