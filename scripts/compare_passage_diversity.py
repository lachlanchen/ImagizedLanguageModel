"""Compare repeated and distinct pictured passages at matched exposure and compute."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from scripts.confirm_joint_reference_writer import OUT as PREVIOUS, integrated_check
from scripts.confirm_visual_writer_groups import prepare
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_local_teacher_writer import FRESH, infer
from scripts.fit_local_teacher_writer import DATA, ROOT, OUT as HINTS, load_split
from scripts.fit_physical_answer_preference import OUT as JOINT, fit
from scripts.fit_pictured_idiom_alignment import local_counts
from scripts.fit_pictured_idioms import guard
from scripts.measure_expression_concentration import frequencies
from scripts.score_visual_writer_transfer import matching_pairs, paired_interval

OUT = ROOT/'artifacts/passage_diversity_v1'
MODELS = ROOT/'models/pictured_idiom_zh'
PUBLIC = ROOT/'references/evidence/expression_preference_20261118'
ARMS = ('repeat', 'diverse')


def verified_split(folder):
    receipt = load(folder/'preparation.json')
    for file, key in [('pixels.npy','pixels_sha256'), ('contexts.npz','contexts_sha256')]:
        require(sha(folder/file) == receipt[key], f'Changed {folder.name}/{file}')
    return load_split(folder/'contexts.npz')


def original():
    report = load(ROOT/'references/evidence/pictured_idioms_20261118/preparation.json')
    for split in ('train','validation'):
        require(sha(DATA/(split+'.npz')) == report['splits'][split]['sha256'], 'Original data changed')
    return load_split(DATA/'train.npz')


def prepare_data(start):
    OUT.mkdir()
    train = original()
    development = load_split(DATA/'validation.npz')
    older = [verified_split(k) for k in (FRESH, HINTS/'confirmation', JOINT/'confirmation', PREVIOUS)]
    old_bank = np.load(PREVIOUS/'pixels.npy')
    fresh_dir, pool_dir = OUT/'fresh', OUT/'pool'
    fresh_dir.mkdir()
    fresh, bank = prepare(train, development, old_bank, buckets=range(52,60), destination=fresh_dir, excluded=older)
    guard(start)
    pool_dir.mkdir()
    pool, _ = prepare(train, development, bank, buckets=range(64,104), destination=pool_dir, excluded=[*older,fresh])
    size = len(train['target'])
    require(size == 71439 and len(pool['target']) >= size, 'Insufficient eligible distinct passages')
    selected = np.random.default_rng(844).choice(len(pool['target']), size=size, replace=False)
    np.save(OUT/'selected.npy', selected)
    used = set(map(bytes, train['context'])) | set(map(bytes, pool['context'][selected]))
    require(len(used) == 2*size and used.isdisjoint(map(bytes, fresh['context'])), 'Repeated pictured training context')
    write_new(OUT/'data.json', dict(original=size, eligible=len(pool['target']), new_eval=len(fresh['target']),
              selected_sha256=sha(OUT/'selected.npy'), fresh=load(fresh_dir/'preparation.json'),
              pool=load(pool_dir/'preparation.json'), protocol_sha256=sha(ROOT/'references/passage_diversity_protocol.md')))


def corpora():
    train = original()
    info = load(OUT/'data.json')
    require(sha(OUT/'selected.npy') == info['selected_sha256'], 'Selection changed')
    pool = verified_split(OUT/'pool')
    sel = np.load(OUT/'selected.npy')
    augmented = {k:np.concatenate([v, pool[k][sel]]) for k,v in train.items()}
    return train, augmented


def load_initial():
    release = load(PUBLIC/'release.json')
    base = MODELS/'reader_writer.safetensors'
    patch = MODELS/'joint_reference_writer.safetensors'
    require(sha(base)==release['base_sha256'] and sha(patch)==release['sha256'], 'Public checkpoint changed')
    initial = load_file(str(base)) | load_file(str(patch))
    path = JOINT/'reference/weights.pt'
    require(sha(path) == load(path.parent/'fit.json')['weights_sha256'], 'Archived start changed')
    archive = torch.load(path, weights_only=True, map_location='cpu')
    require(initial.keys()==archive.keys() and all(torch.equal(v,archive[k]) for k,v in initial.items()),
            'Training state differs from the released model')
    return initial


def physical_bank(initial, pixels):
    model = PicturedIdiom(initial['landmarks']).cuda().eval().requires_grad_(False)
    model.load_state_dict(initial)
    images = torch.as_tensor(np.unpackbits(pixels,axis=1), device='cuda').bool()
    bits = images[:,model.landmarks].float()
    drawings = torch.cat([model.motor(block)>0 for block in bits.split(128)])
    reachable = (drawings==images).all(-1).cpu().numpy()
    return visual_input(images,model.landmarks), bits, drawings[:,model.landmarks].float(), reachable


def train_pair(start):
    train, augmented = corpora()
    fresh = verified_split(OUT/'fresh')
    initial = load_initial()
    features, bits, feedback, exact = physical_bank(initial,np.load(OUT/'pool/pixels.npy'))
    coverage = {}
    for name, y in [('original',train['target']), ('new',augmented['target'][len(train['target']):]),
                    ('fresh',fresh['target'])]:
        glyphs = np.unique(y)
        coverage[name] = dict(cells=y.size, exact_cells=int(exact[y].sum()), types=len(glyphs),
                exact_types=int(exact[glyphs].sum()), fully_reachable=int(exact[y].all(1).sum()), rows=len(y))
    for name in ARMS:
        torch.manual_seed(720)
        x,y = [augmented[k] if name=='diverse' else np.tile(train[k],(2,1)) for k in ('context','target')]
        require(len(y)==2*71439, 'Exposure mismatch')
        x,y = [torch.as_tensor(v.astype(np.int32),device='cuda') for v in (x,y)]
        fit('reference',initial,x,y,None,features,bits,feedback,start,epochs=4,destination=OUT/name)
    write_new(OUT/'fits.json', dict(coverage=coverage, exposures=4*len(y), updates=4*((len(y)+191)//192),
             arms={a:load(OUT/a/'fit.json') for a in ARMS}, device=torch.cuda.get_device_name()))


def summarize(archive, test, bank, train, original_frequency, pair):
    x,y = test['context'],test['target']
    ink, length, stopped = (archive[k] for k in ('packed','length','stopped'))
    words = [a[:n].tobytes() for a,n in zip(ink,length)]
    hit = stopped & (length==4) & (ink[:,:4]==bank[y]).all((1,2))
    own = Counter(map(bytes,train['target']))
    i,j = pair
    absent = ~(y[:,:,None] == x[:,None,:]).any((1,2))
    report = dict(exact=int(hit.sum()), **frequencies(words), stopped=int(stopped.sum()),
                  all_absent=int(absent.sum()), absent_correct=int(hit[absent].sum()),
                  pairs=len(i), both_correct=int((hit[i]&hit[j]).sum()))
    for key, freq in [('original_frequency',original_frequency), ('arm_frequency',own)]:
        counts = np.array([freq[v.tobytes()] for v in y])
        bounds = [(0,0),(1,5),(6,20),(21,max(21,int(counts.max())))]
        report[key] = []
        for lo,hi in bounds:
            mask = (counts>=lo) & (counts<=hi)
            report[key].append(dict(lo=lo,hi=hi,cases=int(mask.sum()),correct=int(hit[mask].sum())))
    return hit,report


def evaluate(start):
    train, augmented = corpora()
    original_frequency = Counter(map(bytes,train['target']))
    bank = np.load(OUT/'pool/pixels.npy')
    reports = {}
    for name, folder in [('fresh',OUT/'fresh'), ('preceding',PREVIOUS)]:
        test = verified_split(folder)
        previous_bank = np.load(folder/'pixels.npy')
        require(np.array_equal(previous_bank,bank[:len(previous_bank)]), 'Incompatible physical bank')
        local = {k:int(local_counts(training,test).sum()) for k,training in zip(ARMS,(train,augmented))}
        pair = matching_pairs(test['context'],test['target'],test['group'])
        root = OUT/('evaluation_'+name)
        root.mkdir()
        hits,stats = {},{}
        for arm in ('start',*ARMS):
            training = augmented if arm=='diverse' else train
            model = JOINT/'reference' if arm=='start' else OUT/arm
            dest = PREVIOUS/'reference' if (arm,name)==('start','preceding') else root/arm
            if dest != PREVIOUS/'reference':
                infer(model/'weights.pt',dest,training,test,bank,start)
            else:
                require(sha(dest/'generated.npz')==load(PREVIOUS/'result.json')['outcomes']['reference']['archive_sha256'],
                        'Changed earlier starting streams')
            hits[arm],stats[arm] = summarize(load_split(dest/'generated.npz'),test,bank,training,original_frequency,pair)
            recorded = load(dest/'result.json')
            require(stats[arm]['exact']==recorded['complete_exact'], 'Physical recount differs')
            stats[arm].update(known=recorded['training_target_shape_hits'],emitted=recorded['emitted_cells'],
                              archive_sha256=sha(dest/'generated.npz'))
        comparison = {}
        for control in ('start','repeat'):
            a,b = hits['diverse'],hits[control]
            comparison[control] = dict(gained=int((a&~b).sum()),lost=int((b&~a).sum()),
                                        interval_pp=paired_interval(a,b,test['group']))
        reports[name] = dict(rows=len(test['group']), groups=len(np.unique(test['group'])), local=local,
                             source=frequencies(map(bytes,test['target'])), arms=stats, comparisons=comparison)
        print(name,{k:v['exact'] for k,v in stats.items()},local,flush=True)
    fresh, prev = reports['fresh'],reports['preceding']['arms']
    arms = fresh['arms']
    candidate = arms['diverse']
    criteria = dict(interval=all(v['interval_pp'][0]>0 for v in fresh['comparisons'].values()),
                    hard_pairs=all(candidate['both_correct']>=arms[k]['both_correct'] for k in ('start','repeat')),
                    ink=candidate['known']>=.99*candidate['emitted'],
                    concentration=candidate['most_frequent_ten']<arms['repeat']['most_frequent_ten'],
                    preceding=candidate['exact']>prev['repeat']['exact'])
    reports['criteria'] = criteria
    if all(criteria.values()):
        weights = torch.load(OUT/'diverse/weights.pt',weights_only=True,map_location='cpu')
        net = PicturedIdiom(weights['landmarks']).cuda().eval()
        net.load_state_dict(weights)
        reports['historical'] = integrated_check(net,verified_split(OUT/'fresh'),bank,start,
                                                  generated=OUT/'evaluation_fresh/diverse/generated.npz')
    write_new(OUT/'evaluation.json',reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','train','evaluate'])
    phase = parser.parse_args().phase
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = time.monotonic()
        guard(start)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        require('4090' in torch.cuda.get_device_name(), 'Expected registered GPU')
        {'prepare':prepare_data,'train':train_pair,'evaluate':evaluate}[phase](start)
        guard(start)
        write_new(OUT/(phase+'_resources.json'), dict(seconds=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(),program_sha256=sha(Path(__file__))))
