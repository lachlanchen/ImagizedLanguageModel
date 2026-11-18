"""Qualify the frozen joint reference writer on unseen groups and historical source plates."""
import fcntl
import time
import resource

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import PicturedIdiom
from ilm.visual_lm.source_plate import SourcePlateReader, StageSpanReader
from scripts.confirm_visual_writer_groups import prepare
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_local_teacher_writer import FRESH, infer
from scripts.fit_local_teacher_writer import DATA, GUIDED, OUT as HINTS, ROOT, load_split
from scripts.fit_physical_answer_preference import OUT as JOINT
from scripts.fit_pictured_idiom_alignment import local_counts
from scripts.fit_pictured_idioms import guard
from scripts.fit_source_plate_coverage import new_phrasing
from scripts.fit_source_plate_reader import DATA as PLATES, MODELS, assess, tensor
from scripts.fit_source_plate_spans import OUT as HISTORICAL, third_phrasing
from scripts.score_visual_writer_transfer import matching_pairs, paired_interval

OUT = ROOT/'artifacts/joint_reference_confirmation_v1'


def integrated_check(net, fresh, bank, start):
    state = load_file(str(MODELS/'historical_reader.safetensors'))
    plate, spans = SourcePlateReader().cuda().eval(), StageSpanReader().cuda().eval()
    for module, prefix in [(plate, 'reader.'), (spans, 'span.')]:
        module.load_state_dict({k.removeprefix(prefix): v for k,v in state.items() if k.startswith(prefix)})
    px = np.load(PLATES/'pixels.npy')
    z = load_split(PLATES/'development.npz')
    x, rows, lang, nref, truth = (z[k] for k in ('context','row','language','length','target'))
    other, expanded = new_phrasing(x, rows, lang, nref, px)
    third, more = third_phrasing(x, rows, lang, nref, px)
    moved = third.copy()
    moved[:,28:44:5] = np.roll(third[:,28:44:5], 1, axis=1)
    report = {}
    source_results = load(HISTORICAL/'result.json')
    for name, ids, pictures in [('intact',x,px), ('paraphrase',other,expanded), ('third',third,more),
                                ('relocated',moved,more), ('blank_question',x,px), ('blank_plate',x,px)]:
        packed, length, stopped, route = assess(net,plate,tensor(np.unpackbits(pictures, axis=1)).bool(),ids,name,start,spans)
        target = truth if name != 'relocated' else np.take_along_axis(moved[:,28:44:5], rows.clip(min=0), axis=1)
        same = (packed[:,:2] == px[target]).all(2)
        correct = stopped & (nref == length) & same[:,0] & ((nref == 1) | same[:,1])
        archived = HISTORICAL/(name+'.npz')
        require(sha(archived) == source_results['arms'][name]['sha256'], 'Historical outputs changed')
        old = load_split(archived)
        unchanged = ((packed == old['packed']).all((1,2)) & (length == old['length']) &
                     (stopped == old['stopped']) & (route == old['copy']))
        if name.startswith('blank'):
            require(not correct.any(), 'Historical erasure succeeded')
        else:
            require(unchanged.all() and route.all(), 'Historical regression')
        report[name] = dict(exact=int(correct.sum()), unchanged=int(unchanged.sum()), routed=int(route.sum()))
    ink, n, stop, route = assess(net,plate,tensor(np.unpackbits(bank, axis=1)).bool(),fresh['context'],'intact',start,spans)
    reference = load_split(OUT/'reference/generated.npz')
    require(not route.any() and all(np.array_equal(a,reference[k]) for a,k in
            [(ink,'packed'),(n,'length'),(stop,'stopped')]), 'Modern route or output changed')
    report['modern_preserved'] = len(route)
    return report


def main():
    start = time.monotonic()
    guard(start)
    require(not load(JOINT/'evaluation/result.json')['qualified'], 'Expected completed matched comparison')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    OUT.mkdir()
    train, val = [load_split(DATA/(k+'.npz')) for k in ('train','validation')]
    older = [FRESH, HINTS/'confirmation', JOINT/'confirmation']
    exclude = []
    for old in older:
        prep = load(old/'preparation.json')
        require(sha(old/'contexts.npz') == prep['contexts_sha256'] and sha(old/'pixels.npy') == prep['pixels_sha256'],
                'Prior confirmation changed')
        exclude.append(load_split(old/'contexts.npz'))
    fresh, px = prepare(train, val, np.load(older[-1]/'pixels.npy'), buckets=range(44,52),
                        destination=OUT, excluded=exclude)
    x,y,group = (fresh[k] for k in ('context','target','group'))
    local = local_counts(train, fresh)
    i,j = matching_pairs(x,y,group)
    known = set(map(tuple,train['target']))
    unseen = np.array([tuple(v) not in known for v in y])
    absent = ~(y[:,:,None] == x[:,None]).any((1,2))
    outcomes, hits = {}, {}
    for name, parent in [('guided',GUIDED/'distilled_targets'), ('reference',JOINT/'reference')]:
        infer(parent/'weights.pt', OUT/name, train, fresh, px, start)
        generated = load_split(OUT/name/'generated.npz')
        hit = generated['stopped'] & (generated['length'] == 4) & (generated['packed'][:,:4] == px[y]).all((1,2))
        hits[name] = hit
        stats = load(OUT/name/'result.json')
        require(hit.sum() == stats['complete_exact'], 'Recount differs')
        outcomes[name] = dict(exact=int(hit.sum()), both_correct=int((hit[i] & hit[j]).sum()), unseen=int(hit[unseen].sum()),
                              all_absent=int(hit[absent].sum()), known=stats['training_target_shape_hits'],
                              emitted=stats['emitted_cells'], distinct=stats['unique_correct'],
                              local_correct=int((local & hit).sum()), archive_sha256=sha(OUT/name/'generated.npz'))
    delta = paired_interval(hits['reference'],hits['guided'],group)
    c, b = outcomes['reference'], outcomes['guided']
    passed = bool(delta[0] > 0 and c['known'] >= .99 * c['emitted'] and c['both_correct'] >= b['both_correct'])
    result = dict(n=len(x), groups=len(set(group)), pairs=len(i), unseen=int(unseen.sum()),all_absent=int(absent.sum()),
                  local_exact=int(local.sum()),outcomes=outcomes,interval_pp=delta, passed_modern=passed,
                  gains=int((hits['reference'] & ~hits['guided']).sum()),losses=int((~hits['reference'] & hits['guided']).sum()))
    if passed:
        weights = torch.load(JOINT/'reference/weights.pt', weights_only=True, map_location='cpu')
        net = PicturedIdiom(weights['landmarks']).cuda().eval()
        net.load_state_dict(weights)
        result['integrated'] = integrated_check(net, fresh, px, start)
    result['qualified_for_cpu_check'] = passed and 'integrated' in result
    result['seconds'] = time.monotonic()-start
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['host_peak_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_new(OUT/'result.json',result)
    print(result, flush=True)


if __name__ == '__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
