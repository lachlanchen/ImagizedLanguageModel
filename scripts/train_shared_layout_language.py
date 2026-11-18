"""Train and compare a coarse-to-fine physical writer on real Chinese passages."""
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from ilm.visual_lm.layout_patch_language import LayoutPatchLanguage
from ilm.visual_lm.patch_count_language import assemble, tile_features
from scripts.compare_visual_passage_registers import physical_records
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.measure_visual_successors import ROOT, guard
from scripts.train_patch_count_language import OUT as ORIGINAL, fresh_articles, rolls, test_cases, tiles_and_streams

OUT = ROOT/'artifacts/shared_layout_language_v1'
PROTOCOL = ROOT/'references/shared_layout_writing_protocol.md'


def preflight(bank, streams):
    data = streams[:8]
    positions = sum(len(x)-1 for x in data)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        for rotated in (False, True):
            model = LayoutPatchLanguage.fit(bank, data, rotated=rotated)
            prompt = bank[data[0][:32]]
            require(model.ending[-1].counts.sum() == positions+len(data), 'Real endings')
            hist = model.history(prompt)
            p = model.layout_probabilities(hist)
            layout = int(model.layouts[p.argmax()])
            for j, (_,_,coarse,joined) in enumerate(model.parts):
                dist = model.patch_probabilities(j, layout, 0)
                require(np.isclose(dist.sum(),1) and np.all(dist>0), 'Normalized local law')
                require(joined.counts.sum() == coarse.counts.sum() == positions, 'Patch training exposure')
            require(np.isclose(p.sum(),1) and np.all(p>0), 'Normalized layout law')
            original = model.draw(hist)
            require(np.array_equal(original,assemble(tile_features(original[None])[1][0])), 'Ink bit order')
            path = Path(tmp)/f'{rotated}.npz'
            model.save(path)
            restored = LayoutPatchLanguage.load(path)
            actual, stopped = restored.generate(prompt, maximum_new=8)
            expected, expected_stop = model.generate(prompt, maximum_new=8)
            require(np.array_equal(expected,actual) and expected_stop==stopped, 'Restored free roll')
    return dict(passages=len(data), targets=positions, discarded_models=2)


def physical_metrics(pixels, stopped, targets, bank):
    exact = (pixels==targets).all((1,2)) & ~stopped
    nonempty = pixels.any((1,2)) & ~stopped
    records = np.packbits(pixels.reshape(-1,1024),axis=1)
    in_bank = np.array([r.tobytes() in bank for r in records]) & nonempty
    return exact, dict(exact=int(exact.sum()), both=int(exact.reshape(-1,2).all(1).sum()),
                       known_nonempty=int(in_bank.sum()), blank=int((~nonempty & ~stopped).sum()))


def main():
    start = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use the registered CPU execution')
    guard(start)
    OUT.mkdir(exist_ok=False)
    previous = load(ORIGINAL/'result.json')
    require(sha(ORIGINAL/'result.json') == sha(ROOT/'references/evidence/patch_count_writing_20261118/result.json'), 'Original trial changed')
    handles, _, modern, corpus = physical_records(start)
    bank, streams = tiles_and_streams(handles,modern)
    inventory = set(handles)
    proof = preflight(bank,streams)
    rows = fresh_articles(handles,corpus)
    require([r for _,_,r in rows] == load(ORIGINAL/'articles.json') and len(rows)==334, 'Exact original paired histories')
    del handles, modern, corpus
    seen = np.unique(np.concatenate([s[1:] for s in streams]))
    layouts,_ = tile_features(bank[seen])
    _,size = np.unique(layouts,return_counts=True)
    targets_count = sum(len(s)-1 for s in streams)
    require(targets_count==1047284 and len(streams)==2127, 'Exposure')
    result = dict(preflight=proof, training_passages=len(streams), transitions=targets_count,
                  training_target_bitmaps=len(seen), observed_layouts=len(size), singleton_layouts=int((size==1).sum()),
                  model_sha256=sha(ROOT/'ilm/visual_lm/layout_patch_language.py'), program_sha256=sha(Path(__file__)),
                  protocol_sha256=sha(PROTOCOL), source_trial_sha256=sha(ORIGINAL/'result.json'), arms={})
    for arm,rotated in [('ordered',False),('rotated',True)]:
        guard(start)
        folder = OUT/arm
        folder.mkdir()
        began=time.monotonic()
        actor=LayoutPatchLanguage.fit(bank,streams,rotated=rotated)
        seconds=time.monotonic()-began
        require(actor.ending[-1].counts.sum() == targets_count+len(streams) and
                actor.layout_suffix[-1].counts.sum()==targets_count and all(part[-1].counts.sum()==targets_count
                for part in actor.parts),'Complete train exposures')
        checkpoint=folder/'model.npz'
        actor.save(checkpoint)
        del actor
        actor=LayoutPatchLanguage.load(checkpoint)
        stats,exact=test_cases(actor,bank,rows,folder)
        stats['rolls']=rolls(actor,bank,rows,folder)
        old=ORIGINAL/arm/'successors.npz'
        require(sha(old)==previous['files'][f'{arm}/successors.npz'], 'Frozen comparator')
        with np.load(folder/'successors.npz') as f, np.load(old) as base:
            targets=f['targets']
            require(np.array_equal(targets,base['targets']), 'Pixel targets differ')
            old_exact,old_counts=physical_metrics(base['ink'],base['stopped'],targets,inventory)
            reproduced,counts=physical_metrics(f['ink'],f['stopped'],targets,inventory)
        require(np.array_equal(exact,reproduced),'Independent exact recount')
        target_layouts,_=tile_features(targets)
        conditional=np.stack([actor.draw_layout(z) for z in target_layouts])
        np.savez_compressed(folder/'given_source_layout.npz',ink=conditional)
        _,given=physical_metrics(conditional,np.zeros(len(conditional),dtype=bool),targets,inventory)
        dif=(exact.astype(float)-old_exact).reshape(-1,2).mean(1)
        boot=np.random.default_rng(20261122).integers(0,len(dif),(10000,len(dif)))
        lower,upper=np.quantile(dif[boot].mean(1),[.025,.975])
        result['arms'][arm]=dict(stats=stats, free=counts, previous=old_counts, given_source_layout=given,
            paired_exact_difference=[float(dif.mean()),float(lower),float(upper)], fit_seconds=seconds,
            checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=sha(checkpoint),
            physical_pass=bool(counts['known_nonempty']>=old_counts['known_nonempty']+100 and counts['blank']*2<=old_counts['blank']),
            contextual_exact_improves=bool(lower>0))
        print(arm, counts, given, old_counts, flush=True)
    guard(start)
    result['files']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    result['elapsed_seconds']=time.monotonic()-start
    result['rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    write_new(OUT/'result.json',result)


if __name__=='__main__':
    main()
