#!/usr/bin/env python3
"""Authenticate the completed reset trial and recount every emitted pixel."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/predictive_state_distillation_preflight_v1'
OUT = ROOT / 'references/evidence/predictive_state_distillation_result_20261118.json'
REG = ROOT / 'references/evidence/predictive_state_distillation_registration_20261118.json'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only result export')
    began = time.perf_counter()
    from scripts.train_full_rank_visual_span import training_source
    reg = json.loads(REG.read_text())
    raw = json.loads((RUN/'result.json').read_text())
    assert raw['registration_sha256'] == sha(REG)
    assert not raw['support'] and not raw['full_fit_authorized'] and raw['all_trial_updates_reset']
    assert not raw['autonomous_language_measured'] and not list(RUN.glob('*.pt'))
    owner_path = ROOT/'artifacts/predictive_state_distillation_preflight_v1.exit.json'
    owner = json.loads(owner_path.read_text())
    assert owner['phase'] == 'terminal' and owner['exit_code'] == 0
    corpus, plan, _, _, _ = training_source(ROOT/reg['teacher'])
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in [REG, RUN/'result.json', owner_path, Path(__file__)]}
    target_shards = set()
    def recount(row):
        case = row['case']; assert case in reg['cases']
        source = corpus[case['corpus_index']]
        assert len(source['ink']) == case['glyphs']
        starts = [0,*source['span_glyph_ends'].tolist()[:-1]]
        ends = source['span_glyph_ends'].tolist()
        first = max(0, source['prompt_spans']-1)
        target = source['ink'][ends[first]:].numpy().astype(bool)
        lengths = np.array([b-a for a,b in zip(starts,ends)][first+1:],dtype=np.int32)
        path = RUN/row['raw_file']; assert sha(path) == row['raw_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files) == {'ink','span_lengths'}
            ink = z['ink']; assert ink.dtype == np.uint8 and np.all((ink==0)|(ink==1))
            assert ink.shape == target.shape and np.array_equal(z['span_lengths'],lengths)
            a,b = ink.astype(bool),target
            p = dict(positions=len(a),tp=int((a&b).sum()),fp=int((a&~b).sum()),fn=int((~a&b).sum()),
                false_blanks=int((~a.reshape(len(a),-1).any(1)&b.reshape(len(b),-1).any(1)).sum()),
                exact=int((a==b).reshape(len(a),-1).all(1).sum()),source_blanks=int((~b.reshape(len(b),-1).any(1)).sum()))
            p['f1'] = 2*p['tp']/max(1,2*p['tp']+p['fp']+p['fn'])
        assert p == row['pixels']
        assert row['provided_lengths'] and row['correct_image_history'] and not row['autonomous_language']
        bindings[str(path.relative_to(ROOT))]=sha(path)
        report=path.with_suffix('.json'); assert json.loads(report.read_text())==row
        bindings[str(report.relative_to(ROOT))]=sha(report)
        target_shards.update(ROOT/plan['data']/n for n in corpus.shards[case['shard']]['files'] if n.endswith('_pixels.npz'))
    for row in raw['initial']:
        recount(row)
    for arm in raw['arms']:
        assert [x['update'] for x in arm['updates']] == list(range(1,7))
        assert len(arm['assessment']) == len(reg['cases']) == 3
        for step,row in enumerate(arm['updates']):
            assert row['case'] == reg['cases'][step%3] and row['complete_record']
            assert len(row['losses']) == 6 and all(math.isfinite(v) for v in row['losses'].values())
            assert math.isfinite(row['gradient_norm']) and row['optimizer_moment_bytes']>0
            path=RUN/f"{arm['arm']}_update_{step+1:02d}.json";assert json.loads(path.read_text())==row
            bindings[str(path.relative_to(ROOT))]=sha(path)
        for row in arm['assessment']:
            recount(row)
        reset=RUN/f"{arm['arm']}_reset.json";check=json.loads(reset.read_text())
        assert check['model_exact'] and check['projection_exact'] and check['model_sha256']==raw['model_tensor_sha256']
        bindings[str(reset.relative_to(ROOT))]=sha(reset)
    same=RUN/'unchanged_other_losses.json';assert json.loads(same.read_text())['exact']
    bindings[str(same.relative_to(ROOT))]=sha(same)
    bindings.update({str(p.relative_to(ROOT)):sha(p) for p in target_shards})
    initial,control,candidate=raw['initial_pooled'],raw['arms'][0],raw['arms'][1]
    checks=dict(all_record_kl_down=all(b['predictive_kl']<a['predictive_kl'] for a,b in zip(raw['initial'],candidate['assessment'])),
        writing_gain_at_least_005=candidate['pooled']['f1']>=max(initial['f1'],control['pooled']['f1'])+.05,
        false_blanks_no_worse=candidate['pooled']['false_blanks']<=min(initial['false_blanks'],control['pooled']['false_blanks']))
    assert checks==raw['support_checks'] and all(v is False for v in checks.values())
    resources={k:v for k,v in raw['resources'].items() if k!='admission'}
    result={**raw,'resources':resources,'successful_supervisor_elapsed_seconds':owner['finished_unix']-owner['began_unix'],
        'bindings':bindings,'all_1656_output_tiles_recounted_from_original_arrays':True,
        'teacher_readout_cpu_parameters':math.prod(raw['teacher_readout_shape']),
        'export_elapsed_seconds':time.perf_counter()-began,
        'interpretation':'The six-update KL replacement fails its preflight. No full fit or working visual language is qualified. All trial weights reset. This does not disprove longer or different distillation procedures.',
        'limitations':['Only three already-seen training records; no generalization or autonomous answers measured.',
            'Correct source histories and supplied next-span lengths isolate physical writing; F1 is canonical pixel agreement, not semantic accuracy or readability.',
            'KL scalars derive from the bound neural execution, not an independent second model replay.',
            'Both arms have fresh moments; this is additional adaptation, not replay of an old training update.',
            'Inherited model/teacher pretraining and prior extraction/training costs remain separate.']}
    with OUT.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(file=str(OUT.relative_to(ROOT)),sha256=sha(OUT),support=False,
        tiles=1656,complete_training_positions=552,export_seconds=result['export_elapsed_seconds'])))


if __name__=='__main__':
    main()
