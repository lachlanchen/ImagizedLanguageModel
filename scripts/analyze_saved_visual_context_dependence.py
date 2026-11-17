#!/usr/bin/env python3
"""Describe prompt-dependent raw-pixel variation in all 792 retained streams."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/trainable_visual_context_comparison_v1'
OUT = ROOT / 'references/evidence/visual_context_saved_dependence_20261117.json'
PROTOCOL = ROOT / 'references/visual_context_saved_dependence_protocol_20261117.md'
REG = ROOT / 'references/evidence/trainable_visual_context_comparison_registration_20261116.json'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    began = time.perf_counter()
    manifest_path = RUN / 'generation_manifest.json'
    score_path = RUN / 'review/scores.json'
    manifest, scores = load(manifest_path), load(score_path)
    require(manifest['registration_sha256'] == scores['registration_sha256'] == sha(REG),
            'Original comparison identity changed')
    frozen = load(RUN / 'review/readings_frozen.json')
    require(sha(RUN / 'review/readings_live.json') == frozen['readings_sha256'] == scores['readings_sha256'],
            'Frozen readings changed')
    expected = {(r['arm'], r['step'], r['profile'], r['case']): r for r in scores['records']}
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in
        [manifest_path, score_path, REG, PROTOCOL, Path(__file__), RUN / 'review/readings_frozen.json']}
    records = {}
    for name in manifest['runs']:
        folder = ROOT / name
        request, report = load(folder / 'request.json'), load(folder / 'report.json')
        require(report['request_sha256'] == sha(folder / 'request.json'), 'Generation request changed')
        arm, step, profile = folder.parent.name, report['optimizer_updates'], request['profile']
        require(request['seed'] == 202611153 and request['flow_steps'] == 8, 'Sampling configuration changed')
        report_hash = sha(folder / 'report.json')
        for p in (folder / 'request.json', folder / 'report.json'):
            bindings[str(p.relative_to(ROOT))] = sha(p)
        for row in report['records']:
            require(time.perf_counter() - began < 60, 'Saved-output CPU time ceiling')
            key = (arm, step, profile, row['id'])
            require(key in expected and key not in records, 'Unexpected or repeated stream')
            require(expected[key]['report_sha256'] == report_hash, 'Frozen score cohort changed')
            path = folder / row['id'] / 'ink.npz'
            digest = sha(path)
            require(digest == row['raw_sha256'], 'Raw emitted array changed')
            bindings[str(path.relative_to(ROOT))] = digest
            with np.load(path, allow_pickle=False) as array:
                ink = array['ink']
                require(ink.shape == (row['emitted_glyphs'], 1, 32, 32)
                        and np.isin(ink, [0, 1]).all(), 'Require original complete binary arrays')
                full_digest = hashlib.sha256(ink.tobytes()).hexdigest()
                first_count = row['emitted_pixel_feedback'][0]['glyphs'] if len(ink) else 0
                require(sum(x['glyphs'] for x in row['emitted_pixel_feedback']) == len(ink),
                        'Actual feedback spans do not cover the stream')
                records[key] = dict(case=row['id'], arm=arm, step=step, profile=profile,
                    condition=expected[key]['condition'], kind=expected[key]['kind'],
                    length=len(ink), full_digest=full_digest,
                    first_tile=ink[0].copy() if len(ink) else None,
                    first_span=ink[:first_count].copy(), prefix=ink[:16].copy(),
                    stopped=row['stopped'], capped=row['capped'])
    require(records.keys() == expected.keys() and len(records) == 792, 'Incomplete saved cohort')
    cohorts = defaultdict(list)
    for row in records.values():
        cohorts[(row['arm'], row['step'], row['profile'], row['kind'])].append(row)
    summaries, pairs = [], []
    for (arm, step, profile, kind), rows in sorted(cohorts.items()):
        normal = [r for r in rows if r['condition'] == 'normal']
        nonempty = [r for r in normal if r['length']]
        first = Counter(r['first_tile'].tobytes() for r in nonempty)
        full = Counter((r['length'], r['full_digest']) for r in nonempty)
        summaries.append(dict(arm=arm, step=step, profile=profile, kind=kind,
            normal=len(normal), nonempty=len(nonempty), unique_first_tiles=len(first),
            largest_first_tile_multiplicity=max(first.values(), default=0),
            unique_complete_nonempty_streams=len(full), largest_complete_multiplicity=max(full.values(), default=0)))
        indexed={(r['case'].removesuffix('_' + r['condition']), r['condition']): r for r in rows}
        chosen=[]
        for left in normal:
            base=left['case'].removesuffix('_normal')
            for condition in sorted({r['condition'] for r in rows} - {'normal'}):
                if (base, condition) in indexed:
                    chosen.append((left, indexed[(base, condition)], condition))
            if kind == 'qa' and base.endswith('_a'):
                require((base[:-1]+'b', 'normal') in indexed, 'Paired question missing')
                chosen.append((left, indexed[(base[:-1]+'b', 'normal')], 'paired_questions'))
        for left, right, comparison in chosen:
            present=left['length']>0 and right['length']>0
            matched=present and len(left['first_span'])==len(right['first_span'])
            common=min(16,left['length'],right['length'])
            pairs.append(dict(arm=arm, step=step, profile=profile, kind=kind,
                left=left['case'], right=right['case'], comparison=comparison,
                lengths=[left['length'],right['length']], both_nonempty=present,
                complete_nonempty_equal=present and left['length']==right['length'] and left['full_digest']==right['full_digest'],
                stopping_equal=(left['stopped'],left['capped'])==(right['stopped'],right['capped']),
                first_tile_hamming=int(np.count_nonzero(left['first_tile'] != right['first_tile'])) if present else None,
                first_span_lengths=[len(left['first_span']),len(right['first_span'])], matched_initial_noise_shape=matched,
                first_span_hamming=int(np.count_nonzero(left['first_span'] != right['first_span'])) if matched else None,
                common_prefix_tiles=common, common_prefix_hamming=int(np.count_nonzero(left['prefix'][:common] != right['prefix'][:common])) if common else None))
    pair_groups=defaultdict(list)
    for row in pairs:
        pair_groups[(row['arm'],row['step'],row['profile'],row['comparison'])].append(row)
    pair_summaries=[]
    for (arm,step,profile,comparison), rows in sorted(pair_groups.items()):
        matched=[r for r in rows if r['matched_initial_noise_shape']]
        distances=[r['first_tile_hamming'] for r in matched]
        pair_summaries.append(dict(arm=arm,step=step,profile=profile,comparison=comparison,
            pairs=len(rows),both_nonempty=sum(r['both_nonempty'] for r in rows),matched_first_span_lengths=len(matched),
            exact_nonempty_streams=sum(r['complete_nonempty_equal'] for r in rows),
            identical_first_spans=sum(r['first_span_hamming']==0 for r in matched),
            median_matched_first_tile_hamming=float(np.median(distances)) if distances else None))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak <= 1024**3 and time.perf_counter()-began < 60, 'Saved-output resource ceiling')
    result=dict(schema='ilm-saved-visual-context-dependence-v1',protocol_sha256=sha(PROTOCOL),
        streams=792,cohort_summaries=summaries,pair_summaries=pair_summaries,pairs=pairs,
        bindings=bindings,elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=peak,
        model_calls=0,readings_changed=False,original_scores_changed=False,goal_achieved=False,
        interpretation='Visible pixel dependence and repetition only. Matched seed/first-span shape controls do not establish semantic correctness or full conditional-distribution independence.')
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)),sha256=sha(OUT),streams=792,pairs=len(pairs),seconds=result['elapsed_seconds'],peak_rss_bytes=peak)))


if __name__ == '__main__':
    main()
