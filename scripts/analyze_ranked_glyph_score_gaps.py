#!/usr/bin/env python3
"""Compare already saved source/emitted scores; never load a neural model."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

OPERATION_BEGIN = time.perf_counter()

from scripts.review_joint_writer_physical import load, sha, require, write_new

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/global_glyph_ranking_v1'
OLD_EXPORT = ROOT / 'references/evidence/ranked_writer_whitespace_result_20261115.json'
OLD_REG = ROOT / 'references/evidence/ranked_writer_whitespace_registration_20261115.json'
RUN = ROOT / 'artifacts/ranked_glyph_score_gaps_v1'
PUBLIC = ROOT / 'references/evidence/ranked_glyph_score_gap_registration_20261116.json'
RESULT = ROOT / 'references/evidence/ranked_glyph_score_gap_result_20261116.json'
SOURCES = ('scripts/analyze_ranked_glyph_score_gaps.py', 'tests/test_ranked_glyph_score_gaps.py',
           'references/ranked_glyph_score_gap_protocol.md', 'scripts/review_joint_writer_physical.py')
GROUPS = {'development/book/fixed_cells', 'development/qa/fixed_cells', 'development/qa/proportional',
          'train/book/fixed_cells', 'train/book/proportional', 'train/qa/fixed_cells', 'train/qa/proportional'}
BOOKS = ('development/book/fixed_cells', 'train/book/fixed_cells', 'train/book/proportional')
FIELDS = ('positions', 'source_whitespace', 'false_ink', 'false_blanks', 'exact_bitmaps',
          'nonempty_mismatches', 'source_preferred', 'emitted_preferred', 'tied')


def validated(gold, emitted, values):
    import numpy as np
    n = len(gold)
    require(n > 0 and gold.shape == emitted.shape == (n, 1, 32, 32), 'Complete32x32images required')
    require(gold.dtype == emitted.dtype == np.uint8 and np.isin(gold, [0, 1]).all()
            and np.isin(emitted, [0, 1]).all(), 'Binary source/output images required')
    needed = {'source_logp', 'emitted_logp', 'source_blank', 'emitted_blank', 'saved_argmax_disagreements'}
    require(needed <= values.keys() and all(a.shape == (n,) and np.isfinite(a).all() for a in values.values()),
            'Complete finite saved score vectors required')
    require(values['source_blank'].dtype == values['emitted_blank'].dtype == np.bool_, 'Typed blank flags required')
    sb, eb = ~gold.reshape(n, -1).any(1), ~emitted.reshape(n, -1).any(1)
    require(np.array_equal(sb, values['source_blank']) and np.array_equal(eb, values['emitted_blank']),
            'Blank flags differ from original pixels')
    require(not values['saved_argmax_disagreements'].any(), 'Original argmax choices must reproduce')
    same = (gold == emitted).reshape(n, -1).all(1)
    s, e = values['source_logp'], values['emitted_logp']
    require(np.all(s <= 1e-5) and np.all(e <= 1e-5), 'Normalized log probabilities required')
    require(np.all(np.abs(s[same] - e[same]) <= 1e-5), 'Identical image scores differ')
    return sb, eb, same, s, e


def count_vectors(gold, emitted, values):
    sb, eb, same, s, e = validated(gold, emitted, values)
    mismatch = ~sb & ~eb & ~same
    source = mismatch & (s > e + .001)
    output = mismatch & (e > s + .001)
    return dict(positions=len(sb), source_whitespace=int(sb.sum()), false_ink=int((sb & ~eb).sum()),
        false_blanks=int((~sb & eb).sum()), exact_bitmaps=int(same.sum()), nonempty_mismatches=int(mismatch.sum()),
        source_preferred=int(source.sum()), emitted_preferred=int(output.sum()),
        tied=int((mismatch & ~source & ~output).sum()))


def count_scalars(gold, emitted, values):
    """Independent counting logic for verification; shared validation only."""
    sb, eb, same, s, e = validated(gold, emitted, values)
    result = Counter({k: 0 for k in FIELDS})
    for blank_source, blank_output, equal, source_score, output_score in zip(sb, eb, same, s, e):
        result['positions'] += 1
        result['source_whitespace'] += int(blank_source)
        result['false_ink'] += int(blank_source and not blank_output)
        result['false_blanks'] += int(not blank_source and blank_output)
        result['exact_bitmaps'] += int(equal)
        if not blank_source and not blank_output and not equal:
            result['nonempty_mismatches'] += 1
            if float(source_score) > float(output_score) + .001:
                result['source_preferred'] += 1
            elif float(output_score) > float(source_score) + .001:
                result['emitted_preferred'] += 1
            else:
                result['tied'] += 1
    return dict(result)


def decision(groups):
    require(set(groups) == GROUPS, 'All seven groups required')
    for row in groups.values():
        require(set(row) == set(FIELDS) and all(type(v) is int and v >= 0 for v in row.values()), 'Typed counts required')
        require(row['source_preferred'] + row['emitted_preferred'] + row['tied'] == row['nonempty_mismatches'],
                'Score preferences must partition nonempty mismatches')
    books = {}
    for key in BOOKS:
        row = groups[key]; n = row['nonempty_mismatches']; minimum = (n + 1) // 2
        books[key] = dict(nonempty_mismatches=n, source_preferred=row['source_preferred'],
                          required=minimum, passed=n > 0 and row['source_preferred'] >= minimum)
    passed = all(row['passed'] for row in books.values())
    return dict(book_groups=books, source_preference_rule_passed=passed,
        action=('After the current trial, consider a separately bound first-divergence/prefix diagnosis; '
                'complete-path source ranking alone is not a sufficient explanation.' if passed else
                'No search-only repair is supported by this rule; after the current trial, '
                'conditional source-mode learning remains a candidate for controlled testing.'),
        current_training_changed=False, old_gate_changed=False, autonomous_generation_authorized=False)


def payload():
    require(sha(OLD_EXPORT) == 'c9e6ba7cae33988b9d6a527075bd70a98e7b68a5aab1b8ab562c56f2d9adf508',
            'Use the original completed diagnostic export')
    export, oldreg = load(OLD_EXPORT), load(OLD_REG)
    require(sha(OLD_REG) == export['registration_sha256'], 'Original diagnostic registration changed')
    files = {}
    def bind(path, expected=None):
        name = str(path.relative_to(ROOT)); digest = sha(path)
        require(expected is None or digest == expected, 'Original input changed: ' + name)
        files[name] = digest
    bind(OLD_EXPORT); bind(OLD_REG)
    diag = OLD / 'whitespace_density/report.json'
    bind(diag, export['input_sha256'][str(diag.relative_to(ROOT))]); report = load(diag)
    require(report['complete'] and report['model_tensors_unchanged'] and report['actual_context_recomputed'],
            'Completed actual-context scores required')
    for name in ['attempts/start.json', 'attempts/end.json', 'registration.json']:
        p = OLD / 'whitespace_density' / name
        bind(p, export['input_sha256'][str(p.relative_to(ROOT))])
    end = load(OLD / 'whitespace_density/attempts/end.json')
    require(end['complete'] and end['error'] is None, 'Original diagnostic must be terminal and complete')
    q = OLD / 'qualify/report.json'; bind(q, oldreg['bindings_sha256'][str(q.relative_to(ROOT))])
    originals = {r['job']['id']: r for r in load(q)['records']}
    cases = []
    for row in report['cases']:
        p = ROOT / row['file']; bind(p, export['input_sha256'][row['file']])
        require(files[row['file']] == row['sha256'], 'Saved score manifest differs')
        original = originals[row['case']]; pixels = OLD / 'qualify/cases' / row['case'] / 'pixels.npz'
        bind(pixels, original['pixels_sha256'])
        require(files[str(pixels.relative_to(ROOT))] == oldreg['bindings_sha256'][str(pixels.relative_to(ROOT))],
                'Original source/output binding differs')
        require(row['positions'] == original['target_glyphs'], 'Keep complete source length')
        cases.append(dict(case=row['case'], positions=row['positions'], job=original['job'],
                          scores=row['file'], pixels=str(pixels.relative_to(ROOT))))
    require(len(cases) == len(originals) == 124 and len({c['case'] for c in cases}) == 124
            and {c['case'] for c in cases} == set(originals)
            and sum(c['positions'] for c in cases) == 12597, 'Keep every original position')
    sources = {}
    for name in SOURCES:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit new calculation before binding: ' + name)
        sources[name] = sha(ROOT / name)
    return dict(schema='ranked-glyph-score-gap-registration-v1', source_sha256=sources, input_sha256=files,
        cases=cases, positions=12597, numerical_margin_nats=.001, identical_path_tolerance_nats=1e-5,
        book_rule='At least half the nonempty mismatches, rounded upward, prefer the source in each of three book groups',
        maximum_seconds=180, maximum_rss_gib=2, minimum_host_gib=48, cpu_threads=2,
        prior_cohort_familiarity=True, independent_confirmation=False, new_model_calls=0,
        new_optimizer_updates=0, new_output_images=0, old_gate_changed=False, current_training_changed=False)


def protected_owners():
    h = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == h['active_model_worker_boot_id'],
            'Original owner boot identity changed')
    result = []
    for key, start in [('active_controller', 'active_controller_start_ticks'),
                       ('active_model_worker_pid', 'active_model_worker_start_ticks')]:
        pid = h[key]; raw = Path(f'/proc/{pid}/stat').read_text(); fields = raw[raw.rfind(')') + 2:].split()
        require(fields[19] == h[start] and fields[0] != 'Z', 'Preserve original live owner')
        result.append((pid, fields[19]))
    r = load(ROOT / 'references/evidence/white_target_ranking_registration_20261115.json')
    require(len(r['source_sha256']) == 115 and all(sha(ROOT / n) == v for n, v in r['source_sha256'].items()),
            'Current115sources changed')
    return result


def calculate(registration, scalar, guard):
    import numpy as np
    groups, cases = {}, []
    for row in registration['cases']:
        guard()
        with np.load(ROOT / row['scores'], allow_pickle=False) as a:
            values = {k: a[k] for k in a.files}
        with np.load(ROOT / row['pixels'], allow_pickle=False) as a:
            counts = (count_scalars if scalar else count_vectors)(a['gold_ink'], a['candidate_ink'], values)
        require(counts['positions'] == row['positions'], 'Incomplete saved case')
        key = '/'.join(row['job'][k] for k in ['split', 'kind', 'layout'])
        groups.setdefault(key, Counter()).update(counts); cases.append(dict(case=row['case'], group=key, **counts))
    groups = {k: dict(v) for k, v in sorted(groups.items())}
    totals = Counter({k: 0 for k in FIELDS})
    for row in groups.values():
        totals.update(row)
    total = dict(totals)
    require([total[k] for k in FIELDS[:5]] == [12597, 252, 25, 37, 4609], 'Original cohort totals differ')
    return dict(schema='ranked-glyph-score-gap-result-v1', complete=True, registration_sha256=sha(PUBLIC),
        groups=groups, totals=total, cases=cases, decision=decision(groups), prior_cohort_familiarity=True,
        independent_confirmation=False, original_scores_and_readings_changed=False, new_model_calls=0,
        new_optimizer_updates=0, new_output_images=0, current_training_changed=False,
        complete_language_unmeasured=True, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'analyze', 'verify']); args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only saved-data analysis required')
    began = OPERATION_BEGIN; owners = protected_owners()
    previous = sum(load(p)['seconds'] for p in RUN.glob('*_receipt.json'))
    previous += sum(load(p)['seconds'] for p in (RUN.parent / 'ranked_glyph_score_gap_failures_v1').glob('*.json'))
    def guard():
        mem = dict(r.split(':', 1) for r in Path('/proc/meminfo').read_text().splitlines())
        require(int(mem['MemAvailable'].split()[0]) >= 48 * 1024**2
                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**2
                and previous + time.perf_counter() - began < 180, 'CPU analysis resource ceiling')
    guard()
    with (ROOT / 'artifacts/ilm_cpu_analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = payload(); guard()
        if args.action == 'register':
            require(not RUN.exists() and not PUBLIC.exists(), 'Preserve existing registration')
            write_new(RUN / 'registration.json', registration); write_new(PUBLIC, registration)
        else:
            require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() == subprocess.check_output(
                ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit exact registration before calculation')
            require(load(PUBLIC) == registration, 'Bound calculation or saved input changed')
            require(args.action == 'verify' or not RESULT.exists(), 'Preserve existing result')
            value = calculate(registration, args.action == 'verify', guard); guard()
            require(protected_owners() == owners, 'Original live ownership changed before result publication')
            if args.action == 'verify':
                expected = load(RESULT); expected.pop('resources')
                require(value == expected, 'Independent scalar recount differs')
            else:
                value['resources'] = dict(elapsed_seconds=time.perf_counter() - began,
                    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                    all115training_sources_unchanged=True, original_live_owners_preserved=True)
                write_new(RUN / 'report.json', value); write_new(RESULT, value)
            print(json.dumps(dict(action=args.action, totals=value['totals'], decision=value['decision'])))
        require(protected_owners() == owners, 'Original live ownership changed during operation')
        guard()
        write_new(RUN / (args.action + '_receipt.json'), dict(action=args.action, seconds=time.perf_counter()-began,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            protected_owner_identities=owners, original115sources_unchanged=True, new_model_calls=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_new(RUN.parent / 'ranked_glyph_score_gap_failures_v1' / f'{time.time_ns()}.json',
            dict(action=sys.argv[1:] or None, error=repr(error), seconds=time.perf_counter()-OPERATION_BEGIN,
                 peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                 new_model_calls=0, new_optimizer_updates=0))
        raise
