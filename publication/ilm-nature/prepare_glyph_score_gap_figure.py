#!/usr/bin/env python3
"""Export completed saved-score counts for the native manuscript plate."""
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())

def owners():
    h = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']; result = []
    for key, start in [('active_controller', 'active_controller_start_ticks'),
                       ('active_model_worker_pid', 'active_model_worker_start_ticks')]:
        pid = h[key]; fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        assert fields[19] == h[start] and fields[0] != 'Z'; result.append((pid, fields[19]))
    return result

def main():
    began = time.perf_counter(); protected = owners()
    r = load(ROOT / 'references/evidence/white_target_ranking_registration_20261115.json')
    def guard():
        memory = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
        assert int(memory['MemAvailable'].split()[0]) >= 48 * 1024**2
        assert time.perf_counter()-began < 180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**2
        assert len(r['source_sha256']) == 115 and all(sha(ROOT / n) == d for n, d in r['source_sha256'].items())
    guard()
    paths = [ROOT / 'references/evidence' / n for n in [
        'ranked_glyph_score_gap_result_20261116.json', 'ranked_glyph_score_gap_verification_20261116.json',
        'ranked_glyph_score_gap_registration_20261116.json']]
    assert sha(paths[0]) == '929d22b15cd9813c6e49836170d2dddb9fbcb8c3070d51b6d20638905c7e1572'
    result, verification, registration = map(load, paths)
    assert verification['result_sha256'] == sha(paths[0]) and verification['registration_sha256'] == sha(paths[2])
    assert verification['vector_and_independent_scalar_counts_agree'] and verification['CPU_checks']['passed'] == 13
    assert not result['decision']['source_preference_rule_passed'] and result['new_model_calls'] == 0
    assert result['totals'] == dict(positions=12597, source_whitespace=252, false_ink=25, false_blanks=37,
        exact_bitmaps=4609, nonempty_mismatches=7926, source_preferred=717, emitted_preferred=7206, tied=3)
    groups = []
    for key, row in result['groups'].items():
        split, kind, layout = key.split('/')
        label = ('Dev' if split == 'development' else 'Train') + (' book' if kind == 'book' else ' QA')
        label += ' / ' + ('fixed' if layout == 'fixed_cells' else 'proportional')
        groups.append(dict(key=key, label=label, **row))
    books = []
    for key, row in result['decision']['book_groups'].items():
        assert not row['passed']
        label = ('Dev books' if key.startswith('development') else 'Train books')
        label += '\n' + ('fixed' if key.endswith('fixed_cells') else 'proportional')
        books.append(dict(key=key, label=label, **row,
            observed_fraction=round(row['source_preferred']/row['nonempty_mismatches'], 6),
            required_fraction=round(row['required']/row['nonempty_mismatches'], 6)))
    assert len(groups) == 7 and len(books) == 3
    out = dict(schema='glyph-score-gap-figure-data-v1', source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        exporter_sha256=sha(__file__), groups=groups, books=books, totals=result['totals'],
        numerical_margin_nats=registration['numerical_margin_nats'], chart_fraction_decimals=6,
        descriptive_reused_cohort=True, semantic_accuracy_measured=False, complete_language='unmeasured',
        new_model_calls=0, new_optimizer_updates=0, original_images_and_readings_unchanged=True)
    guard(); assert owners() == protected
    with (HERE / 'evidence/glyph_score_gap_figure_data.json').open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(dict(groups=7, books=3, new_model_calls=0, seconds=time.perf_counter()-began)))

if __name__ == '__main__':
    with (ROOT / 'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); main()
