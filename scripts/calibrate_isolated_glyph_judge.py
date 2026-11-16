#!/usr/bin/env python3
"""Register, transcribe once, then score a fixed training-only glyph judge."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from ilm.training_only import isolated_glyph_judge as judge

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
CORPUS = ROOT / 'data/visual_sequence_corpus_v1'
REFS = ROOT / 'artifacts/compact_visual_span_evaluation_v1/references.json'
PREVIOUS = ROOT / 'references/evidence/whole_answer_reward_calibration_registration_20261116.json'
RUN = ROOT / 'artifacts/isolated_glyph_judge_v1'
REG = ROOT / 'references/evidence/isolated_glyph_judge_registration_20261116.json'
OUT = ROOT / 'references/evidence/isolated_glyph_judge_calibration_20261116.json'
PROTOCOL = ROOT / 'references/isolated_glyph_judge_protocol_20261116.md'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def available_gib():
    line, = [s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')]
    return int(line.split()[1]) / 1024**2


def gpu_state():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(v[0]): dict(used_mib=int(v[1]), free_mib=int(v[2]))
            for v in (line.split(',') for line in raw.splitlines())}


def guard(began, limit, reserve=24):
    if time.monotonic() - began > limit:
        raise RuntimeError('Registered wall-time ceiling exceeded')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 > 2:
        raise RuntimeError('Client process exceeds 2 GiB RSS')
    if available_gib() < reserve:
        raise RuntimeError('Shared workstation host reserve exhausted')
    free = os.statvfs(ROOT)
    if free.f_bavail * free.f_frsize / 1024**3 < 48:
        raise RuntimeError('Project disk reserve exhausted')


def sources():
    old = load(PREVIOUS)
    bound = {}
    # Preserve the original data authentication, not obsolete executor code.
    for name, digest in old['input_sha256'].items():
        if not name.startswith(('data/', 'artifacts/')):
            continue
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen data changed: ' + name)
        bound[name] = digest
    for path in [PREVIOUS, PROTOCOL, Path(__file__),
                 ROOT / 'ilm/training_only/__init__.py',
                 ROOT / 'ilm/training_only/isolated_glyph_judge.py',
                 ROOT / 'tests/test_isolated_glyph_judge.py',
                 ROOT / 'scripts/prepare_compact_visual_span_evaluation.py',
                 ROOT / 'references/evidence/whole_answer_reward_calibration_20261116.json']:
        bound[relative(path)] = sha(path)
    return old['rows'], bound


class ReferencePixels:
    def __init__(self):
        self.shards = []; offset = 0
        for shard in load(CORPUS / 'report.json')['shards']:
            if shard['split'] != 'development':
                continue
            name, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
            self.shards.append((offset, offset + shard['records'], CORPUS / name, shard['files'][name]))
            offset += shard['records']
        self.path = self.data = None

    def __call__(self, index):
        first, _, path, digest = next(v for v in self.shards if v[0] <= index < v[1])
        if path != self.path:
            if sha(path) != digest:
                raise ValueError('Reference shard identity changed')
            with np.load(path, allow_pickle=False) as data:
                self.data = {n: data[n] for n in ('ink', 'record_glyph_offsets', 'prompt_glyphs')}
            self.path = path
        local = index - first
        begin, end = self.data['record_glyph_offsets'][local:local + 2]
        begin += self.data['prompt_glyphs'][local]
        packed = self.data['ink'][begin:end]
        if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != 128:
            raise ValueError('Keep original packed reference geometry')
        return [v.tobytes() for v in packed]


def register():
    began = time.monotonic(); guard(began, 300)
    rows, bound = sources()
    if RUN.exists() or REG.exists():
        raise FileExistsError('Preserve the existing registration and preparation')
    (RUN / 'cells').mkdir(parents=True)
    reference = ReferencePixels()
    refs = {r['id']: r for r in load(REFS)['reference_rows'] if r['kind'] == 'qa'}
    cells = {}; cases = []; stopped_positions = 0
    def store(packed):
        key = judge.cell_key(packed)
        if key not in cells:
            (RUN / 'cells' / (key + '.bin')).write_bytes(packed)
            (RUN / 'cells' / (key + '.png')).write_bytes(judge.cell_png(packed))
            cells[key] = dict(blank=not any(packed), png_sha256=sha(RUN / 'cells' / (key + '.png')))
        return key
    for row in rows:
        guard(began, 300)
        folder = OLD / 'evaluation/step_041298/core128' / (row['layout'] + '__' + row['case'])
        report = load(folder / 'report.json')
        with np.load(folder / 'ink.npz', allow_pickle=False) as data:
            ink = data['ink']
        if ink.dtype != np.uint8 or ink.shape[1:] != (1, 32, 32) or not np.isin(ink, [0, 1]).all():
            raise ValueError('Keep original emitted cells')
        if sha(folder / 'ink.npz') != report['raw_sha256'] or type(report['stopped']) is not bool:
            raise ValueError('Original trajectory differs')
        target = reference(row['corpus_index'])
        if len(target) != len(refs[row['case'].removesuffix('_normal')]['answer']):
            raise ValueError('Keep the entire reference extent')
        # Caps retain their original count and zero-reward law. They need no OCR.
        emitted = [store(judge.packed_cell(v[0])) for v in ink] if report['stopped'] else []
        stopped_positions += len(emitted)
        cases.append(dict(**row, stopped=report['stopped'], emitted_glyphs=len(ink),
                          emitted_cells=emitted, reference_cells=[store(v) for v in target]))
    keys = sorted(key for key, row in cells.items() if not row['blank'])
    inventory = dict(cases=cases, cells=cells, query_order=keys)
    write_new(RUN / 'inventory.json', inventory)
    guard(began, 300)
    write_new(REG, dict(schema='isolated-glyph-judge-registration-v1', input_sha256=bound,
        inventory_sha256=sha(RUN / 'inventory.json'), teacher=judge.fingerprint(),
        cases=len(cases), naturally_stopped_cases=sum(r['stopped'] for r in cases),
        naturally_stopped_positions=stopped_positions, nonblank_unique_queries=len(keys),
        prompt=judge.PROMPT, request_template=judge.payload(b''),
        resources=dict(cpu_threads=2, maximum_client_rss_gib=2, maximum_elapsed_seconds=1800,
                       first_request_timeout_seconds=180, later_request_timeout_seconds=60,
                       entry_host_available_gib=48, runtime_host_available_gib=24,
                       entry_gpu0_free_gib=20, maximum_observed_gpu0_used_gib=22,
                       maximum_foreign_gpu_increase_mib=512, minimum_disk_free_gib=48),
        checks=['all clean reference cells correctly transcribed', 'all 9 correct answers content accepted',
                'no readable-wrong or uncertain complete content accepted', 'correct/wrong AUC >= .95',
                'min correct reward > max uncredited', 'every capped reward zero'],
        action_on_failure='Reject this fixed judge; preserve responses and original readings. No policy fit.',
        action_on_pass='Limited reused-output calibration only; separately register training-only actor preflight.',
        preparation_seconds=time.monotonic() - began, policy_fit_authorized=False,
        reused_development=True, independent_human_adjudication=False))
    print(json.dumps(dict(registered=relative(REG), cases=len(cases), queries=len(keys),
                          stopped_positions=stopped_positions)), flush=True)


def authenticate(reg):
    for name, digest in reg['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Registered source changed: ' + name)
    if sha(RUN / 'inventory.json') != reg['inventory_sha256']:
        raise ValueError('Prepared physical inventory changed')
    inventory = load(RUN / 'inventory.json')
    for key, row in inventory['cells'].items():
        packed = (RUN / 'cells' / (key + '.bin')).read_bytes()
        if judge.cell_key(packed) != key or (not any(packed)) != row['blank']:
            raise ValueError('Physical cell identity changed')
        if sha(RUN / 'cells' / (key + '.png')) != row['png_sha256']:
            raise ValueError('Teacher image changed')
    return inventory


def git_admission():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'],
                                     cwd=ROOT, text=True, timeout=30).split()[0]
    committed = subprocess.check_output(['git', 'show', 'HEAD:' + relative(REG)], cwd=ROOT)
    if remote != head or hashlib.sha256(committed).hexdigest() != sha(REG):
        raise ValueError('Commit and push the fixed registration before neural calls')
    return head


def transcribe():
    reg = load(REG); inventory = authenticate(reg); commit = git_admission()
    if judge.fingerprint() != reg['teacher']:
        raise ValueError('Registered teacher identity changed')
    began = time.monotonic(); guard(began, 1800, reserve=48)
    initial_gpu = gpu_state()
    if initial_gpu[0]['free_mib'] < 20 * 1024:
        raise RuntimeError('GPU0 entry reserve not met')
    if judge.local_json('/api/ps')['models']:
        raise RuntimeError('Shared teacher service has a resident owner; do not replace it')
    (RUN / 'responses').mkdir()  # Existing partial work is terminal; no automatic replay.
    pid = os.getpid()
    owner = dict(pid=pid, start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 run=relative(RUN), registration_sha256=sha(REG), code_commit=commit,
                 initial_gpu=initial_gpu, started_unix=time.time(), no_automatic_retry=True)
    write_new(RUN / 'owner.json', owner)
    records = {}; minimum_host = available_gib(); peak_gpu = initial_gpu[0]['used_mib']
    def check_resources():
        nonlocal minimum_host, peak_gpu
        guard(began, 1800)
        minimum_host = min(minimum_host, available_gib())
        current = gpu_state(); peak_gpu = max(peak_gpu, current[0]['used_mib'])
        if current[0]['used_mib'] > 22 * 1024:
            raise RuntimeError('Observed physical GPU0 ceiling exceeded')
        if any(current[i]['used_mib'] > row['used_mib'] + 512 for i, row in initial_gpu.items() if i != 0):
            raise RuntimeError('Foreign GPU use increased; placement cannot be isolated')
        return current
    try:
        for number, key in enumerate(inventory['query_order'], 1):
            check_resources()
            image = (RUN / 'cells' / (key + '.png')).read_bytes()
            request = judge.payload(image)
            # Only a constant instruction and this physical cell cross the API boundary.
            started = time.monotonic()
            response = judge.local_json('/api/chat', request,
                timeout=min(180 if number == 1 else 60, max(.1, 1800 - (started - began))))
            record = dict(cell=key, request_sha256=hashlib.sha256(
                json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                response=response, reading=judge.parse_response(response),
                wall_seconds=time.monotonic() - started, gpu=gpu_state())
            path = RUN / 'responses' / (key + '.json'); write_new(path, record)
            records[key] = sha(path)
            check_resources()
            print(json.dumps(dict(phase='isolated_cell', completed=number,
                                  total=len(inventory['query_order']), accepted=record['reading']['accepted'],
                                  elapsed_seconds=time.monotonic() - began)), flush=True)
        if judge.fingerprint() != reg['teacher']:
            raise ValueError('Teacher identity changed during transcription')
        authenticate(reg)
        write_new(RUN / 'transcription_freeze.json', dict(registration_sha256=sha(REG),
            inventory_sha256=reg['inventory_sha256'], response_sha256=records,
            model_calls=len(records), elapsed_seconds=time.monotonic() - began,
            minimum_host_available_gib=minimum_host, peak_sampled_gpu0_used_gib=peak_gpu / 1024,
            sampling_scope='Physical device occupancy before and after each request; not allocator peak.',
            final_service_models=judge.local_json('/api/ps'),
            client_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            optimizer_updates=0, teacher_in_deployed_generation=False))
    except BaseException as error:
        write_new(RUN / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            completed_responses=len(records), elapsed_seconds=time.monotonic() - began,
            partial_responses_preserved=True, automatic_retry=False, training_authorized=False))
        raise


def score():
    reg = load(REG); inventory = authenticate(reg)
    freeze = load(RUN / 'transcription_freeze.json')
    if freeze['registration_sha256'] != sha(REG) or set(freeze['response_sha256']) != set(inventory['query_order']):
        raise ValueError('Require the complete first transcription freeze')
    readings = {}
    for key, digest in freeze['response_sha256'].items():
        path = RUN / 'responses' / (key + '.json')
        if sha(path) != digest:
            raise ValueError('Frozen teacher response changed')
        row = load(path)
        if row['cell'] != key or row['reading'] != judge.parse_response(row['response']):
            raise ValueError('Frozen transcription differs from strict response parser')
        readings[key] = row['reading']
    for key, row in inventory['cells'].items():
        if row['blank']:
            readings[key] = dict(text=' ', accepted=True, uncertain=False, reason='physically_white')
    # Reference strings and old human/model labels enter only after the cache freeze.
    refs = {r['id']: r for r in load(REFS)['reference_rows'] if r['kind'] == 'qa'}
    truth = {layout: {r['case']: r for r in load(OLD / 'review' / (layout + '_core_scores.json'))['records']
                     if r['step'] == 41298 and r['kind'] == 'qa' and r['condition'] == 'normal'}
             for layout in ('fixed_cells', 'proportional')}
    results = []
    for case in inventory['cases']:
        old = truth[case['layout']][case['case']]
        if case['stopped'] != old['stopped'] or case['emitted_glyphs'] != old['emitted_glyphs']:
            raise ValueError('Frozen stopping or extent differs')
        text = refs[case['case'].removesuffix('_normal')]['answer']
        flags = [inventory['cells'][k]['blank'] for k in case['reference_cells']]
        target = [' ' if blank else c for c, blank in zip(text, flags)]
        covered = all(readings[k]['accepted'] and readings[k]['text'] == c
                      for k, c in zip(case['reference_cells'], target))
        result = judge.score_sequence(case['emitted_cells'], readings, text, flags, stopped=case['stopped'])
        results.append(dict(case=case['case'], layout=case['layout'], **result,
            complete_exact=old['complete_exact'], readable=old['readable'], uncertain=old['reading']['uncertain'],
            stopped=case['stopped'], emitted_glyphs=case['emitted_glyphs'], reference_covered=covered))
    correct = [r for r in results if r['complete_exact']]
    wrong = [r for r in results if r['readable'] and not r['complete_exact']]
    uncertain = [r for r in results if not r['readable']]
    if (len(correct), len(wrong), len(uncertain)) != (9, 9, 46):
        raise ValueError('Keep all original review outcomes')
    auc = sum(float(a['reward'] > b['reward']) + .5 * float(a['reward'] == b['reward'])
              for a in correct for b in wrong) / 81
    checks = dict(all_clean_references_covered=all(r['reference_covered'] for r in results),
        all_nine_correct_content_accepted=all(r['complete_content'] for r in correct),
        no_readable_wrong_content_accepted=all(not r['complete_content'] for r in wrong),
        no_uncertain_content_accepted=all(not r['complete_content'] for r in uncertain),
        correct_wrong_auc_at_least_095=auc >= .95,
        all_correct_above_every_uncredited=min(r['reward'] for r in correct) > max(r['reward'] for r in wrong + uncertain),
        capped_reward_zero=all(r['reward'] == 0 for r in results if not r['stopped']))
    write_new(OUT, dict(schema='isolated-glyph-judge-calibration-v1', registration_sha256=sha(REG),
        transcription_freeze_sha256=sha(RUN / 'transcription_freeze.json'), rows=results, checks=checks,
        calibration_passed=all(checks.values()), correct_wrong_auc=auc,
        accepted_correct=sum(r['complete_content'] for r in correct),
        accepted_readable_wrong=sum(r['complete_content'] for r in wrong),
        accepted_uncertain=sum(r['complete_content'] for r in uncertain),
        resource_evidence=freeze, original_readings_unchanged=True, reused_development=True,
        independent_human_adjudication=False, policy_fit_authorized=False, goal_achieved=False))
    print(json.dumps(dict(checks=checks, calibration_passed=all(checks.values()), correct_wrong_auc=auc,
        accepted_correct=sum(r['complete_content'] for r in correct),
        accepted_readable_wrong=sum(r['complete_content'] for r in wrong),
        accepted_uncertain=sum(r['complete_content'] for r in uncertain))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'transcribe', 'score'))
    args = parser.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '2' or os.environ.get('OMP_NUM_THREADS') != '2':
        raise ValueError('Require the registered two-thread client')
    if args.action == 'register':
        register()
    elif args.action == 'score':
        score()
    else:
        with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            transcribe()
