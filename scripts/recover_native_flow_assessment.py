#!/usr/bin/env python3
"""Correct one recorded CPU name collision; retain all original model draws."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT/'artifacts/native_visual_flow_learning_v1'
ASSESSMENT = ROOT/'artifacts/native_visual_flow_assessment_v1'
FIELD = ROOT/'artifacts/native_flow_condition_probe_v1'
POSITIVE = ROOT/'artifacts/native_flow_positive_sampling_probe_v1'
ASSESSOR = ROOT/'scripts/assess_native_visual_flow.py'
ASSESSOR_SHA = '85c91dc887939e6ef6eb59aa880c8b0e47c274d5830912f67f1fd92540de6870'
MODEL_PYTHON = ROOT.parent/'LocalLLM/.local/image-generation/venv/bin/python'
SOURCE_GROUPS = ('training_code_sha256', 'assessment_code_sha256',
                 'condition_probe_code_sha256', 'code_sha256')
OWNED_SCRIPTS = {
    'scripts/train_native_visual_flow.py', 'scripts/infer_native_visual_flow.py',
    'scripts/run_native_visual_flow_assessment.py',
    'scripts/run_native_flow_condition_probe.py',
    'scripts/run_native_flow_positive_sampling_probe.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def corrected_source():
    require(sha(ASSESSOR) == ASSESSOR_SHA, 'only the recorded assessor revision is supported')
    source = ASSESSOR.read_text()
    replacements = (
        ('import identity,validate_checkpoint,schedule,digest\n',
         'import identity,validate_checkpoint,schedule,digest as schedule_digest\n'),
        ('or digest(plan)!=', 'or schedule_digest(plan)!='),
    )
    corrected = source
    for before, after in replacements:
        require(corrected.count(before) == 1, 'exact name-collision correction no longer applies')
        corrected = corrected.replace(before, after, 1)
    # Verify semantic scope independently of formatting: only one import alias
    # and one call name may differ from the exact recorded original AST.
    class Alias(ast.NodeTransformer):
        imports = 0
        calls = 0

        def visit_ImportFrom(self, node):
            if node.module == 'ilm.image_edit.native_flow_run':
                for name in node.names:
                    if name.name == 'digest':
                        require(name.asname is None, 'unexpected existing digest alias')
                        name.asname = 'schedule_digest'
                        self.imports += 1
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'digest':
                node.func.id = 'schedule_digest'
                self.calls += 1
            return node

    alias = Alias()
    expected = alias.visit(ast.parse(source))
    require((alias.imports, alias.calls) == (1, 1), 'unexpected original digest use')
    require(ast.dump(expected) == ast.dump(ast.parse(corrected)),
            'correction changes more than the imported function alias and call')
    compile(corrected, '<corrected_offline_assessor>', 'exec')
    return corrected


def require_no_owned_jobs():
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path/'cmdline').read_bytes().split(b'\0')
            argv = [item.decode(errors='replace') for item in argv if item]
            cwd = (path/'cwd').resolve()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if cwd == ROOT and any(item in OWNED_SCRIPTS for item in argv):
            raise ValueError('project model/coordinator still live: '+path.name)


def stage_bindings():
    stages = load(ASSESSMENT/'completed_stages.json')
    expected = {'export'} | {prefix+arm+'_'+variant
        for prefix in ('infer_', 'score_') for arm in ('initial', 'final')
        for variant in ('normal', 'question_erased', 'passage_erased', 'copy')}
    require(set(stages) == expected, 'exactly17 completed precursor stages required')
    for entry in stages.values():
        require(sha(entry['primary']) == entry['primary_sha256'], 'completed primary changed')
    return stages


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for preparation')
    require_no_owned_jobs()
    corrected = corrected_source()
    stages = stage_bindings()
    for folder in (ASSESSMENT, FIELD, POSITIVE):
        require(load(folder/'runtime.json')['phase'] == 'failed', 'terminal failed prerequisite expected')
    error = ASSESSMENT/'assessment.attempt_001.log'
    require("TypeError: 'str' object is not callable" in error.read_text()
            and 'digest(plan)' in error.read_text(), 'the recorded aggregation failure differs')
    require(not (ASSESSMENT/'assessment.json').exists(), 'preserve any existing complete assessment')
    for folder in (FIELD, POSITIVE):
        for name in ('final', 'initial', 'calibration', 'final_ocr', 'initial_ocr', 'result.json'):
            require(not (folder/name).exists(), 'a diagnostic model/scoring attempt already exists')
    positive = load(POSITIVE/'preparation.json')
    sources = {}
    for group in SOURCE_GROUPS:
        for name, expected in positive[group].items():
            require(name not in sources or sources[name] == expected, 'inconsistent source binding')
            require(sha(ROOT/name) == expected, 'registered source changed: '+name)
            sources[name] = expected
    require(len(sources) == 19, 'all19 registered sources required')
    bound = dict(sources)
    for folder in (TRAINING, ASSESSMENT, FIELD, POSITIVE):
        path = folder/'preparation.json'
        bound[str(path.relative_to(ROOT))] = sha(path)
    for path in (FIELD/'cases.json', POSITIVE/'inputs.json', POSITIVE/'references.jsonl'):
        bound[str(path.relative_to(ROOT))] = sha(path)
    for variant, count in (('normal',64), ('question_erased',32), ('passage_erased',32), ('copy',64)):
        for suffix in ('_inputs.json', '_references.jsonl'):
            path = TRAINING/'evaluation'/(variant+suffix)
            bound[str(path.relative_to(ROOT))] = sha(path)
        for arm in ('initial', 'final'):
            folder = ASSESSMENT/(arm+'_'+variant+'_inference')
            require(len(load(folder/'outputs.json')) == count
                    and load(folder/'report.json')['complete_generations'] == count,
                    'all384 original image-index entries required')
            for path in (folder/'outputs.json', folder/'report.json',
                         ASSESSMENT/(arm+'_'+variant+'_ocr/report.json')):
                bound[str(path.relative_to(ROOT))] = sha(path)
    require(not subprocess.check_output(['git','status','--porcelain','--',
        str(Path(__file__).relative_to(ROOT)), 'references/native_flow_assessment_recovery.md'],
        cwd=ROOT, text=True), 'commit the recovery implementation and scope first')
    output.mkdir(parents=True, exist_ok=False)
    (output/'corrected_assessor.py').write_text(corrected)
    for label, folder in (('assessment',ASSESSMENT), ('field',FIELD), ('positive',POSITIVE)):
        (output/(label+'_original_failed_runtime.json')).write_bytes((folder/'runtime.json').read_bytes())
    save(output/'original_completed_stages.json', stages)
    save(output/'preparation.json', dict(implementation_sha256=sha(__file__),
        scope_sha256=sha(ROOT/'references/native_flow_assessment_recovery.md'),
        original_assessor_sha256=ASSESSOR_SHA, corrected_assessor_sha256=sha(output/'corrected_assessor.py'),
        correction='alias imported digest as schedule_digest and call that alias; no other AST changes',
        bound_files_sha256=bound, original_failure_log_sha256=sha(error),
        original_completed_stages_sha256=sha(output/'original_completed_stages.json'),
        original_failed_runtime_sha256={label:sha(output/(label+'_original_failed_runtime.json'))
            for label in ('assessment','field','positive')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        regenerated_assessment_images=0, optimizer_updates=0, model_loaded=False,
        scheduled_field_forwards=144, scheduled_diagnostic_images=10,
        general_ilm_goal_complete=False))
    print('Prepared exact offline name correction; all19 sources and384 draws remain unchanged.')


def unchanged(output):
    meta = load(output/'preparation.json')
    require(sha(__file__) == meta['implementation_sha256'], 'recovery implementation changed')
    require(sha(ROOT/'references/native_flow_assessment_recovery.md') == meta['scope_sha256'], 'scope changed')
    require(sha(output/'corrected_assessor.py') == meta['corrected_assessor_sha256'], 'corrected assessor changed')
    require((output/'corrected_assessor.py').read_text() == corrected_source(), 'correction no longer reproduces')
    require(sha(ASSESSMENT/'assessment.attempt_001.log') == meta['original_failure_log_sha256'], 'failure log changed')
    for name, expected in meta['bound_files_sha256'].items():
        require(sha(ROOT/name) == expected, 'bound source, data or report changed: '+name)
    require(sha(output/'original_completed_stages.json') == meta['original_completed_stages_sha256'],
            'original stage ledger changed')
    for label, expected in meta['original_failed_runtime_sha256'].items():
        require(sha(output/(label+'_original_failed_runtime.json')) == expected, 'archived failed runtime changed')
    return meta


def assess(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for offline assessment')
    meta = unchanged(output)
    require(not (ASSESSMENT/'assessment.json').exists(), 'preserve an existing assessment result')
    namespace = runpy.run_path(str(output/'corrected_assessor.py'), run_name='explicit_repaired_offline_assessor')
    namespace['assess'](TRAINING, ASSESSMENT)
    result = load(ASSESSMENT/'assessment.json')
    require(result['actual_new_generations'] == 384
            and result['all_PNGs_authenticated_against_actual_decoder_tensors'], 'full assessment did not authenticate')
    result['offline_assessor_recovery'] = dict(preparation_sha256=sha(output/'preparation.json'),
        implementation_sha256=meta['implementation_sha256'], original_assessor_sha256=ASSESSOR_SHA,
        corrected_assessor_sha256=meta['corrected_assessor_sha256'],
        correction=meta['correction'], regenerated_images=0, model_or_metric_changed=False)
    save(ASSESSMENT/'assessment.json', result)
    unchanged(output)


def run(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for coordinator')
    require_no_owned_jobs()
    unchanged(output)
    lock = (output/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (output/'runtime.json').exists(), 'preserve existing recovery attempt; no automatic retry')
    child = None
    phase = 'offline_assessment'
    started = time.time()
    active = None

    def status(**extra):
        record = dict(phase=phase, pid=os.getpid() if phase not in ('failed','completed') else None,
            child_pid=child.pid if child is not None and child.poll() is None else None,
            started_unix=started, gpu=None, novnc_url=None, general_ilm_goal_complete=False, **extra)
        save(output/'runtime.json', record)
        if active is not None:
            save(active/'runtime.json', dict(record, recovery_preparation_sha256=sha(output/'preparation.json')))

    def stop(signum, frame):
        raise InterruptedError('recovery signal '+str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    def call(label, script, arguments, gpu=False):
        nonlocal child, phase
        unchanged(output)
        phase = label
        command = [str(Path.home()/'.local/bin/with-current-nvidia'), 'env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES='+('0' if gpu else ''), 'PYTHONNOUSERSITE=1', 'OMP_NUM_THREADS=4',
            'OPENBLAS_NUM_THREADS=2', 'MKL_NUM_THREADS=4', 'PYTHONPATH=.', str(MODEL_PYTHON), script, *map(str,arguments)]
        with (output/(label+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(child_command=command, child_gpu=0 if gpu else None)
            try:
                require(child.wait() == 0, 'recovery child failed: '+label+'; preserve attempt')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        unchanged(output)
        return command

    try:
        status()
        command = call('offline_assessment', 'scripts/recover_native_flow_assessment.py',
                       ['--stage','assess','--output',output])
        original = load(output/'original_completed_stages.json')
        require(load(ASSESSMENT/'completed_stages.json') == original, 'precursor stage ledger changed')
        completed = dict(original)
        completed['assessment'] = dict(primary=str(ASSESSMENT/'assessment.json'),
            primary_sha256=sha(ASSESSMENT/'assessment.json'), command=command,
            explicit_offline_correction_preparation_sha256=sha(output/'preparation.json'))
        save(ASSESSMENT/'completed_stages.json', completed)
        call('resume_assessment', 'scripts/run_native_visual_flow_assessment.py',
             ['--stage','run','--training-run',TRAINING,'--output',ASSESSMENT,'--resume'])
        require(load(ASSESSMENT/'runtime.json')['phase'] == 'ready_for_raw_review', 'assessment is not ready')
        require(load(ASSESSMENT/'completed_stages.json') == completed, 'resume changed completed stages')
        # No diagnostic arm existed before the preserved prerequisite failures.
        # Use the same registered arm entry points and data, without a reroll.
        active = FIELD
        for arm in ('final','initial'):
            call('field_'+arm, 'scripts/run_native_flow_condition_probe.py',
                 ['--stage','arm','--output',FIELD,'--arm',arm], gpu=True)
        call('field_collect', 'scripts/run_native_flow_condition_probe.py', ['--stage','collect','--output',FIELD])
        require(load(FIELD/'result.json')['actual_denoiser_forwards'] == 144, 'field collection incomplete')
        save(FIELD/'runtime.json', dict(phase='completed', pid=None, child_pid=None,
            recovery_preparation_sha256=sha(output/'preparation.json'),
            gpu=None, novnc_url=None, general_ilm_goal_complete=False))
        active = POSITIVE
        for name in ('calibrate','final','score_final','initial','score_initial','collect'):
            if name in ('final','initial','collect'):
                call('positive_'+name, 'scripts/run_native_flow_positive_sampling_probe.py',
                    ['--stage','collect' if name=='collect' else 'arm','--output',POSITIVE]
                    + ([] if name=='collect' else ['--arm',name]), gpu=name!='collect')
            else:
                arguments = ['--stage','calibrate' if name=='calibrate' else 'score', '--task','qa','--train-limit','2',
                    '--manifest',POSITIVE/'references.jsonl','--output',
                    POSITIVE/('calibration' if name=='calibrate' else name[6:]+'_ocr')]
                if name != 'calibrate':
                    arguments += ['--calibration',POSITIVE/'calibration/report.json',
                                  '--generations',POSITIVE/name[6:]/'outputs.json']
                call('positive_'+name, 'scripts/score_paired_writing.py', arguments)
        phase = 'completed'
        status()
        save(output/'result.json', dict(preparation_sha256=sha(output/'preparation.json'),
            assessment_sha256=sha(ASSESSMENT/'assessment.json'),
            field_result_sha256=sha(FIELD/'result.json'), positive_result_sha256=sha(POSITIVE/'result.json'),
            elapsed_seconds=time.time()-started, regenerated_assessment_images=0, optimizer_updates=0,
            original19_sources_unchanged=True, general_ilm_goal_complete=False))
    except BaseException as error:
        phase = 'failed'
        status(reason=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare','run','assess'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'assess':assess}[args.stage](args.output.resolve())
