#!/usr/bin/env python3
"""Matched native guidance-one diagnostic after the complete field probe."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.image_edit.native_flow_run import CONDITION, identity, restore_adapters
from ilm.image_edit import visual_semantic as vs
from scripts.run_native_flow_condition_probe import owner

FILES = ('scripts/run_native_flow_positive_sampling_probe.py',
         'references/native_flow_positive_sampling_protocol.md')
POSITIONS = (0, 1, 32, 33)


def load(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def code_identity():
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--', *FILES], cwd=ROOT),
            'commit diagnostic code/protocol before preparation')
    return {name: sha(ROOT/name) for name in FILES}


def prepare(training, assessment, probe, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for CPU preparation')
    code = code_identity()
    meta = load(training/'preparation.json')
    require(identity() == meta['code_sha256'], 'registered training source changed')
    registered = load(assessment/'preparation.json')
    require(registered['training_preparation_sha256'] == sha(training/'preparation.json'), 'assessment owner differs')
    predecessor = load(probe/'preparation.json')
    deployment = training/'deployment/deployment.json'
    info = load(deployment)
    require(info['completed_updates'] == 1536 and predecessor['deployment_sha256'] == sha(deployment),
            'same final checkpoint as the registered field probe required')
    index = training/'evaluation/normal_inputs.json'
    refs_path = training/'evaluation/normal_references.jsonl'
    for path in (index, refs_path):
        require(sha(path) == registered['evaluation_files_sha256'][path.name], 'fixed normal data changed')
    inputs = load(index)
    refs = [json.loads(line) for line in refs_path.read_text().splitlines()]
    require(len(inputs) == len(refs) == 64, 'complete original normal index required')
    selected_inputs, selected_refs = [], []
    for position in POSITIONS:
        inp, ref = inputs[position], refs[position]
        require(inp['id'] == ref['id'] and set(inp) == {'id', 'input', 'input_sha256', 'seed'}, 'original input differs')
        given = dict(inp, input=os.path.relpath((index.parent/inp['input']).resolve(), output))
        reference = dict(ref)
        for key in ('input', 'target'):
            path = (refs_path.parent/ref[key]).resolve()
            require(sha(path) == ref[key+'_sha256'], 'original image changed')
            reference[key] = os.path.relpath(path, output)
        selected_inputs.append(given)
        selected_refs.append(reference)
    require([r['split'] for r in selected_refs] == ['train', 'train', 'validation', 'validation']
            and all(selected_refs[i]['pair'] == selected_refs[i+1]['pair'] for i in (0, 2)), 'complete fixed passage pairs required')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'inputs.json', selected_inputs)
    (output/'references.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in selected_refs))
    save_json(output/'preparation.json', dict(code_sha256=code, training_code_sha256=meta['code_sha256'],
        assessment_code_sha256=registered['assessment_code_sha256'],
        assessment=str(assessment), assessment_preparation_sha256=sha(assessment/'preparation.json'),
        condition_probe=str(probe), condition_probe_preparation_sha256=sha(probe/'preparation.json'),
        condition_probe_code_sha256=predecessor['code_sha256'],
        deployment=str(deployment), deployment_sha256=sha(deployment), weights_sha256=info['weights_sha256'],
        inputs_sha256=sha(output/'inputs.json'), references_sha256=sha(output/'references.jsonl'),
        source_QA_sha256=registered['source_QA_sha256'], positions=list(POSITIONS),
        arm_order=['final', 'initial'], guidance=1., steps=50, canvas=[1024, 1024],
        scheduled_autonomous_images=10, scheduled_denoiser_calls=500, scheduled_denoiser_batch_elements=600,
        reused_known_failing_diagnostic_cases=True, model_loaded=False, optimizer_updates=0,
        general_ilm_goal_complete=False))
    print('Prepared4 fixed input-only cases;8 future guidance-one answers and2 execution replays. No model loaded.')


def unchanged(output, *, with_references=True):
    meta = load(output/'preparation.json')
    require(code_identity() == meta['code_sha256'] and identity() == meta['training_code_sha256'], 'registered sources changed')
    files = [('inputs.json', 'inputs_sha256')]
    if with_references:
        files.append(('references.jsonl', 'references_sha256'))
    for name, key in files:
        require(sha(output/name) == meta[key], 'fixed diagnostic data changed')
    require(sha(meta['deployment']) == meta['deployment_sha256']
            and sha(Path(meta['deployment']).parent/'adapters.safetensors') == meta['weights_sha256'], 'final deployment changed')
    for path, value in meta['condition_probe_code_sha256'].items():
        require(sha(ROOT/path) == value, 'registered predecessor code changed')
    for path, value in meta['assessment_code_sha256'].items():
        require(sha(ROOT/path) == value, 'registered assessment code changed')
    require(sha(Path(meta['condition_probe'])/'preparation.json') == meta['condition_probe_preparation_sha256']
            and sha(Path(meta['assessment'])/'preparation.json') == meta['assessment_preparation_sha256'], 'predecessor preparation changed')
    return meta


def require_predecessors(meta):
    assessment, probe = Path(meta['assessment']), Path(meta['condition_probe'])
    require(load(assessment/'runtime.json')['phase'] == 'ready_for_raw_review'
            and load(assessment/'assessment.json')['actual_new_generations'] == 384,
            'complete384-image assessment must finish first')
    require(load(probe/'runtime.json')['phase'] == 'completed'
            and load(probe/'result.json')['actual_denoiser_forwards'] == 144,
            'complete144-forward condition probe must finish first')


def arm(output, name):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'use GPU0 only')
    meta = unchanged(output, with_references=False)
    require_predecessors(meta)
    inputs = load(output/'inputs.json')
    assessment = Path(meta['assessment'])
    original = assessment/(name+'_normal_inference')
    prior = load(original/'outputs.json')
    original_report = load(original/'report.json')
    require(len(prior) == original_report['complete_generations'] == 64, 'complete original counterpart required')
    selected = [prior[i] for i in POSITIONS]
    require(all(all(a[key] == b[key] for key in ('id', 'input_sha256', 'seed'))
            for a, b in zip(inputs, selected)), 'original matched cases differ')
    folder = output/name
    folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    with gpu_zero() as device:
        import torch
        from PIL import Image
        from safetensors.torch import load_file
        from ilm.image_edit.native_flow_step import add_adapters
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source = vs.verify_release()
        info = load(meta['deployment'])
        require(source == info['source_sha256'], 'fixed complete source changed')
        pipe, loader = vs.load_pipeline(device)
        vs.install_compiled(pipe, CONDITION)
        removal = vs.remove_lexical_modules(pipe)
        require(add_adapters(pipe, info['rank']) == info['adapter_layer_names'], 'adapter graph differs')
        if name == 'final':
            restore_adapters(pipe, load_file(str(Path(meta['deployment']).parent/'adapters.safetensors'), device='cpu'))
        for component in ('transformer', 'connector_module'):
            getattr(pipe, component).eval()
        batches = []
        def before_forward(module, args, kwargs):
            batches.append(int(kwargs['hidden_states'].shape[0]))
        hook = pipe.transformer.register_forward_pre_hook(before_forward, with_kwargs=True)
        def generate(row, guidance, destination):
            unchanged(output, with_references=False)
            require(sha(output/row['input']) == row['input_sha256'], 'input image changed')
            batches.clear()
            value = vs.generate(pipe, output/row['input'], seed=row['seed'], width=1024, height=1024,
                                steps=50, guidance=guidance)
            saved = vs.save_generation(value, destination)
            require(batches == [2 if guidance == 4. else 1]*50, 'native denoiser call count or branch batch differs')
            return value, saved, list(batches)
        replay, saved_replay, replay_batches = generate(inputs[0], 4., folder/'guidance4_replay')
        baseline = torch.load(original/selected[0]['path'].replace('answer.png', 'actual_tensors.pt'),
                              map_location='cpu', weights_only=True)
        require(sha(original/selected[0]['path']) == selected[0]['sha256'], 'original replay PNG changed')
        with Image.open(original/selected[0]['path']) as image:
            baseline['image'] = image.copy()
        equality = vs.exact_replay(baseline, replay)
        save_json(folder/'guidance4_replay.json', dict(original_index_sha256=sha(original/'outputs.json'),
            original_PNG_sha256=selected[0]['sha256'], result=saved_replay, equality=equality,
            denoiser_batch_sizes=replay_batches))
        require(equality['decoded_tensor_exact'] and equality['png_pixels_exact']
                and all(equality['forward_fields_exact'].values()), 'native guidance-four replay failed; preserve attempt')
        del baseline, replay
        outputs = []
        for i, (given, old) in enumerate(zip(inputs, selected)):
            value, saved, calls = generate(given, 1., folder/f'{i:03}')
            path = original/old['path'].replace('answer.png', 'actual_tensors.pt')
            require(sha(path) == old['result']['tensor_sha256'], 'original counterpart tensors changed')
            baseline = torch.load(path, map_location='cpu', weights_only=True)['forward_inputs']
            fields = value['forward_inputs']
            require(set(fields) == set(baseline) and all((fields[k] is None and baseline[k] is None)
                    if fields[k] is None else torch.equal(fields[k], baseline[k]) for k in fields),
                    'guidance-one input conditioning differs from original case')
            outputs.append(dict(id=given['id'], input_sha256=given['input_sha256'], seed=given['seed'],
                path=f'{i:03}/answer.png', sha256=saved['sha256'], result=saved,
                original_tensor_sha256=old['result']['tensor_sha256'], reader_inputs_bit_exact_to_guidance4=True,
                denoiser_batch_sizes=calls))
            save_json(folder/'outputs.json', outputs)
            save_json(folder/'status.json', dict(phase='generating', pid=os.getpid(), completed=i+1, total=4))
            del baseline, value
            print(f'Generated guidance-one case{i+1}/4', flush=True)
        hook.remove()
        unchanged(output, with_references=False)
        save_json(folder/'report.json', dict(arm=name, preparation_sha256=sha(output/'preparation.json'),
            outputs_sha256=sha(folder/'outputs.json'), learned_adapter_weights_loaded=name == 'final',
            deployment_sha256=meta['deployment_sha256'], source=source, loader=loader, removal=removal,
            runtime_parameters=vs.parameters(pipe), guidance1_answers=4, guidance4_replays=1,
            actual_autonomous_images=5, actual_denoiser_calls=250, actual_denoiser_batch_elements=300,
            elapsed_seconds=time.perf_counter()-started, peak_allocated_vram_GiB=torch.cuda.max_memory_allocated()/1024**3,
            target_or_reference_manifest_loaded=False, external_teacher_or_OCR=False, changing_text_condition=False,
            optimizer_updates=0, general_ilm_goal_complete=False))
    save_json(folder/'status.json', dict(phase='completed', pid=None))


def collect(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for offline collection')
    from ilm.image_edit.native_flow_assessment import authenticate_native_png, summarize
    from scripts.evaluate_image_edit_baseline import normalize
    import torch
    torch.set_num_threads(4)
    meta = unchanged(output)
    require_predecessors(meta)
    refs = [json.loads(x) for x in (output/'references.jsonl').read_text().splitlines()]
    given = load(output/'inputs.json')
    source_path = ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    require(sha(source_path) == meta['source_QA_sha256'], 'source-accepted answers changed')
    source = {r['id']: r for r in map(json.loads, source_path.read_text().splitlines())}
    calibration = load(output/'calibration/report.json')
    require(calibration['complete_cohort'] and calibration['manifest_sha256'] == meta['references_sha256'], 'owning calibration differs')
    original = load(Path(meta['assessment'])/'assessment.json')
    result, bindings = {}, {}
    for name in meta['arm_order']:
        folder = output/name
        report, rows, scored = load(folder/'report.json'), load(folder/'outputs.json'), load(output/(name+'_ocr/report.json'))
        require(report['preparation_sha256'] == sha(output/'preparation.json') and report['actual_autonomous_images'] == 5
                and report['actual_denoiser_calls'] == 250 and report['actual_denoiser_batch_elements'] == 300
                and report['outputs_sha256'] == sha(folder/'outputs.json')
                and report['learned_adapter_weights_loaded'] == (name == 'final'), 'complete same-arm inference required')
        require(scored['complete_cohort'] and scored['reference_calibration_sha256'] == sha(output/'calibration/report.json')
                and scored['manifest_sha256'] == meta['references_sha256']
                and scored['engine'] == calibration['engine'] and scored['implementation'] == calibration['implementation'],
                'owning unchanged OCR calibration required')
        require([r['id'] for r in rows] == [r['id'] for r in refs] == [r['id'] for r in scored['rows']], 'complete original case order required')
        observed = []
        for out, score, ref, inp in zip(rows, scored['rows'], refs, given):
            require(out['sha256'] == score['png_sha256'] and score['expected'] == ref['reference_text']
                    and out['input_sha256'] == inp['input_sha256'] == ref['input_sha256']
                    and out['seed'] == inp['seed'] and sha(output/inp['input']) == inp['input_sha256']
                    and sha(output/ref['target']) == ref['target_sha256'], 'observed output differs')
            signature = authenticate_native_png(folder/out['path'], out['sha256'], out['result']['tensor_sha256'])
            text = normalize(score['raw_transcript'], ref['language'])
            canonical = text == normalize(ref['reference_text'], ref['language'])
            require(canonical == score['ocr_exact'] and text == score['normalized'], 'canonical score differs')
            accepted = [a['text'] for a in source[ref['id']]['references']]
            observed.append(dict(id=ref['id'], split=ref['split'], pair=ref['pair'], raw_transcript=score['raw_transcript'],
                canonical_exact=canonical, accepted_exact=text in {normalize(a, ref['language']) for a in accepted},
                actual_pixel_signature=signature, PNG_sha256=out['sha256']))
        replay = load(folder/'guidance4_replay.json')
        authenticate_native_png(folder/'guidance4_replay/answer.png', replay['result']['sha256'], replay['result']['tensor_sha256'])
        require(replay['equality']['decoded_tensor_exact'] and replay['equality']['png_pixels_exact']
                and all(replay['equality']['forward_fields_exact'].values()), 'original native replay differs')
        selected = [r for r in original['rows'] if r['arm'] == name and r['variant'] == 'normal' and r['id'] in {v['id'] for v in refs}]
        require(len(selected) == 4, 'same four original guidance-four observations required')
        result[name] = dict(guidance1=summarize(observed, paired=True), guidance4=summarize(selected, paired=True), rows=observed)
        bindings[name] = dict(report_sha256=sha(folder/'report.json'), outputs_sha256=sha(folder/'outputs.json'),
            OCR_report_sha256=sha(output/(name+'_ocr/report.json')), replay_sha256=sha(folder/'guidance4_replay.json'))
    require(not torch.cuda.is_initialized(), 'offline collection initialized CUDA')
    save_json(output/'result.json', dict(preparation_sha256=sha(output/'preparation.json'), results=result, bindings=bindings,
        calibration_sha256=sha(output/'calibration/report.json'),
        original_assessment_sha256=sha(Path(meta['assessment'])/'assessment.json'),
        condition_probe_sha256=sha(Path(meta['condition_probe'])/'result.json'),
        actual_autonomous_generations=10, guidance1_generations=8, guidance4_replay_generations=2,
        actual_denoiser_calls=500, actual_denoiser_batch_elements=600,
        all_original_guidance4_replays_bit_exact=True, all_answer_PNGs_authenticated=True,
        optimizer_updates=0, raw_review_completed=False, small_posthoc_diagnostic_only=True,
        semantic_learning_validated=False, general_ilm_goal_complete=False))
    print(json.dumps(result, indent=2))


def run(output, wait_pid):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for coordinator')
    meta = unchanged(output)
    waiting = owner(wait_pid)
    require(waiting and 'scripts/run_native_flow_condition_probe.py' in waiting['command']
            and meta['condition_probe'] in waiting['command'], 'wait for the exact live field-probe coordinator')
    lock = (output/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (output/'runtime.json').exists(), 'preserve existing diagnostic attempt; no automatic rerun')
    phase, child = 'waiting_condition_probe', None
    def status(**extra):
        save_json(output/'runtime.json', dict(phase=phase, pid=os.getpid() if phase not in ('failed', 'completed') else None,
            child_pid=child.pid if child is not None and child.poll() is None else None, wait_owner=waiting,
            gpu=None, novnc_url=None, general_ilm_goal_complete=False, **extra))
    def stop(signum, frame):
        raise InterruptedError('positive sampling coordinator signal'+str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        status()
        while owner(wait_pid) == waiting:
            unchanged(output)
            time.sleep(30)
        require(owner(wait_pid) is None, 'field-probe PID was reused')
        require_predecessors(meta)
        for phase in ('calibrate', 'final', 'score_final', 'initial', 'score_initial', 'collect'):
            unchanged(output)
            gpu = phase in ('final', 'initial')
            command = [str(Path.home()/'.local/bin/with-current-nvidia'), 'env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
                'CUDA_VISIBLE_DEVICES='+('0' if gpu else ''), 'PYTHONNOUSERSITE=1', 'OMP_NUM_THREADS=4',
                'OPENBLAS_NUM_THREADS=2', 'MKL_NUM_THREADS=4', 'PYTHONPATH=.',
                str(ROOT.parent/'LocalLLM/.local/image-generation/venv/bin/python')]
            if gpu or phase == 'collect':
                command += ['scripts/run_native_flow_positive_sampling_probe.py', '--stage', 'arm' if gpu else 'collect', '--output', str(output)]
                if gpu:
                    command += ['--arm', phase]
            else:
                command += ['scripts/score_paired_writing.py', '--stage', 'calibrate' if phase == 'calibrate' else 'score',
                    '--task', 'qa', '--train-limit', '2', '--manifest', str(output/'references.jsonl'),
                    '--output', str(output/('calibration' if phase == 'calibrate' else phase[6:]+'_ocr'))]
                if phase != 'calibrate':
                    command += ['--calibration', str(output/'calibration/report.json'), '--generations', str(output/phase[6:]/'outputs.json')]
            with (output/(phase+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status()
                try:
                    require(child.wait() == 0, 'positive sampling stage failed: '+phase)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
        phase = 'completed'
        status()
    except BaseException as error:
        phase = 'failed'
        status(reason=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run', 'arm', 'collect'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--assessment', type=Path)
    parser.add_argument('--condition-probe', type=Path)
    parser.add_argument('--wait-for-probe-pid', type=int)
    parser.add_argument('--arm', choices=('final', 'initial'))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == 'prepare':
        prepare(args.training_run.resolve(), args.assessment.resolve(), args.condition_probe.resolve(), output)
    elif args.stage == 'run':
        run(output, args.wait_for_probe_pid)
    elif args.stage == 'arm':
        arm(output, args.arm)
    else:
        collect(output)
