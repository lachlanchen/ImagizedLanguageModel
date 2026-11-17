#!/usr/bin/env python3
"""Register and measure competing gradients on the two retained final actors."""
import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/trainable_visual_context_comparison_v1'
REG = ROOT / 'references/evidence/visual_context_gradient_registration_20261117.json'
PROTOCOL = ROOT / 'references/visual_context_gradient_protocol_20261117.md'
RUN = ROOT / 'artifacts/visual_context_gradient_diagnostic_v1'
PREFLIGHT = ROOT / 'references/evidence/trainable_visual_context_preflight_registration_20261116.json'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Register with CUDA hidden')
    previous = load(PREFLIGHT)
    paths = {ROOT / name for name in previous['input_sha256']}
    paths.update((Path(__file__), PROTOCOL, PREFLIGHT,
        ROOT / 'ilm/visual_lm/gradient_signal_accounting.py',
        ROOT / 'tests/test_gradient_signal_accounting.py',
        ROOT / 'references/evidence/visual_context_gradient_cpu_check_20261117.json',
        ROOT / 'ilm/visual_lm/trainable_context_checkpoint.py',
        ROOT / 'ilm/visual_lm/visual_context_deployment_resources.py',
        ROOT / 'references/evidence/trainable_visual_context_result_20261117.json'))
    arms = {}
    for arm in ('trainable', 'frozen'):
        directory = OLD / arm / 'fit'
        report = load(directory / 'report.json')
        audit = load(directory / 'audit.json')
        require(audit['audit_passed'] and audit['latest_matches_full_deploy']
                and audit['updates'] == report['updates'] == 8172, 'Require audited final actors')
        deploy = directory / report['final_checkpoint']
        require(sha(deploy) == report['final_checkpoint_sha256'] == audit['final_checkpoint_sha256'],
                'Final deployment file changed')
        paths.update((directory / 'checkpoint_latest.pt', deploy,
                      directory / 'report.json', directory / 'audit.json'))
        arms[arm] = dict(latest=str((directory / 'checkpoint_latest.pt').relative_to(ROOT)),
                         deploy=str(deploy.relative_to(ROOT)),
                         final_checkpoint_sha256=report['final_checkpoint_sha256'])
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}
    for name, digest in previous['input_sha256'].items():
        require(bindings[name] == digest, 'Protected training source changed: ' + name)
    result = dict(schema='ilm-visual-context-gradient-diagnostic-v1',
        run=str(RUN.relative_to(ROOT)), cases=previous['cases'], arms=arms,
        input_sha256=bindings, packages=previous['packages'],
        codec='artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt',
        teacher='artifacts/visual_sequence_teacher_v1', seed=202611172,
        order=['trainable', 'frozen'], passes=['flow', 'auxiliary', 'combined'],
        resources=dict(maximum_seconds=1200, maximum_pressure_wait_seconds=300,
            minimum_entry_host_gib=48, minimum_runtime_host_gib=24,
            minimum_free_gpu_gib=20, maximum_allocated_gpu_gib=22,
            minimum_free_disk_gib=48, cpu_threads=2, chunk_elements=262144),
        complete_records_only=True, gradients_from_bf16_forward_and_fp32_weights=True,
        same_rng_reset_before_each_pass=True, optimizer_updates_permitted=False,
        generation_permitted=False, retained_model_permitted=False,
        support_rule='Treatment book and at least one of its two QA records: combined g_flow dot FP32 AdamW displacement >= 0, while flow-only displacement dot g_flow < -1e-12. Otherwise interference is not established by this diagnostic.',
        action_on_support='Only prepare a separately registered objective-change update/reset preflight; no full fit.',
        action_on_no_support='Do not assume auxiliary interference or tune weights. Record the measured limitation before choosing another bounded change.',
        action_on_execution_failure='Preserve partial evidence and stop; no automatic retry or record truncation.',
        language_quality_measured=False, goal_achieved=False)
    write_new(REG, result)
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=sha(REG),
                          bindings=len(bindings), complete_records_per_arm=len(result['cases']))))


def require_pushed():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'],
                                     cwd=ROOT, text=True, timeout=30).split()[0]
    require(head == remote, 'Push the registration and diagnostic sources before execution')
    for name in load(REG)['input_sha256']:
        path = ROOT / name
        if name.startswith(('scripts/', 'ilm/', 'tests/', 'references/')):
            require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT) == path.read_bytes(),
                    'Uncommitted diagnostic or evidence source: ' + name)
    require(subprocess.check_output(['git', 'show', 'HEAD:' + str(REG.relative_to(ROOT))],
                                   cwd=ROOT) == REG.read_bytes(), 'Registration not committed')
    return head


def parameters_and_states(model, objective, saved, train_context):
    groups = []
    image = []
    for prefix, module in (('reader', model.encoder), ('writer', model.writer), ('projection', objective)):
        image.extend((prefix + '.' + name, parameter) for name, parameter in module.named_parameters())
    groups.append(image)
    if train_context:
        groups.append([('context.' + name, parameter) for name, parameter in model.context.named_parameters()])
    stored = saved['optimizer']
    require(len(groups) == len(stored['param_groups']), 'Optimizer groups differ')
    rows = []
    for names, group in zip(groups, stored['param_groups']):
        require(len(names) == len(group['params']), 'Optimizer parameter order differs')
        for (name, parameter), identity in zip(names, group['params']):
            state = stored['state'][identity]
            require(state['step'] == 8172, 'Require final retained moments')
            require(all(state[key].dtype == parameter.dtype and state[key].numel() == parameter.numel()
                        for key in ('exp_avg', 'exp_avg_sq')), 'Retained moment shape differs')
            rows.append((name, parameter, state, group))
    require(len(rows) == (409 if train_context else 100), 'Trainable parameter set changed')
    return rows


def example_for(case, corpus, plan, cache, lookup, references, teacher, device):
    import numpy as np
    from ilm.visual_lm.compact_visual_span_training import VisualSpanTrainingExample
    from scripts.cache_visual_sequence_teacher_states import read_sequences
    from scripts.train_full_rank_visual_span import load_states
    records = read_sequences(corpus.directory, plan['shards'][case['shard']])
    states, offsets = load_states(teacher, cache[case['shard']], records)
    record = records[case['local']]
    row = corpus[case['corpus_index']]
    alignment = corpus.offline_teacher_alignment(case['corpus_index'])
    selected = lookup[alignment['unit_ids'].long()]
    require(not bool((selected < 0).any()) and np.array_equal(record.units, alignment['unit_ids'].numpy()),
            'Training alignment changed')
    require(len(row['ink']) == case['glyphs'] and len(record.units) == case['native_units'],
            'Complete record support changed; do not truncate')
    result = VisualSpanTrainingExample(row['ink'].to(device), row['span_glyph_ends'],
        alignment['span_unit_ends'], row['prompt_spans'], references[selected],
        states[offsets[case['local']]:offsets[case['local'] + 1]])
    return result


def measure(model, objective, params, example, case, arm, reg, monitor, device):
    import torch
    from ilm.visual_lm.gradient_signal_accounting import PairMoments, adamw_displacement
    chunk = reg['resources']['chunk_elements']
    flow = {}
    loss_values = None
    auxiliary_geometry = defaultdict(PairMoments)
    combined_geometry = defaultdict(PairMoments)
    combined_direction = defaultdict(PairMoments)
    flow_direction = defaultdict(PairMoments)
    gradient_presence = {}
    norms = {}
    clips = {}
    began = time.perf_counter()
    for pass_name in reg['passes']:
        monitor.position = dict(arm=arm, role=case['role'], gradient_pass=pass_name)
        monitor.status('measuring_complete_record_gradients')
        monitor.guard()
        for _, parameter, _, _ in params:
            parameter.grad = None
        torch.manual_seed(reg['seed'])
        torch.cuda.manual_seed_all(reg['seed'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = objective(model, [example])
            values = {name: float(value.detach()) for name, value in losses.items()}
            require(len(values) == 6 and all(math.isfinite(value) for value in values.values()),
                    'Require all six finite complete-record losses')
            if pass_name == 'flow':
                selected = losses['output_flow']
            elif pass_name == 'auxiliary':
                selected = torch.stack([value for name, value in losses.items() if name != 'output_flow']).sum()
            else:
                selected = torch.stack(list(losses.values())).sum()
        if loss_values is None:
            loss_values = values
        require(values == loss_values, 'RNG replay changed the complete-record objective')
        selected.backward()
        del selected, losses
        present = {}
        if pass_name == 'flow':
            for name, parameter, _, _ in params:
                monitor.guard()
                if parameter.grad is not None:
                    flow[name] = parameter.grad.detach().cpu()
        else:
            geometry = auxiliary_geometry if pass_name == 'auxiliary' else combined_geometry
            for name, parameter, _, _ in params:
                monitor.guard()
                group = name.split('.', 1)[0]
                if parameter.grad is None and name not in flow:
                    continue
                gradient = parameter.grad.detach().view(-1) if parameter.grad is not None else None
                for first in range(0, parameter.numel(), chunk):
                    part = (gradient[first:first + chunk] if gradient is not None else
                            torch.zeros(min(chunk, parameter.numel() - first), device=device))
                    left = (flow[name].view(-1)[first:first + chunk].to(device)
                            if name in flow else torch.zeros_like(part))
                    geometry[group].add(left, part)
                    geometry['all'].add(left, part)
        for name, parameter, _, _ in params:
            group = name.split('.', 1)[0]
            present.setdefault(group, dict(parameters=0, present=0))
            present[group]['parameters'] += 1
            present[group]['present'] += parameter.grad is not None
        gradient_presence[pass_name] = present
        norm = torch.nn.utils.clip_grad_norm_([p for _, p, _, _ in params], 1., error_if_nonfinite=True)
        clip = torch.clamp(norm.new_tensor(1.) / (norm + 1e-6), max=1.)
        norms[pass_name], clips[pass_name] = float(norm), float(clip)
        if pass_name == 'combined':
            for name, parameter, state, settings in params:
                monitor.guard()
                group = name.split('.', 1)[0]
                weight = parameter.detach().view(-1)
                gradient = parameter.grad.detach().view(-1) if parameter.grad is not None else None
                for first in range(0, parameter.numel(), chunk):
                    if gradient is None and name not in flow:
                        continue
                    last = min(first + chunk, parameter.numel())
                    part = weight[first:last]
                    raw_flow = (flow[name].view(-1)[first:last].to(device)
                                if name in flow else torch.zeros_like(part))
                    first_moment = state['exp_avg'][first:last].to(device)
                    second_moment = state['exp_avg_sq'][first:last].to(device)
                    kwargs = dict(step=state['step'], **{key: settings[key]
                        for key in ('lr', 'betas', 'eps', 'weight_decay')})
                    together = adamw_displacement(part,
                        gradient[first:last] if gradient is not None else None,
                        first_moment, second_moment, **kwargs)
                    alone = adamw_displacement(part,
                        raw_flow * clips['flow'] if name in flow else None,
                        first_moment, second_moment, **kwargs)
                    combined_direction[group].add(raw_flow, together)
                    combined_direction['all'].add(raw_flow, together)
                    flow_direction[group].add(raw_flow, alone)
                    flow_direction['all'].add(raw_flow, alone)
        monitor.guard()
    for _, parameter, _, _ in params:
        parameter.grad = None
    require(bool(flow), 'No conditional-flow gradients reached the final actor')
    result = dict(arm=arm, case=case, complete_record=True, losses=loss_values,
        gradient_presence=gradient_presence, unclipped_total_norms=norms, global_clip_scalars=clips,
        flow_vs_auxiliary={key: value.result() for key, value in auxiliary_geometry.items()},
        flow_vs_combined_gradient={key: value.result() for key, value in combined_geometry.items()},
        flow_vs_combined_adamw_displacement={key: value.result() for key, value in combined_direction.items()},
        flow_vs_flow_only_adamw_displacement={key: value.result() for key, value in flow_direction.items()},
        optimizer_updates=0, seconds=time.perf_counter() - began,
        interpretation='First-order diagnostic on a new training batch with retained moments; no actual loss change or semantic result measured.')
    del flow
    gc.collect()
    return result


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Require registered physical GPU0 environment')
    reg = load(REG)
    require(not RUN.exists(), 'Preserve first execution; no automatic retry')
    head = require_pushed()
    import torch
    from ilm.visual_lm.trainable_visual_context import build_model
    from ilm.visual_lm.trainable_context_checkpoint import restore
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_full_rank_visual_span import training_source, source_targets
    RUN.mkdir(parents=True)
    monitor = GlyphReaderMonitor(RUN, reg['resources']['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds = reg['resources']['maximum_pressure_wait_seconds']
    monitor.status('verifying_registered_diagnostic')
    def timeout_handler(_signal, _frame):
        raise TimeoutError('Diagnostic external execution deadline reached')
    signal.signal(signal.SIGTERM, timeout_handler)
    results = []
    state_checks = []
    try:
        require(reg['packages'] == {name: importlib.metadata.version(name) for name in reg['packages']},
                'Registered packages changed')
        for name, digest in reg['input_sha256'].items():
            require(sha(ROOT / name) == digest, 'Diagnostic binding changed: ' + name)
        with full_reader_gpu() as (device, admission):
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            monitor.admission = admission
            monitor.guard()
            corpus, plan, cache, _, _ = training_source(ROOT / reg['teacher'])
            archive = open_compact_archive(ROOT.parent / 'LocalLLM')
            lookup, references = source_targets(archive, plan['training_native_units'])
            del archive
            for arm in reg['order']:
                monitor.position = dict(arm=arm)
                monitor.status('loading_exact_final_training_state')
                monitor.guard()
                saved = torch.load(ROOT / reg['arms'][arm]['latest'], weights_only=True, map_location='cpu', mmap=True)
                final = torch.load(ROOT / reg['arms'][arm]['deploy'], weights_only=True, map_location='cpu', mmap=True)
                require(saved['deploy']['updates'] == final['updates'] == 8172, 'Require final checkpoint')
                require(saved['deploy']['identity'] == final['identity'], 'Final actor identity differs')
                require(tensor_state_sha(saved['deploy']['image_heads']['encoder']) == tensor_state_sha(final['image_heads']['encoder'])
                        and tensor_state_sha(saved['deploy']['image_heads']['writer']) == tensor_state_sha(final['image_heads']['writer']),
                        'Retained final image heads differ')
                if arm == 'trainable':
                    require(tensor_state_sha(saved['deploy']['context']) == tensor_state_sha(final['context']),
                            'Retained final contextual tensors differ')
                del final
                model, objective, provenance = build_model(ROOT.parent / 'LocalLLM', ROOT / reg['codec'], device)
                restore(model, saved['deploy'], saved['deploy']['identity'])
                objective.load_state_dict(saved['teacher_projection'], strict=True)
                model.context.requires_grad_(arm == 'trainable')
                params = parameters_and_states(model, objective, saved, arm == 'trainable')
                before = tensor_state_sha(model.state_dict())
                objective_before = tensor_state_sha(objective.state_dict())
                for case in reg['cases']:
                    monitor.guard()
                    example = example_for(case, corpus, plan, cache, lookup, references, ROOT / reg['teacher'], device)
                    result = measure(model, objective, params, example, case, arm, reg, monitor, device)
                    results.append(result)
                    write_new(RUN / f"{arm}_{case['role']}.json", result)
                    print(json.dumps(dict(arm=arm, role=case['role'], complete_record=True,
                        combined_flow_dot=result['flow_vs_combined_adamw_displacement']['all']['inner_product'],
                        flow_only_dot=result['flow_vs_flow_only_adamw_displacement']['all']['inner_product'],
                        seconds=result['seconds'], optimizer_updates=0)), flush=True)
                    del example
                    gc.collect()
                require(tensor_state_sha(model.state_dict()) == before
                        and tensor_state_sha(objective.state_dict()) == objective_before,
                        'Diagnostic mutated actor or training-only projection')
                check = dict(arm=arm, exact_model_unchanged=True, exact_objective_unchanged=True,
                             model_tensor_sha256=before, objective_tensor_sha256=objective_before)
                state_checks.append(check)
                write_new(RUN / f'{arm}_unchanged.json', check)
                del model, objective, params, saved
                gc.collect()
                torch.cuda.empty_cache()
                monitor.guard()
            adverse = [row['case']['role'] for row in results if row['arm'] == 'trainable'
                       and row['flow_vs_combined_adamw_displacement']['all']['inner_product'] >= 0
                       and row['flow_vs_flow_only_adamw_displacement']['all']['inner_product'] < -1e-12]
            supported = 'complete_513_book' in adverse and len(adverse) >= 2
            result = dict(schema='ilm-visual-context-gradient-result-v1', registration_sha256=sha(REG),
                source_commit=head, results=results, state_checks=state_checks,
                interference_supported_on_prescribed_records=supported, adverse_treatment_roles=adverse,
                resources=monitor.receipt(), optimizer_updates=0, generation_calls=0,
                new_full_fit_authorized=False, language_quality_measured=False, goal_achieved=False)
            write_new(RUN / 'result.json', result)
            monitor.status('complete_gradient_diagnostic', interference_supported=supported)
    except BaseException as error:
        write_new(RUN / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            completed_cases=len(results), automatic_retry=False, optimizer_updates=0, full_fit_authorized=False))
        monitor.status('failed_gradient_diagnostic', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    arguments = parser.parse_args()
    register() if arguments.action == 'register' else run()
