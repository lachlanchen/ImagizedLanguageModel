#!/usr/bin/env python3
"""The original no-update diagnostic with its saved flow gradient on GPU."""
import argparse
from collections import defaultdict
import copy
import gc
import inspect
import math
import os
from pathlib import Path
import time

from scripts import diagnose_visual_context_gradients as base
from scripts.diagnose_visual_context_gradients import ROOT, load, sha, require, write_new

ORIGINAL_REG = ROOT / 'references/evidence/visual_context_gradient_registration_20261117.json'
REG = ROOT / 'references/evidence/visual_context_device_gradient_registration_20261117.json'
RUN = ROOT / 'artifacts/visual_context_device_gradient_diagnostic_v1'
PROTOCOL = ROOT / 'references/visual_context_device_gradient_protocol_20261117.md'
EQUIVALENCE = ROOT / 'references/evidence/visual_context_device_gradient_equivalence_20261117.json'

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
                    flow[name] = parameter.grad.detach().clone()
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


def equivalence():
    original = inspect.getsource(base.measure)
    expected = original.replace('flow[name] = parameter.grad.detach().cpu()',
                                'flow[name] = parameter.grad.detach().clone()')
    require(original.count('flow[name] = parameter.grad.detach().cpu()') == 1,
            'Original gradient placement changed')
    require(inspect.getsource(measure) == expected,
            'Numerical diagnostic changed beyond saved-gradient placement')
    return dict(schema='ilm-device-gradient-source-equivalence-v1',
        original_source_sha256=sha(Path(base.__file__)),
        device_source_sha256=sha(Path(__file__)),
        measure_source_exact_except_saved_flow_gradient_placement=True,
        original_run_and_resource_guard_reused=True, model_calls=0,
        original_copy='Detached FP32 gradient copied to host',
        replacement_copy='Detached FP32 gradient cloned on its original GPU',
        language_quality_proved=False, goal_achieved=False)


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Register on CPU')
    require(equivalence() == load(EQUIVALENCE), 'Source equivalence evidence differs')
    original = load(ORIGINAL_REG)
    for name, digest in original['input_sha256'].items():
        require(sha(ROOT / name) == digest, 'Original protected binding changed: ' + name)
    result = copy.deepcopy(original)
    additions = [Path(__file__), PROTOCOL, EQUIVALENCE, ORIGINAL_REG,
                 ROOT / 'references/evidence/visual_context_gradient_failure_20261117.json']
    result['input_sha256'].update({str(path.relative_to(ROOT)): sha(path) for path in additions})
    result.update(schema='ilm-device-flow-gradient-diagnostic-v1', run=str(RUN.relative_to(ROOT)),
        original_registration_sha256=sha(ORIGINAL_REG),
        single_execution_change='Keep the detached flow-gradient copy on GPU rather than host; all measurement arithmetic, records, actors, moments and guards unchanged.',
        saved_flow_gradient_placement='cuda_fp32', previous_failed_execution_preserved=True,
        unchanged_pressure_wait_allowance=True, unchanged_memory_and_time_ceilings=True)
    write_new(REG, result)
    print(__import__('json').dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=sha(REG),
                                       bindings=len(result['input_sha256']))))


def run():
    equivalence()
    base.REG, base.RUN, base.measure = REG, RUN, measure
    base.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('equivalence', 'register', 'run'))
    arguments = parser.parse_args()
    if arguments.action == 'equivalence':
        write_new(EQUIVALENCE, equivalence())
        print('Verified the single saved-gradient placement change; zero model calls')
    elif arguments.action == 'register':
        register()
    else:
        run()
