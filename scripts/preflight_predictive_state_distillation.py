#!/usr/bin/env python3
"""Finite real-record comparison of cosine versus predictive teacher transfer."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
REG = ROOT / 'references/evidence/predictive_state_distillation_registration_20261118.json'
PROTOCOL = ROOT / 'references/predictive_state_distillation_protocol_20261118.md'
RUN = ROOT / 'artifacts/predictive_state_distillation_preflight_v1'
OLD_REG = ROOT / 'references/evidence/visual_context_gradient_registration_20261117.json'
FILES = (str(Path(__file__).relative_to(ROOT)), str(PROTOCOL.relative_to(ROOT)),
         'ilm/visual_lm/predictive_state_distillation.py',
         'tests/test_predictive_state_distillation.py',
         'references/evidence/predictive_state_distillation_cpu_20261118.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Register with CUDA hidden')
    previous = original.load(OLD_REG)
    cpu = original.load(ROOT / FILES[-1])
    original.require(cpu['exit_code'] == 0 and cpu['tests'] == 4, 'Require numerical qualification')
    for name, digest in cpu['implementation_sha256'].items():
        original.require(original.sha(ROOT / name) == digest, 'CPU-qualified source changed')
    bindings = dict(previous['input_sha256'])
    bindings[str(OLD_REG.relative_to(ROOT))] = original.sha(OLD_REG)
    for name in FILES:
        bindings[name] = original.sha(ROOT / name)
    for name, digest in previous['input_sha256'].items():
        original.require(original.sha(ROOT / name) == digest, 'Protected source changed: ' + name)
    from ilm.visual_lm.grounded_image_weights import MODEL_SHA
    shared = ROOT.parent / 'LocalLLM/.local/models/ollama/blobs' / ('sha256-' + MODEL_SHA)
    original.require(original.sha(shared) == MODEL_SHA, 'Shared teacher archive changed')
    original.write_new(REG, dict(schema='ilm-predictive-state-preflight-v1', run=str(RUN.relative_to(ROOT)),
        input_sha256=bindings, packages=previous['packages'], cases=previous['cases'],
        initial=previous['arms']['trainable'], codec=previous['codec'], teacher=previous['teacher'],
        teacher_archive=str(shared), teacher_archive_sha256=MODEL_SHA,
        objective_change='Replace teacher_context cosine with complete KL(teacher||projected student), weight and temperature one.',
        order=['cosine_control', 'predictive_kl'], updates_per_arm=6, seed=202611181,
        writing_seed=202611182, flow_steps=8, readout_block_rows=2048,
        optimizer=dict(image_lr=.0002, context_lr=.00002, betas=[.9, .999], eps=1e-8,
                       weight_decay=.01, clip=1., fresh_moments=True, schedule=None),
        resources=dict(maximum_seconds=1800, maximum_pressure_wait_seconds=300,
            minimum_entry_host_gib=48, minimum_runtime_host_gib=24, minimum_free_gpu_gib=20,
            maximum_allocated_gpu_gib=22, minimum_free_disk_gib=48, cpu_threads=2),
        support=dict(all_updates=6, all_record_kl_below_initial=True,
            pooled_f1_gain_over_initial_and_control=.05, false_blanks_no_worse_than_initial_and_control=True),
        complete_records_only=True, retained_successor_model_permitted=False,
        reset_and_verify_required=True, automatic_retry=False, language_quality_measured=False,
        full_fit_authorized=False, goal_achieved=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG),
                          bindings=len(bindings), updates_per_arm=6)), flush=True)


def readout(guard):
    import numpy as np
    import torch
    from ilm.visual_lm.offline_visual_sequence_teacher import open_archive
    archive = open_archive(ROOT.parent / 'LocalLLM')
    source = archive.omitted['output.weight']
    rows = int(source.shape[-1])
    original.require(int(source.shape[0]) == 4096, 'Causal teacher readout width changed')
    head = torch.empty((rows, 4096), dtype=torch.float32, device='cpu')
    for first in range(0, rows, 2048):
        guard()
        block = archive.gguf.dequantize(source.data[first:first + 2048], source.tensor_type)
        original.require(block.shape == (min(2048, rows-first), 4096)
                         and np.isfinite(block).all(), 'Invalid teacher readout block')
        head[first:first + len(block)].copy_(torch.from_numpy(block))
    del archive
    return head


def readout_sha(head):
    digest = hashlib.sha256()
    for first in range(0, len(head), 2048):
        digest.update(memoryview(head[first:first + 2048].numpy()).cast('B'))
    return digest.hexdigest()


def objective_for(head, saved_projection, device, predictive):
    from ilm.visual_lm.compact_visual_span_training import CompactVisualSpanObjective
    from ilm.visual_lm.predictive_state_distillation import PredictiveStateObjective
    objective = (PredictiveStateObjective(head) if predictive else CompactVisualSpanObjective()).to(device)
    objective.load_state_dict(saved_projection, strict=True)
    return objective


def counts(ink, gold):
    import torch
    original.require(ink.shape == gold.shape and ink.dtype == torch.uint8, 'Complete physical support changed')
    a, b = ink.bool(), gold.bool()
    return dict(positions=len(ink), tp=int((a & b).sum()), fp=int((a & ~b).sum()), fn=int((~a & b).sum()),
        false_blanks=int((~a.flatten(1).any(1) & b.flatten(1).any(1)).sum()),
        exact=int((a == b).flatten(1).all(1).sum()), source_blanks=int((~b.flatten(1).any(1)).sum()))


def f1(values):
    return 2 * values['tp'] / max(1, 2 * values['tp'] + values['fp'] + values['fn'])


def pooled(rows):
    keys = ('positions', 'tp', 'fp', 'fn', 'false_blanks', 'exact', 'source_blanks')
    result = {key: sum(row['pixels'][key] for row in rows) for key in keys}
    return dict(result, f1=f1(result))


def assess(model, objective, examples, cases, head, reg, monitor, directory):
    import numpy as np
    import torch
    from ilm.visual_lm.compact_visual_span_training import CompactVisualSpanObjective
    from ilm.visual_lm.predictive_state_distillation import complete_readout_kl
    directory.mkdir()
    result = []
    model.eval()
    for index, (example, case) in enumerate(zip(examples, cases)):
        monitor.position = dict(assessment=str(directory.relative_to(RUN)), role=case['role'])
        monitor.status('assessing_complete_training_record')
        monitor.guard()
        projected = []
        handle = objective.teacher_projection.register_forward_hook(lambda _m, _i, out: projected.append(out))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(reg['seed']); torch.cuda.manual_seed_all(reg['seed'])
            try:
                old_losses = CompactVisualSpanObjective.forward(objective, model, [example])
            finally:
                handle.remove()
            original.require(len(projected) == 1, 'One projected complete state route required')
            kl = float(complete_readout_kl(projected[0].float(), example.teacher_states.to(projected[0].device).float(),
                        head, block_rows=reg['readout_block_rows']))
            # Capture the actual target-preceding context from the same causal
            # observation route. Its complete gold prefixes are diagnostic input.
            ends = example.glyph_ends.tolist()
            spans = list(example.ink.split([b-a for a,b in zip([0, *ends[:-1]], ends)]))
            units = torch.diff(example.update_ends, prepend=example.update_ends.new_zeros(1)).to(example.ink.device)
            observed = model.observe(spans, units, [len(spans)])
            first = max(0, example.prompt_spans - 1)
            context = observed.contexts[first:]
            targets = spans[first + 1:]
            lengths = [len(t) for t in targets] + [0]
            length_guess = model.writer.length(context.float()).argmax(-1)
            output = [None] * len(targets)
            for length in sorted(set(lengths) - {0}):
                monitor.guard()
                locations = [i for i, target in enumerate(targets) if len(target) == length]
                noise = []
                for location in locations:
                    generator = torch.Generator(device=example.ink.device).manual_seed(
                        reg['writing_seed'] + 100000 * index + location)
                    noise.append(torch.randn((length, model.writer.latent_width), device=example.ink.device,
                                             generator=generator))
                value = torch.stack(noise)
                valid = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
                condition = context[locations]
                for step in range(reg['flow_steps']):
                    phase = torch.full((len(value),), step / reg['flow_steps'], device=value.device)
                    value = value + model.writer.velocity(value, phase, condition, valid).float() / reg['flow_steps']
                generated = (model.codec.decode(value.flatten(0, 1)) < 0).to(torch.uint8).split(length)
                for location, ink in zip(locations, generated):
                    output[location] = ink
            original.require(all(item is not None for item in output), 'Every target span must be emitted')
            ink = torch.cat(output).cpu()
            gold = torch.cat(targets).cpu()
            pixels = counts(ink, gold)
            path = directory / (case['role'] + '.npz')
            np.savez_compressed(path, ink=ink.numpy(), span_lengths=np.array(lengths[:-1], dtype=np.int32))
            row = dict(case=case, predictive_kl=kl, old_losses={k: float(v) for k,v in old_losses.items()},
                pixels=dict(pixels, f1=f1(pixels)), next_span_length_correct=int((length_guess.cpu() == torch.tensor(lengths)).sum()),
                length_decisions=len(lengths), natural_stop_predicted_at_terminal=bool(length_guess[-1] == 0),
                provided_lengths=True, correct_image_history=True, autonomous_language=False,
                raw_file=str(path.relative_to(RUN)), raw_sha256=original.sha(path))
            result.append(row)
            original.write_new(directory / (case['role'] + '.json'), row)
            del old_losses, observed, context, projected
    model.train(); model.codec.eval()
    return result


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Require physical GPU0 deterministic environment')
    reg = original.load(REG)
    original.require(not RUN.exists(), 'Preserve all attempts; no automatic retry')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/main'], text=True, timeout=30).split()[0]
    original.require(commit == remote, 'Commit and push before execution')
    for name in [str(REG.relative_to(ROOT)), *FILES]:
        original.require(subprocess.check_output(['git','show','HEAD:'+name]) == (ROOT/name).read_bytes(), 'Uncommitted bound source')
    import torch
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.trainable_visual_context import build_model
    from ilm.visual_lm.trainable_context_checkpoint import restore
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from ilm.visual_lm.offloaded_adamw import OffloadedAdamW
    from scripts.train_full_rank_visual_span import training_source, source_targets
    RUN.mkdir()
    monitor = GlyphReaderMonitor(RUN, reg['resources']['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds = reg['resources']['maximum_pressure_wait_seconds']
    signal.signal(signal.SIGTERM, lambda _s,_f: (_ for _ in ()).throw(TimeoutError('External preflight deadline')))
    arm_results = []
    updates_done = []
    try:
        monitor.status('verifying_registered_preflight')
        for name, digest in reg['input_sha256'].items():
            original.require(original.sha(ROOT/name) == digest, 'Bound source changed: '+name)
        original.require(reg['packages'] == {n:importlib.metadata.version(n) for n in reg['packages']}, 'Packages changed')
        original.require(original.sha(Path(reg['teacher_archive'])) == reg['teacher_archive_sha256'], 'Teacher archive changed')
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            monitor.guard()
            head = readout(monitor.guard)
            head_sha = readout_sha(head)
            corpus, plan, cache, _, _ = training_source(ROOT/reg['teacher'])
            archive = open_compact_archive(ROOT.parent/'LocalLLM')
            lookup, references = source_targets(archive, plan['training_native_units'])
            del archive
            examples = [original.example_for(c,corpus,plan,cache,lookup,references,ROOT/reg['teacher'],device) for c in reg['cases']]
            initial = torch.load(ROOT/reg['initial']['deploy'], weights_only=True, map_location='cpu', mmap=True)
            training = torch.load(ROOT/reg['initial']['latest'], weights_only=True, map_location='cpu', mmap=True)
            projection = training['teacher_projection']
            model, unused, provenance = build_model(ROOT.parent/'LocalLLM', ROOT/reg['codec'], device)
            del unused
            restore(model, initial, initial['identity'])
            model.context.requires_grad_(True)
            before = tensor_state_sha(model.state_dict())
            # The same loaded visual actor is reset before and after each arm.
            initial_assessment = None
            for arm in reg['order']:
                monitor.guard()
                restore(model, initial, initial['identity'])
                original.require(tensor_state_sha(model.state_dict()) == before, 'Initial model reset failed')
                objective = objective_for(head, projection, device, arm == 'predictive_kl')
                projection_before = tensor_state_sha(objective.state_dict())
                original.require(set(objective.state_dict()) == set(projection), 'Training readout entered saved state')
                if initial_assessment is None:
                    initial_assessment = assess(model,objective,examples,reg['cases'],head,reg,monitor,RUN/'initial')
                if arm == 'predictive_kl':
                    control = objective_for(head, projection, device, False)
                    torch.manual_seed(reg['seed']); torch.cuda.manual_seed_all(reg['seed'])
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        a = control(model,[examples[0]])
                        torch.manual_seed(reg['seed']); torch.cuda.manual_seed_all(reg['seed'])
                        b = objective(model,[examples[0]])
                    original.require(all(torch.equal(a[k],b[k]) for k in a if k != 'teacher_context'), 'Other objectives changed')
                    original.write_new(RUN/'unchanged_other_losses.json',dict(exact=True,losses=[k for k in a if k!='teacher_context']))
                    del control, a, b
                image = [*model.encoder.parameters(),*model.writer.parameters(),*objective.parameters()]
                context = list(model.context.parameters())
                original.require(all(p.requires_grad for p in image+context), 'Complete contextual adaptation required')
                opt = OffloadedAdamW([dict(params=image,lr=reg['optimizer']['image_lr']),
                    dict(params=context,lr=reg['optimizer']['context_lr'])],guard=monitor.guard)
                steps = []
                for step in range(reg['updates_per_arm']):
                    case_index = step % len(examples)
                    monitor.position = dict(arm=arm, update=step+1, role=reg['cases'][case_index]['role'])
                    monitor.status('updating_complete_training_record'); monitor.guard()
                    opt.zero_grad(set_to_none=True)
                    torch.manual_seed(reg['seed']+step); torch.cuda.manual_seed_all(reg['seed']+step)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        losses = objective(model,[examples[case_index]])
                        loss = torch.stack(list(losses.values())).sum()
                    original.require(bool(torch.isfinite(loss)), 'Nonfinite preflight objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(image+context,1.,error_if_nonfinite=True)
                    opt.step()
                    item = dict(arm=arm,update=step+1,case=reg['cases'][case_index],losses={k:float(v) for k,v in losses.items()},
                        gradient_norm=float(norm),complete_record=True,optimizer_moment_bytes=opt.moment_bytes())
                    original.write_new(RUN/f'{arm}_update_{step+1:02d}.json',item)
                    steps.append(item); updates_done.append(dict(arm=arm,update=step+1))
                    del loss, losses
                opt.zero_grad(set_to_none=True)
                after = assess(model,objective,examples,reg['cases'],head,reg,monitor,RUN/arm)
                original.require(tensor_state_sha(model.state_dict()) != before, 'No actual visual model update occurred')
                arm_results.append(dict(arm=arm,updates=steps,assessment=after,pooled=pooled(after)))
                del opt, image, context
                restore(model, initial, initial['identity'])
                objective.load_state_dict(projection,strict=True)
                original.require(tensor_state_sha(model.state_dict()) == before
                    and tensor_state_sha(objective.state_dict()) == projection_before, 'Update reset failed')
                original.write_new(RUN/f'{arm}_reset.json',dict(model_exact=True,projection_exact=True,model_sha256=before,
                    projection_sha256=projection_before))
                del objective
                gc.collect(); torch.cuda.empty_cache(); monitor.guard()
            original.require(readout_sha(head) == head_sha, 'Frozen teacher readout changed')
            a,b = arm_results
            base = pooled(initial_assessment)
            kl_down = all(y['predictive_kl'] < x['predictive_kl'] for x,y in zip(initial_assessment,b['assessment']))
            writing_gain = b['pooled']['f1'] >= max(base['f1'],a['pooled']['f1'])+.05
            blanks = b['pooled']['false_blanks'] <= min(base['false_blanks'],a['pooled']['false_blanks'])
            result = dict(schema='ilm-predictive-state-preflight-result-v1',registration_sha256=original.sha(REG),
                source_commit=commit,initial=initial_assessment,initial_pooled=base,arms=arm_results,
                support_checks=dict(all_record_kl_down=kl_down,writing_gain_at_least_005=writing_gain,false_blanks_no_worse=blanks),
                support=kl_down and writing_gain and blanks,resources=monitor.receipt(),
                teacher_readout_shape=list(head.shape),teacher_readout_sha256=head_sha,
                deployed_model_parameters=sum(p.numel() for p in model.parameters()),all_trial_updates_reset=True,
                model_tensor_sha256=before,retained_successor=False,autonomous_language_measured=False,
                full_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result)
            monitor.status('complete_predictive_state_preflight',support=result['support'])
            print(json.dumps({k:result[k] for k in ['support','support_checks','initial_pooled','resources']},default=str),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            completed_updates=updates_done,retained_successor=False,automatic_retry=False,full_fit_authorized=False))
        monitor.status('failed_predictive_state_preflight',error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action == 'register' else run()
