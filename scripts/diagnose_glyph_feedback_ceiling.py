#!/usr/bin/env python3
"""Teacher-only feedback counterfactual; never an autonomous ILM result.

Compare native-unit, exact-character-vector, and learned-image feedback with
the same frozen image prefill and complete source head. Evaluation strings
enter only the later scorer. No diagnostic output may become training data.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/diagnose_glyph_feedback_ceiling.py'
PROTOCOL = 'references/glyph_feedback_ceiling_protocol.md'
CANDIDATE = ROOT / 'artifacts/direct_pixel_readout_v1/g_only'
LANGUAGE = ROOT / 'artifacts/semantic_glyph_language_v1'
CORPUS = ROOT / 'artifacts/pixel_language_diversity_v1'
MAPPING = ROOT / 'artifacts/semantic_glyph_readout_distillation_v1/mapping.json'
SOURCES = (OWN, PROTOCOL, 'scripts/train_semantic_glyph_language.py',
           'scripts/check_native_visual_transfer_v71.py',
           'ilm/visual_lm/semantic_glyph_loop.py', 'ilm/visual_lm/image_context_interface.py',
           'ilm/visual_lm/semantic_image_memory.py', 'ilm/visual_lm/grounded_image_weights.py',
           'ilm/visual_lm/grounded_image_actor.py', 'scripts/score_pixel_language_first_blocks.py')
ROUTES = ('native_units', 'exact_character_vector', 'canonical_image_encoder')


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def identity():
    for name in SOURCES:
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit diagnostic sources first: ' + name)
    return {name: sha(ROOT / name) for name in SOURCES}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires hidden CUDA')
    require(not output.exists(), 'preserve earlier diagnostic runs')
    code = identity()
    require(load(CANDIDATE / 'assessment/status.json')['phase'] == 'completed'
            and load(CANDIDATE / 'raw_review/manual_review.json')['all40_actual_raw_outputs_viewed'],
            'complete and actually review the image candidate first')
    meta = load(LANGUAGE / 'preparation.json')
    corpus = load(CORPUS / 'manifest.json')
    raw = load(CANDIDATE / 'assessment/report.json')['records']
    bindings = {}

    def bind(path):
        bindings[rel(path)] = sha(path)
        return rel(path)

    for path in [LANGUAGE / 'preparation.json', CORPUS / 'manifest.json', MAPPING,
                 CANDIDATE / 'preparation.json', CANDIDATE / 'fit/report.json',
                 CANDIDATE / 'assessment/report.json', CANDIDATE / 'raw_review/manual_review.json',
                 CANDIDATE / 'raw_review/canonical_scores.json']:
        bind(path)
    primitive = ROOT / meta['primitive']
    for name in ['training.pt', 'inventory.json', 'preparation.json']:
        bind(primitive / name)
    checkpoint = CANDIDATE / 'fit/checkpoint_latest.pt'
    require(sha(checkpoint) == load(CANDIDATE / 'fit/report.json')['checkpoint_sha256'], 'candidate changed')
    bind(checkpoint)
    jobs = []
    for split, rows in [('train', meta['jobs'][:8]), ('development', corpus['development'][:16])]:
        for index, row in enumerate(rows):
            path = ROOT / row['prompt'] if split == 'train' else CORPUS / row['normal']
            expected = meta['bindings_sha256'][rel(path)] if split == 'train' else row['normal_sha256']
            require(sha(path) == expected, 'previously assessed input changed')
            prior = next(v for v in raw if v['split'] == split and v['index'] == index and v['route'] == 'normal')
            state = CANDIDATE / 'assessment' / prior['file']
            require(sha(state) == prior['sha256'], 'actual image trajectory changed')
            jobs.append(dict(split=split, index=index, prompt=bind(path),
                             first_state_file=bind(state)))
    output.mkdir(parents=True)
    save_json(output / 'preparation.json', dict(code_sha256=code, bindings_sha256=bindings,
        primitive=rel(primitive), checkpoint=rel(checkpoint), mapping=rel(MAPPING),
        jobs=jobs, routes=list(ROUTES), maximum_steps=64, maximum_seconds=2400.,
        prefill_maximum_absolute_error=.002, source_head_rounding='BF16 then FP32, as prior reader helper',
        unsupported_source_action='record and terminate that route; never replace or reject the case',
        evaluation_references_read=False, training_data_reuse_allowed=False,
        C3_used=False, optimizer_updates=0, helper_only_symbolic_decoding=True,
        ILM_image_answers_generated=0, usable_language_demonstrated=False))


def verify(output):
    meta = load(output / 'preparation.json')
    require(identity() == meta['code_sha256'], 'fixed diagnostic source changed')
    for name, digest in meta['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'fixed diagnostic input changed: ' + name)
    return meta


def run(output):
    meta = verify(output)
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    from scripts.train_semantic_glyph_language import Monitor, reader, bridge_model, image, numerics, atomic_tensor
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer, embed_rows
    from scripts.score_pixel_language_first_blocks import diagnostic_gpu
    folder = output / 'execution'
    folder.mkdir(exist_ok=False)
    monitor = Monitor(folder, meta['maximum_seconds'])
    monitor.status('admitting')
    try:
        records = []
        with diagnostic_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission
            numerics()
            monitor.guard()
            interface, provenance = reader(device)
            bridge = bridge_model({}, device, ROOT / meta['checkpoint'])
            archive, tokenizer, _ = archive_and_tokenizer()
            source = archive.omitted['output.weight']
            array = np.array(archive.gguf.dequantize(source.data, source.tensor_type), dtype=np.float32, copy=True)
            head = torch.from_numpy(array).to(device, torch.bfloat16).float()
            del array
            mapping = load(ROOT / meta['mapping'])['teacher_to_glyph']
            require(len(mapping) == len(head), 'teacher mapping shape differs')
            primitive = ROOT / meta['primitive']
            data = torch.load(primitive / 'training.pt', weights_only=True, map_location='cpu')
            inventory = load(primitive / 'inventory.json')['records']
            char_ids = {c: k for row in inventory for c, k in zip(row['chars'], row['source_ids'])}
            images = (1 - 2 * data['ink'].float()).to(device)
            encoded = torch.cat([bridge.feedback_embedding(images[i:i+128]) for i in range(0, len(images), 128)])
            hook = interface.actor.context.register_forward_pre_hook(lambda module, args: monitor.guard())
            for job in meta['jobs']:
                observed = torch.load(ROOT / job['first_state_file'], weights_only=True, map_location='cpu')
                for route in meta['routes']:
                    monitor.position = dict(split=job['split'], index=job['index'], route=route,
                                            completed=len(records), total=72, step=0)
                    arguments, geometry = interface.assemble(native_budget_image(image(ROOT / job['prompt'])))
                    response = interface.actor.context(**arguments, use_cache=True)
                    state, past = response.last_hidden_state[:, -1:], response.past_key_values
                    del response, arguments
                    initial_error = float((state[0, -1].float().cpu() - observed['states'][0]).abs().max())
                    require(initial_error < meta['prefill_maximum_absolute_error'], 'image prefill is not matched')
                    require(geometry == observed['observer_geometry'], 'image geometry differs')
                    selected, forms, states, feedback = [], [], [], []
                    offset = geometry['rotary_offset']
                    reason = 'cap'
                    for step in range(meta['maximum_steps'] + 1):
                        monitor.position['step'] = step
                        monitor.guard()
                        states.append(state[0, -1].float().cpu())
                        winner = int(F.linear(state[:, -1].float(), head).argmax(-1))
                        label = mapping[winner]
                        token_text = tokenizer.decode([winner], skip_special_tokens=False)
                        selected.append(dict(id=winner, decoded_unit=token_text, first_glyph_label=label))
                        if label == -1:
                            reason = 'native_terminal'
                            break
                        if step == meta['maximum_steps']:
                            break
                        if route == 'native_units':
                            embedded = embed_rows(archive, [winner]).to(device)
                        else:
                            if label < 0:
                                reason = 'unsupported_first_complete_form'
                                break
                            char = token_text[0]
                            require(char in inventory[label]['chars'] and char in char_ids,
                                    'mapped first complete character differs')
                            forms.append(char)
                            embedded = (embed_rows(archive, [char_ids[char]]).to(device)
                                        if route == 'exact_character_vector' else encoded[label:label+1])
                            feedback.append(dict(char=char, image_sha256=tensor_hash(images[label:label+1]),
                                                 embedding_sha256=tensor_hash(embedded)))
                        position = torch.tensor([past.get_seq_length(), offset, offset, offset], device=device).reshape(4,1,1)
                        response = interface.actor.context(inputs_embeds=embedded[:, None], position_ids=position,
                            past_key_values=past, use_cache=True)
                        state, past = response.last_hidden_state[:, -1:], response.past_key_values
                        del response, embedded
                        offset += 1
                    # The final entry queries stopping/cap and is not consumed.
                    used_units = [v['id'] for v in selected[:-1]]
                    text = tokenizer.decode(used_units, skip_special_tokens=False) if route == 'native_units' else ''.join(forms)
                    stem = f"{job['split']}_{job['index']:04}_{route}"
                    value = dict(selected=selected, forms=forms, feedback=feedback, text=text,
                                 states=torch.stack(states), reason=reason, geometry=geometry,
                                 initial_state_maximum_absolute_error=initial_error)
                    atomic_tensor(folder / (stem + '.pt'), value)
                    records.append(dict(split=job['split'], index=job['index'], route=route,
                        text=text, reason=reason, emitted_steps=len(used_units) if route == 'native_units' else len(forms),
                        first_decoded_source_unit=selected[0]['decoded_unit'],
                        tensor=stem + '.pt', tensor_sha256=sha(folder / (stem + '.pt')),
                        initial_state_maximum_absolute_error=initial_error))
                    del state, past, states
                    monitor.status('teacher_feedback_counterfactual')
                    print(json.dumps(records[-1], ensure_ascii=False), flush=True)
            hook.remove()
            require(interface.actor.context.embed_tokens is None, 'teacher embedding table entered actor')
            save_json(folder / 'report.json', dict(preparation_sha256=sha(output / 'preparation.json'),
                records=records, provenance=provenance, evaluation_references_read=False,
                helper_only_symbolic_decoding=True, source_head_attached_to_actor=False,
                runtime_font=False, canonical_glyph_bank_used_by_helper=True,
                ILM_image_answers_generated=0, optimizer_updates=0, training_data_reuse_allowed=False,
                **monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed', reason=repr(error))
        raise
    finally:
        save_json(folder / 'status.json', dict(load(folder / 'status.json'), pid=None))


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU scoring requires hidden CUDA')
    verify(output)
    require(load(output / 'execution/status.json')['phase'] == 'completed', 'complete all routes before scoring')
    records = load(output / 'execution/report.json')['records']
    require(len(records) == 72, 'incomplete counterfactual')
    corpus = load(CORPUS / 'manifest.json')
    train_jobs = load(LANGUAGE / 'preparation.json')['jobs'][:8]
    source = {split: {r['id']: r for r in map(json.loads, (ROOT / f'data/visual_qa/cmrc2018_short/{split}.jsonl').read_text().splitlines())}
              for split in ('train', 'development')}
    results = []
    for row in records:
        split, index = row['split'], row['index']
        case_id = Path(train_jobs[index]['source']).stem if split == 'train' else corpus['development'][index]['id']
        references = [v['text'] for v in source[split][case_id]['references']]
        results.append(dict(**row, references=references,
            literal_reference_exact=bool(row['reason'] == 'native_terminal' and row['text'].strip() in references)))
    groups = {}
    for split in ('train', 'development'):
        for route in ROUTES:
            rows = [r for r in results if r['split'] == split and r['route'] == route]
            groups[split + '/' + route] = dict(cases=len(rows),
                literal_reference_exact=sum(r['literal_reference_exact'] for r in rows),
                native_terminal=sum(r['reason'] == 'native_terminal' for r in rows),
                unsupported=sum(r['reason'] == 'unsupported_first_complete_form' for r in rows),
                capped=sum(r['reason'] == 'cap' for r in rows))
    save_json(output / 'scores.json', dict(records=results, groups=groups,
        helper_only_symbolic_decoding=True, learned_ILM_accuracy=False,
        teacher_language_quality_is_not_a_physical_ceiling=True,
        training_data_reuse_allowed=False, source_pretraining_exposure_unknown=True))
    print(json.dumps(groups, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run', 'score'), required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/glyph_feedback_ceiling_v1'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.stage](args.output.resolve())
