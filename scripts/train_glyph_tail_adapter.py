#!/usr/bin/env python3
"""Adapt the final contextual layers to complete glyph-feedback sequences."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/train_glyph_tail_adapter.py'
SOURCES = (OWN, 'ilm/visual_lm/glyph_tail_adapter.py', 'references/glyph_tail_adaptation_protocol.md',
    'scripts/train_semantic_glyph_language.py', 'ilm/visual_lm/semantic_glyph_loop.py',
    'ilm/visual_lm/image_context_interface.py', 'ilm/visual_lm/grounded_image_actor.py',
    'ilm/visual_lm/grounded_image_weights.py', 'ilm/visual_lm/semantic_image_memory.py',
    'ilm/visual_lm/complete_writing_data.py', 'ilm/visual_lm/causal_visual_prefix.py')
BASE = ROOT / 'artifacts/direct_pixel_readout_v1/g_only'
PRIMITIVE = ROOT / 'artifacts/semantic_glyph_bridge_v1'
GENUINE = ROOT / 'artifacts/genuine_glyph_readout_v1'
TRAIN = ROOT / 'data/general_instructions/complete_lines_train_20261105'
DEV = ROOT / 'data/general_instructions/complete_lines_development_20261105'
SEED = 202611117


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def identity():
    for name in SOURCES:
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit tail training sources first: ' + name)
    return {n: sha(ROOT / n) for n in SOURCES}


def prepare(output, reuse=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation')
    require(not output.exists(), 'preserve prior preparations')
    code = identity()
    if reuse is not None:
        old = load(reuse / 'preparation.json')
        for name, digest in old['code_sha256'].items():
            if name not in (OWN, 'references/glyph_tail_adaptation_protocol.md'):
                require(sha(ROOT / name) == digest, 'prepared-data ancestor implementation changed')
        for name, digest in old['bindings_sha256'].items():
            require(sha(ROOT / name) == digest, 'prepared-data ancestor input changed')
        output.mkdir(parents=True)
        for row in old['input_preview_boards']:
            require(sha(reuse / row['file']) == row['sha256'], 'viewed preview changed')
            shutil.copy2(reuse / row['file'], output / row['file'])
        require(sha(reuse / 'evaluation_references.json') == old['evaluation_references_sha256'], 'held references changed')
        shutil.copy2(reuse / 'evaluation_references.json', output / 'evaluation_references.json')
        bindings = dict(old['bindings_sha256'])
        bindings[rel(reuse / 'preparation.json')] = sha(reuse / 'preparation.json')
        save_json(output / 'preparation.json', dict(old, code_sha256=code, bindings_sha256=bindings,
            body_layers=36, tail_start_layer=32, prepared_data_ancestor=rel(reuse),
            prepared_data_ancestor_sha256=sha(reuse / 'preparation.json'),
            prior_preflight_weights_reused=False, prepared_images_and_selection_unchanged=True))
        print(json.dumps(dict(training_questions=len(old['training_jobs']), reused_input_images=True,
                              fresh_questions=len(old['fresh_cases']), prior_optimizer_state_used=False)))
        return
    import torch
    from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    torch.set_num_threads(2)
    source = ROOT / 'data/visual_qa/cmrc2018_short/train.jsonl'
    dev_source = source.with_name('development.jsonl')
    train = CompleteWritingImageData(TRAIN, purpose='train',
        expected_report_sha256='0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e')
    dev = CompleteWritingImageData(DEV, purpose='development')
    inventory = load(PRIMITIVE / 'inventory.json')['records']
    charmap = {c: i for i, row in enumerate(inventory) for c in row['chars']}
    bindings = {}

    def bind(path):
        bindings[rel(path)] = sha(path)
        return rel(path)

    for p in [source, dev_source, TRAIN / 'report.json', DEV / 'report.json',
              PRIMITIVE / 'training.pt', PRIMITIVE / 'inventory.json',
              BASE / 'fit/checkpoint_latest.pt', BASE / 'fit/report.json',
              GENUINE / 'preparation.json', GENUINE / 'cache/report.json']:
        bind(p)
    require(sha(BASE / 'fit/checkpoint_latest.pt') == load(BASE / 'fit/report.json')['checkpoint_sha256'],
            'fixed bridge changed')
    output.mkdir(parents=True)
    (output / 'inputs').mkdir()

    def page(dataset, name, stem):
        full, boundary = dataset.get(name)
        folded = fold_writing_prompt(full[..., :boundary * 64]).pixels
        a = ((folded[0, 0].numpy() + 1) * 127.5).astype(np.uint8)
        path = output / 'inputs' / (stem + '.png')
        Image.fromarray(a).save(path)
        return bind(path)

    jobs, skipped = [], []
    for row in map(json.loads, source.read_text().splitlines()):
        name = 'cmrc/' + row['id'] + '.pt'
        answer = row['references'][0]['text']
        missing = sorted(set(answer) - charmap.keys())
        if name not in train.names or missing:
            skipped.append(dict(id=row['id'], missing_from_older_cache=name not in train.names,
                                unsupported_reference_forms=missing))
            continue
        prompt = page(train, name, 'train_' + row['id'])
        jobs.append(dict(index=len(jobs), id=row['id'], prompt=prompt,
                         labels=[charmap[c] for c in answer], source_tensor_sha256=train.report['files_sha256'][name]))
    old = load(ROOT / 'artifacts/pixel_language_diversity_v1/manifest.json')
    dev_rows = [json.loads(x) for x in dev_source.read_text().splitlines()]
    by_id = {r['id']: r for r in dev_rows}
    excluded = {by_id[r['id']]['context_hash'] for r in old['development'] if r['source'] == 'cmrc'}
    excluded |= {r['context_hash'] for r in map(json.loads, source.read_text().splitlines())}
    groups = defaultdict(list)
    for row in dev_rows:
        if row['context_hash'] not in excluded and 'cmrc/' + row['id'] + '.pt' in dev.names:
            groups[row['context_hash']].append(row)
    key = lambda s: hashlib.sha256(('glyph-tail-fresh32-v1\0' + s).encode()).hexdigest()
    contexts = sorted([k for k, rows in groups.items() if len(rows) >= 2], key=key)[:16]
    require(len(contexts) == 16, 'need sixteen distinct eligible development contexts')
    fresh, references = [], {}
    for pair, context in enumerate(contexts):
        for row in sorted(groups[context], key=lambda r: key(r['id']))[:2]:
            name = 'cmrc/' + row['id'] + '.pt'
            index = len(fresh)
            prompt = page(dev, name, 'fresh_' + row['id'])
            fresh.append(dict(index=index, id=row['id'], pair=pair, context_hash=context, prompt=prompt,
                              source_tensor_sha256=dev.report['files_sha256'][name]))
            references[f'fresh/{index:04}/normal'] = [r['text'] for r in row['references']]
    save_json(output / 'evaluation_references.json', references)
    previews = []
    for start in range(0, 32, 2):
        a, b = [Image.open(ROOT / r['prompt']).convert('L') for r in fresh[start:start+2]]
        canvas = Image.new('L', (a.width + b.width + 24, max(a.height, b.height) + 28), 245)
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f'fresh {start:02} and {start+1:02}; complete prompt pixels', fill=0)
        canvas.paste(a, (4, 24)); canvas.paste(b, (a.width + 16, 24))
        name = f'input_preview_{start//2:02}.png'; canvas.save(output / name)
        previews.append(dict(file=name, sha256=sha(output / name)))
    save_json(output / 'preparation.json', dict(code_sha256=code, bindings_sha256=bindings,
        primitive=rel(PRIMITIVE), bridge=rel(BASE / 'fit/checkpoint_latest.pt'),
        genuine=rel(GENUINE), training_jobs=jobs, excluded_training=skipped,
        original_source_questions=3211, fresh_cases=fresh, input_preview_boards=previews,
        evaluation_references_sha256=sha(output / 'evaluation_references.json'),
        seed=SEED, fit_steps=18000, readout_learning_rate=.0001, adapter_learning_rate=.00002,
        tail_layers=4, preflight_cases=4, preflight_optimizer_updates=2,
        preflight_state_tolerance=.002, cache_seconds=14400., fit_seconds=10800.,
        maximum_generation_glyphs=64, inherited_pretraining_exposure_unknown=True,
        C3_used=False, diagnostic_trajectories_used=False, development_training=False,
        usable_language_demonstrated=False))
    print(json.dumps(dict(training_questions=len(jobs), skipped=len(skipped), fresh_questions=32)))


def verify(output):
    m = load(output / 'preparation.json')
    require(identity() == m['code_sha256'], 'tail implementation changed')
    for name, digest in m['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'prepared input changed: ' + name)
    return m


def cache(output, preflight):
    m = verify(output)
    import torch
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from ilm.visual_lm.glyph_tail_adapter import load_tail
    from scripts.train_semantic_glyph_language import reader, bridge_model, image, numerics, Monitor, atomic_tensor
    from scripts.score_pixel_language_first_blocks import diagnostic_gpu
    stage = 'preflight_cache' if preflight else 'cache'
    folder = output / stage; folder.mkdir(exist_ok=False)
    monitor = Monitor(folder, 1200. if preflight else m['cache_seconds'])
    monitor.status('admitting')
    try:
        files = []
        prior = None
        if not preflight:
            require(load(output / 'preflight_fit/report.json')['all_checks_passed'], 'pass actual two-update preflight first')
            prior = load(output / 'preflight_cache/report.json')
        with diagnostic_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); monitor.guard()
            interface, provenance = reader(device)
            bridge = bridge_model({}, device, ROOT / m['bridge'])
            data = torch.load(ROOT / m['primitive'] / 'training.pt', weights_only=True, map_location='cpu')
            images = (1 - 2 * data['ink'].float()).to(device)
            embedded = torch.cat([bridge.feedback_embedding(images[i:i+128]) for i in range(0,len(images),128)])
            capture = {}

            def hook(module, args, kwargs):
                capture['lower'] = args[0].detach().cpu().clone()

            start_layer = len(interface.actor.context.layers) - m['tail_layers']
            require(len(interface.actor.context.layers) == 36 and start_layer == 32, 'authenticated body layer identity differs')
            handle = interface.actor.context.layers[start_layer].register_forward_pre_hook(hook, with_kwargs=True)
            jobs = m['training_jobs'][:4] if preflight else m['training_jobs']
            zero_error = causal_error = None
            for index, job in enumerate(jobs):
                monitor.position = dict(completed=index, total=len(jobs), id=job['id']); monitor.guard()
                if prior is not None and index < 4:
                    row = prior['files'][index]
                    require(row['id'] == job['id'] and sha(ROOT / row['file']) == row['sha256'], 'sealed preflight cache changed')
                    files.append(row); continue
                args, geometry = interface.assemble(native_budget_image(image(ROOT / job['prompt'])))
                boundary = args['inputs_embeds'].shape[1]; n = len(job['labels'])
                physical = torch.arange(boundary,boundary+n,device=device)
                rotary = torch.arange(geometry['rotary_offset'],geometry['rotary_offset']+n,device=device)
                positions = torch.stack((physical,rotary,rotary,rotary))[:,None]
                args['inputs_embeds'] = torch.cat((args['inputs_embeds'], embedded[job['labels']][None]),1)
                args['position_ids'] = torch.cat((args['position_ids'],positions),-1)
                args['visual_pos_masks'] = torch.cat((args['visual_pos_masks'], torch.zeros((1,n),dtype=torch.bool,device=device)),1)
                response = interface.actor.context(**args,use_cache=False)
                wanted = response.last_hidden_state[:,boundary-1:].float().cpu().clone()
                lower = capture.pop('lower')
                if preflight and index == 0:
                    tail, tail_start = load_tail(device); tail.eval()
                    require(tail_start == start_layer, 'tail boundary differs')
                    result = tail(inputs_embeds=lower.to(device),position_ids=args['position_ids'],use_cache=False).last_hidden_state
                    zero_error = float((result[:,boundary-1:].cpu()-wanted).abs().max())
                    require(zero_error < m['preflight_state_tolerance'], 'zero tail differs from native body')
                    del result, tail
                    changed = dict(args); changed['inputs_embeds'] = args['inputs_embeds'].clone()
                    changed['inputs_embeds'][:,-1] = embedded[(job['labels'][-1]+1)%len(embedded)]
                    other = interface.actor.context(**changed,use_cache=False).last_hidden_state
                    causal_error = float((other[:,boundary-1:-1].cpu()-wanted[:,:-1]).abs().max())
                    require(causal_error < m['preflight_state_tolerance'], 'future target leaked into prior predictor')
                    capture.clear(); del other, changed
                path = folder / f'case_{index:04}.pt'
                atomic_tensor(path,dict(lower=lower,position_ids=args['position_ids'].cpu(),boundary=boundary,
                    labels=torch.tensor(job['labels']+[-1]),original_final_predictors=wanted,
                    id=job['id'],geometry=geometry))
                files.append(dict(id=job['id'],file=rel(path),sha256=sha(path),states=n+1,
                                  cache_bytes=path.stat().st_size))
                del args, response, lower, wanted
                monitor.status('caching_frozen_prefix_inputs')
                if index%16==0: print(json.dumps(monitor.position),flush=True)
            handle.remove()
            save_json(folder / 'report.json',dict(preparation_sha256=sha(output/'preparation.json'),files=files,
                tail_start_layer=start_layer,zero_tail_maximum_absolute_error=zero_error,
                future_glyph_prior_state_maximum_error=causal_error,provenance=provenance,
                source_vocabulary_parameters=0,lower_body_frozen=True,cache_FP32=True,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error)); raise
    finally:
        save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


def fit(output, preflight):
    m = verify(output)
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.glyph_tail_adapter import load_tail, adapter_state
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    from scripts.train_semantic_glyph_language import bridge_model, numerics
    from scripts.train_semantic_glyph_bridge import bridge_gpu, atomic_tensor
    from scripts import distill_semantic_glyph_readout as base
    from scripts.train_genuine_glyph_readout import pools
    stage = 'preflight_fit' if preflight else 'fit'
    folder=output/stage; folder.mkdir(exist_ok=False)
    monitor=base.Monitor(folder,dict(maximum_stage_seconds=1200. if preflight else m['fit_seconds'],
        maximum_vram_gib=12.,minimum_runtime_ram_gib=32.,maximum_pressure_wait_seconds=300.))
    monitor.status('admitting')
    try:
        source=output/('preflight_cache' if preflight else 'cache')
        require(load(source/'status.json')['phase']=='completed','complete frozen-prefix cache first')
        report=load(source/'report.json')
        if not preflight:
            review=load(output/'input_review.json')
            require(review['preparation_sha256']==sha(output/'preparation.json')
                    and review['all32_fresh_input_images_actually_viewed'],'review all new inputs before fitting')
        rows=report['files']
        for row in rows: require(sha(ROOT/row['file'])==row['sha256'],'frozen prefix changed')
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission):
            monitor.status('loading_tail',admission=admission)
            numerics(); torch.manual_seed(m['seed']); torch.cuda.manual_seed_all(m['seed'])
            tail,start=load_tail(device); tail.train()
            model=bridge_model({},device,ROOT/m['bridge'])
            model.readout.requires_grad_(True); model.stop.requires_grad_(True)
            data=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')
            images=(1-2*data['ink'].float()).to(device)
            copied=None
            if not preflight:
                copied=pools(ROOT/m['genuine'],load(ROOT/m['genuine']/'preparation.json'))['copy_train']
                copied=(copied[0].to(device),copied[1].to(device))
            body_params=[p for p in tail.parameters() if p.requires_grad]
            head_params=[p for p in model.parameters() if p.requires_grad]
            frozen=[(p,p._version) for p in [*tail.parameters(),*model.parameters()] if not p.requires_grad]
            initial_up={n:v.clone() for n,v in adapter_state(tail).items() if n.endswith('.up')}
            initial_readout={n:p.detach().cpu().clone() for n,p in model.readout.named_parameters()} if preflight else None
            opt=torch.optim.AdamW([dict(params=body_params,lr=m['adapter_learning_rate']),
                                  dict(params=head_params,lr=m['readout_learning_rate'])],weight_decay=.01)
            steps=m['preflight_optimizer_updates'] if preflight else m['fit_steps']
            minimum_up_gradient=None
            for step in range(steps):
                monitor.position=dict(step=step,total=steps); monitor.guard(); opt.zero_grad(set_to_none=True)
                if not preflight and step%4==3:
                    ix=torch.randint(len(copied[0]),(128,),device=device)
                    logits,_,_=model.predict(copied[0][ix])
                    pixel=reconstruction_loss(logits,images[copied[1][ix]]); stop=pixel*0
                else:
                    index=step%len(rows) if preflight else int(torch.randint(len(rows),()))
                    value=torch.load(ROOT/rows[index]['file'],weights_only=True,map_location='cpu',mmap=True)
                    lower=value['lower'].to(device); pos=value['position_ids'].to(device)
                    prediction=tail(inputs_embeds=lower,position_ids=pos,use_cache=False).last_hidden_state[0,value['boundary']-1:]
                    labels=value['labels'].to(device); logits,_,stops=model.predict(prediction)
                    pixel=reconstruction_loss(logits[:-1],images[labels[:-1]])
                    stop=.5*(F.softplus(-stops[-1])+F.softplus(stops[:-1]).mean())
                loss=pixel+stop
                require(bool(torch.isfinite(loss)),'nonfinite glyph-tail loss')
                loss.backward()
                if preflight:
                    up_gradient=sum(float(p.grad.square().sum()) for n,p in tail.named_parameters()
                                    if n.endswith('.up') and p.grad is not None)**.5
                    require(math.isfinite(up_gradient) and up_gradient>0,'no real contextual adapter-up gradient')
                    minimum_up_gradient=up_gradient if minimum_up_gradient is None else min(minimum_up_gradient,up_gradient)
                torch.nn.utils.clip_grad_norm_(body_params+head_params,1.); opt.step()
                fraction=(step+1)/steps; decay=.1+.9*.5*(1+math.cos(math.pi*fraction))
                for group,rate in zip(opt.param_groups,[m['adapter_learning_rate'],m['readout_learning_rate']]): group['lr']=rate*decay
                require(all(p._version==version for p,version in frozen),'frozen parameter changed')
                if (step+1)%100==0 or preflight:
                    item=dict(step=step+1,loss=float(loss.detach()),pixel=float(pixel.detach()),stop=float(stop.detach()))
                    with (folder/'history.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
                    monitor.status('training_glyph_context_and_readout',**item); print(json.dumps(item),flush=True)
                if (step+1)%1000==0 or step+1==steps:
                    path=folder/'checkpoint_latest.pt'
                    if path.exists(): path.replace(folder/'checkpoint_previous.pt')
                    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        body_tail_adapters=adapter_state(tail),tail_start_layer=start,
                        optimizer=opt.state_dict(),step=step+1,language_training_complete=not preflight,
                        preparation_sha256=sha(output/'preparation.json'),usable_language_demonstrated=False,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            adapters=adapter_state(tail)
            changed_up=sum(int((adapters[n]!=v).sum()) for n,v in initial_up.items())
            require(changed_up>0,'contextual adapter-up weights did not change')
            readout_changed=(sum(int((p.detach().cpu()!=initial_readout[n]).sum()) for n,p in model.readout.named_parameters())
                             if preflight else None)
            if preflight: require(readout_changed>0,'readout did not change')
            save_json(folder/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),
                checkpoint_sha256=sha(folder/'checkpoint_latest.pt'),actual_optimizer_updates=steps,
                trained_body_parameters=sum(p.numel() for p in body_params),
                trained_bridge_parameters=sum(p.numel() for p in head_params),changed_up_coordinates=changed_up,
                changed_readout_coordinates=readout_changed,minimum_preflight_up_gradient=minimum_up_gradient,
                all_frozen_parameter_versions_unchanged=True,all_checks_passed=True,
                preflight_weights_discarded=preflight,full_body_loaded=False,language_accuracy_not_measured=True,
                **monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error)); raise
    finally:
        save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','cache-preflight','fit-preflight','cache','fit'],required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/glyph_tail_adaptation_v1'))
    parser.add_argument('--reuse-prepared-data',type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    if args.stage=='prepare': prepare(output,args.reuse_prepared_data.resolve() if args.reuse_prepared_data else None)
    elif args.stage in ['cache-preflight','cache']: cache(output,args.stage=='cache-preflight')
    else: fit(output,args.stage=='fit-preflight')
