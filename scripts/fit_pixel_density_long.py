#!/usr/bin/env python3
"""Fixed longer fitting control for the unchanged source-bound pixel head."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.train_conditional_pixel_density import (ROOT, SEED, atomic_tensor,
    equal_tree, numerical_checks, require, save_json, sha, small_head_gpu, unchanged)

UPDATES, BATCH = 16000, 32
FILES = ('scripts/fit_pixel_density_long.py', 'references/pixel_density_long_fit_protocol.md')


def load(path):
    return json.loads(Path(path).read_text())


def check_code():
    for name in FILES:
        content = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        require(hashlib.sha256(content).hexdigest() == sha(ROOT/name), 'commit longer-fit code first')
    return {name: sha(ROOT/name) for name in FILES}


def prepare(source, diagnosis, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA during preparation')
    code, original = check_code(), unchanged(source)
    diagnostic = load(diagnosis/'report.json')
    report = load(source/'execution/report.json')
    require(diagnostic['source_checkpoint_sha256'] == report['final_checkpoint_sha256']
        and diagnostic['actual_correct_prefix_forwards'] == 896
        and diagnostic['groups']['train_cmrc_normal']['correct_prefix_piece_accuracy'] < .99,
        'completed correct-prefix fitting diagnosis required')
    import torch
    torch.set_num_threads(4)
    data = torch.load(source/'data.pt', map_location='cpu', weights_only=True)
    old_schedule = data['schedule'].clone()
    generator = torch.Generator().manual_seed(SEED)
    data['schedule'] = torch.stack([torch.randperm(512, generator=generator).reshape(16, BATCH)
        for _ in range(1000)]).reshape(UPDATES, BATCH)
    require(torch.equal(data['schedule'][:1200], old_schedule), 'original example order changed')
    output.mkdir(parents=True, exist_ok=False)
    atomic_tensor(output/'data.pt', data)
    shutil.copyfile(source/'cases.json', output/'cases.json')
    bound = dict(original['bound_sha256'])
    for path in (source/'preparation.json', source/'execution/report.json',
            source/'execution/initial.pt', source/'execution/nll.pt', diagnosis/'report.json'):
        bound[str(path.relative_to(ROOT))] = sha(path)
    bound.update(code)
    meta = dict(original, bound_sha256=bound, data_sha256=sha(output/'data.pt'),
        cases_sha256=sha(output/'cases.json'), updates=UPDATES, batch=BATCH, epochs=1000,
        implementation_sha256=code, original_execution=str(source.relative_to(ROOT)),
        initial_checkpoint=str((source/'execution/initial.pt').relative_to(ROOT)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        maximum_execution_seconds=900, maximum_allocated_vram_gib=4.,
        monitored_full_training_nll_updates=[0,1200,4000,8000,16000],
        quality_checkpoint_selection=False, general_ilm_goal_complete=False)
    save_json(output/'preparation.json', meta)
    print(json.dumps(dict(prepared=str(output), updates=UPDATES, epochs=1000)), flush=True)


def run(output):
    check_code()
    meta = unchanged(output)
    require(meta['implementation_sha256'] == check_code(), 'longer-fit implementation changed')
    execution = output/'execution'
    execution.mkdir(exist_ok=False)
    started, completed = time.perf_counter(), 0
    def status(phase, **extra):
        save_json(execution/'status.json', dict(phase=phase, pid=os.getpid(), completed_updates=completed,
            elapsed_seconds=time.perf_counter()-started, **extra))
    try:
        with small_head_gpu() as (device, admitted):
            import numpy as np
            import torch
            from torch.nn import functional as F
            from PIL import Image
            from ilm.visual_lm.conditional_pixel_density import ConditionalPixelDensity, unpack_pixels
            data = torch.load(output/'data.pt', map_location='cpu', weights_only=True)
            source_initial = torch.load(ROOT/meta['initial_checkpoint'], map_location='cpu', weights_only=True)
            model = ConditionalPixelDensity(data['mean'], data['scale']).to(device)
            model.load_state_dict(source_initial['model'])
            require(all(torch.equal(v.cpu(), source_initial['model'][k]) for k,v in model.state_dict().items()),
                'the original untrained head must initialize this control exactly')
            h, blank, target = (data[k].to(device) for k in ('normal','blank','patterns'))
            status('numerical_preflight', admission=admitted)
            checks = numerical_checks(model, h[:2], target[:2])
            save_json(execution/'numerical_preflight.json', checks)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.01)
            def nll(features, truth):
                model.eval()
                values = []
                with torch.no_grad():
                    for i in range(0, len(features), 64):
                        logits = model(features[i:i+64], truth[i:i+64])
                        values.append(-F.log_softmax(logits,-1).gather(-1,truth[i:i+64,:,None]).squeeze(-1).mean(1).cpu())
                return torch.cat(values)
            def measure_all():
                return torch.cat((nll(h,target), nll(blank,target[512:])))
            initial_nll = measure_all()
            monitor = [dict(update=0, train_zh_nll=float(initial_nll[:256].mean()),
                train_en_nll=float(initial_nll[256:512].mean()))]
            history, replay = [], None
            training_started = time.perf_counter()
            for i, ix in enumerate(data['schedule']):
                step = i+1
                rate = 3e-4 * min(step/32,1.) * (.1+.9*.5*(1+math.cos(math.pi*i/(UPDATES-1))))
                model.train()
                def update():
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = rate
                    logits = model(h[ix],target[ix])
                    loss = -F.log_softmax(logits,-1).gather(-1,target[ix,:,None]).mean()
                    require(bool(torch.isfinite(loss)), 'nonfinite fitting loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    require(bool(torch.isfinite(norm)), 'nonfinite fitting gradient')
                    optimizer.step()
                    return float(loss),float(norm)
                loss,norm = update()
                if step == 2:
                    atomic_tensor(execution/'update0002.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
                    checkpoint = torch.load(execution/'update0002.pt',map_location='cpu',weights_only=True)
                if step == 3:
                    expected_model, expected_optimizer = copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict())
                    model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
                    repeated_loss,repeated_norm=update()
                    require(loss==repeated_loss and norm==repeated_norm and equal_tree(expected_model,model.state_dict())
                        and equal_tree(expected_optimizer,optimizer.state_dict()), 'longer-fit exact update replay failed')
                    replay=dict(checkpoint_step=2,replayed_step=3,loss_bit_exact=True,gradient_norm_bit_exact=True,
                        all_parameters_bit_exact=True,optimizer_state_bit_exact=True)
                    save_json(execution/'update_replay.json',replay)
                    del checkpoint,expected_model,expected_optimizer
                completed = step
                history.append(dict(step=step,loss=loss,gradient_norm=norm,learning_rate=rate))
                if step in meta['monitored_full_training_nll_updates']:
                    scores=nll(h[:512],target[:512])
                    monitor.append(dict(update=step,train_zh_nll=float(scores[:256].mean()),
                        train_en_nll=float(scores[256:].mean())))
                    save_json(execution/'monitor.json',monitor)
                if step%1000==0:
                    status('training',loss=loss,allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
                    print(json.dumps(history[-1]),flush=True)
                require(time.perf_counter()-started < meta['maximum_execution_seconds'], 'registered time budget exceeded')
                require(torch.cuda.max_memory_allocated()/1024**3 <= 4., 'registered head memory budget exceeded')
            torch.cuda.synchronize()
            training_seconds=time.perf_counter()-training_started
            final_nll=measure_all()
            atomic_tensor(execution/'final.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                config=model.configuration(),logical_updates=UPDATES))
            atomic_tensor(execution/'nll.pt',dict(initial=initial_nll,final=final_nll))
            save_json(execution/'history.json',history)
            save_json(execution/'monitor.json',monitor)
            status('autonomous_first_blocks')
            generation_started=time.perf_counter()
            generations=[]
            for features in (h,blank):
                for i in range(0,len(features),64):
                    generations.append(unpack_pixels(model.generate(features[i:i+64])).cpu())
                    status('autonomous_first_blocks',completed_blocks=sum(len(v) for v in generations))
            bits=torch.cat(generations).numpy()
            require(bits.shape==(896,32,64),'complete registered block cohort required')
            np.save(execution/'ink.npy',bits,allow_pickle=False)
            images=execution/'images';images.mkdir()
            files={}
            for i,ink in enumerate(bits):
                path=images/f'{i:04}.png'
                Image.fromarray((~ink).astype('uint8')*255).save(path)
                files[str(path.relative_to(execution))]=sha(path)
            parameters=sum(p.numel() for p in model.parameters())
            save_json(execution/'report.json',dict(parameters=parameters,trained_parameters=parameters,
                large_backbone_loaded=False,condition_source='frozen V68 custom image-actor cache',
                original_initialization_bit_exact=True,logical_updates=UPDATES,optimizer_step_calls=UPDATES+1,
                example_draws=UPDATES*BATCH,exact_replay_extra_draws=BATCH,
                initial_nll=float(initial_nll[:512].mean()),final_nll=float(final_nll[:512].mean()),
                training_seconds=training_seconds,generation_seconds=time.perf_counter()-generation_started,
                elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                output_sha256=sha(execution/'ink.npy'),final_checkpoint_sha256=sha(execution/'final.pt'),
                nll_sha256=sha(execution/'nll.pt'),png_sha256=files,numerical_checks=checks,actual_update_replay=replay,
                general_ilm_goal_complete=False,complete_answers_generated=False,stopping_evaluated=False))
            status('completed')
    except BaseException as error:
        status('failed',reason=repr(error))
        raise
    finally:
        state=load(execution/'status.json') if (execution/'status.json').exists() else {}
        save_json(execution/'status.json',dict(state,pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run'),required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--diagnosis',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':
        require(args.source is not None and args.diagnosis is not None,'source and diagnosis required')
        prepare(args.source.resolve(),args.diagnosis.resolve(),args.output.resolve())
    else:
        run(args.output.resolve())
