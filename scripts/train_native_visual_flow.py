#!/usr/bin/env python3
"""Complete mixed image-answer training with resumable native flow matching.

prepare is CPU-only. train requires the real codec/gradient preflight to have
completed. Only training images enter optimization. No intermediate checkpoint
is selected by answer scores. Exports omit optimizer and per-example data.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from ilm.image_edit.baseline import ROOT,gpu_zero,save_json,sha
from ilm.image_edit import visual_semantic as vs
from ilm.image_edit.native_flow_run import (CONFIG,DATA,CONDITION,PREFLIGHT,identity,digest,
    schedule,restore_adapters,atomic_checkpoint,checkpoint_state,validate_checkpoint)


def prepare_evaluation(output, all_rows, source):
    """Keep assessment answers separate from the generator's input-only indices."""
    controls=ROOT/'artifacts/paired_language_controls_v1'
    control_prep=json.loads((controls/'preparation.json').read_text())
    original=ROOT/'artifacts/image_edit_learning/pages_v2/pairs.jsonl'
    if sha(original)!=source['original_manifest_sha256']:
        raise ValueError('original assessment reference manifest changed')
    references={r['id']:r for r in map(json.loads,original.read_text().splitlines())}
    by_id={r['id']:r for r in all_rows}
    folder=output/'evaluation';folder.mkdir()

    def index(name):
        file=controls/(name+'_inputs.json')
        if sha(file)!=control_prep['index_files_sha256'][file.name]:
            raise ValueError('registered control index changed')
        return [{**r,'input':os.path.relpath((controls/r['input']).resolve(),folder)}
            for r in json.loads(file.read_text())]

    cohorts={'normal':index('normal_seen')+index('normal_held'),
        'question_erased':index('question_erased_held'),
        'passage_erased':index('passage_erased_held')}
    copies=([r for r in all_rows if r['task']=='copy' and r['split']=='train'][:32]+
        [r for r in all_rows if r['task']=='copy' and r['split']=='validation'])
    cohorts['copy']=[dict(id=r['id'],input=os.path.relpath((DATA/r['input']).resolve(),folder),
        input_sha256=r['input_sha256'],seed=20269450+i) for i,r in enumerate(copies)]
    if {k:len(v) for k,v in cohorts.items()}!={'normal':64,'question_erased':32,'passage_erased':32,'copy':64}:
        raise ValueError('complete fixed 192-case assessment required')
    files={}
    for name,items in cohorts.items():
        if len({r['id'] for r in items})!=len(items):raise ValueError('duplicate assessment ID')
        refs=[]
        for r in items:
            if set(r)!={'id','input','input_sha256','seed'} or sha(folder/r['input'])!=r['input_sha256']:
                raise ValueError('input-only assessment pixels changed')
            row=by_id[r['id']];original_row=references[r['id']]
            if row['original_target_sha256']!=original_row['target_sha256']:
                raise ValueError('assessment target is not the original padded answer')
            refs.append({**row,'input':r['input'],'input_sha256':r['input_sha256'],
                'target':os.path.relpath((DATA/row['target']).resolve(),folder),
                'reference_text':original_row['reference_text']})
        input_file=folder/(name+'_inputs.json');save_json(input_file,items)
        reference_file=folder/(name+'_references.jsonl')
        reference_file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs))
        files[input_file.name]=sha(input_file);files[reference_file.name]=sha(reference_file)
    return dict(files_sha256=files,control_preparation_sha256=sha(controls/'preparation.json'),
        original_reference_manifest_sha256=sha(original),calls_per_model=192,
        input_indices_contain_answers=False,reference_manifests_for_offline_evaluation_only=True)


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for data preparation')
    code=identity();source=json.loads((DATA/'report.json').read_text())
    if sha(DATA/'image_pairs.jsonl')!=source['image_manifest_sha256']:raise ValueError('source image manifest changed')
    all_rows=[json.loads(x) for x in (DATA/'image_pairs.jsonl').read_text().splitlines()]
    train=[r for r in all_rows if r['split']=='train'];held=[r for r in all_rows if r['split']=='validation']
    if len(train)!=384 or len(held)!=64 or Counter(r['task'] for r in train)!=Counter(qa=256,copy=128):
        raise ValueError('complete384/64 image corpus required')
    train_articles={r['article'] for r in train if r['task']=='qa'}
    held_articles={r['article'] for r in held if r['task']=='qa'}
    if train_articles&held_articles:raise ValueError('QA article leakage')
    from PIL import Image
    for r in all_rows:
        if 'reference_text' in r:raise ValueError('use the image-only training manifest')
        for key in ('input','target'):
            path=DATA/r[key]
            if sha(path)!=r[key+'_sha256']:raise ValueError('original image bytes changed')
            with Image.open(path) as im:
                if im.size!=(1024,1024) or im.mode!='RGB':raise ValueError('native full RGB pages required')
    plan=schedule(train)
    if [r['update'] for r in plan]!=list(range(1,1537)) or Counter(r['id'] for r in plan)!=Counter({r['id']:4 for r in train}):
        raise ValueError('exact four-permutation schedule required')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for r in train:
        rows.append({**r,**{k:os.path.relpath((DATA/r[k]).resolve(),output) for k in ('input','target')}})
    (output/'training_pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    save_json(output/'schedule.json',plan)
    evaluation=prepare_evaluation(output,all_rows,source)
    prep=json.loads((PREFLIGHT.parent/'preparation.json').read_text())
    metadata=dict(format='native_visual_flow_training_v1',config=CONFIG,code_sha256=code,
        source_revision=vs.REVISION,source_file_manifest_sha256=sha(vs.ACQUISITION/'report.json'),
        data_preparation_sha256=sha(DATA/'report.json'),original_image_manifest_sha256=source['image_manifest_sha256'],
        training_manifest_sha256=sha(output/'training_pairs.jsonl'),schedule_sha256=digest(plan),
        condition_manifest_sha256=sha(CONDITION/'manifest.json'),condition_payload_sha256=sha(CONDITION/'condition.safetensors'),
        adapter_layer_names=prep['adapter_layer_names'],adapter_trainable_parameters=prep['adapter_trainable_parameters'],
        training_pairs=384,heldout_pairs_not_in_optimizer=64,training_QA_articles=len(train_articles),
        heldout_QA_articles=len(held_articles),fresh_source_initialization=True,preflight_checkpoint_restored=False,
        evaluation=evaluation,
        objective_has_lexical_class_targets=False,target_in_image_condition=False,external_teacher_or_OCR=False,
        retained_reader_language_pretraining=True,general_ilm_goal_complete=False)
    if identity()!=code:raise ValueError('source changed during preparation')
    save_json(output/'preparation.json',metadata)
    print(json.dumps(dict(training_pairs=384,heldout_pairs=64,updates=len(plan),schedule_sha256=digest(plan),
        adapter_trainable_parameters=metadata['adapter_trainable_parameters'],model_loaded=False),indent=2))


def export_final(output):
    import torch
    from safetensors.torch import save_file
    meta=json.loads((output/'preparation.json').read_text())
    if identity()!=meta['code_sha256']:raise ValueError('export source changed')
    value=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True)
    plan=json.loads((output/'schedule.json').read_text());step=validate_checkpoint(value,meta,plan)
    if step!=CONFIG['updates']:raise ValueError('only the final fixed-budget checkpoint is exported')
    folder=output/'deployment';folder.mkdir(exist_ok=False)
    save_file({k:v.contiguous() for k,v in value['adapters'].items()},str(folder/'adapters.safetensors'))
    info=dict(format='native_visual_flow_adapter_v1',source_revision=vs.REVISION,
        source_sha256=json.loads((output/'source_verification.json').read_text()),
        condition_manifest_sha256=meta['condition_manifest_sha256'],condition_payload_sha256=meta['condition_payload_sha256'],
        code_sha256=meta['code_sha256'],rank=CONFIG['rank'],adapter_layer_names=meta['adapter_layer_names'],
        adapter_parameters=meta['adapter_trainable_parameters'],completed_updates=step,
        weights_sha256=sha(folder/'adapters.safetensors'),final_checkpoint_sha256=sha(output/'checkpoint.pt'),
        input_canvas=CONFIG['input_canvas'],output_canvas=CONFIG['output_canvas'],steps=CONFIG['inference_steps'],
        guidance=CONFIG['guidance'],runtime_lexical_vocabulary=False,external_teacher_or_OCR=False,
        language_pretraining_retained=True,language_quality_qualified=False,general_ilm_goal_complete=False)
    save_json(folder/'deployment.json',info)
    print('Exported final adapters and fixed source bindings; no optimizer or example data.')


def train(output,resume):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('admit only GPU0')
    meta=json.loads((output/'preparation.json').read_text());code=identity()
    if meta['code_sha256']!=code or meta['config']!=CONFIG:raise ValueError('registered training code/config changed')
    if sha(output/'training_pairs.jsonl')!=meta['training_manifest_sha256']:raise ValueError('training image manifest changed')
    rows=[json.loads(x) for x in (output/'training_pairs.jsonl').read_text().splitlines()]
    if any(r['split']!='train' for r in rows):raise ValueError('validation image in optimizer stream')
    plan=schedule(rows)
    if digest(plan)!=meta['schedule_sha256'] or plan!=json.loads((output/'schedule.json').read_text()):raise ValueError('schedule changed')
    terminal=json.loads((PREFLIGHT/'runtime.json').read_text()) if (PREFLIGHT/'runtime.json').exists() else {}
    proof=json.loads((PREFLIGHT/'report.json').read_text()) if (PREFLIGHT/'report.json').exists() else {}
    if (terminal.get('phase')!='ready_for_raw_review' or proof.get('actual_codec_decodes')!=8 or
        proof.get('actual_autonomous_generations')!=2 or proof.get('learning',{}).get('actual_optimizer_updates')!=1):
        raise ValueError('complete actual native codec/gradient preflight before longer training')
    replay=proof['zero_adapter_replay']
    if not replay['decoded_tensor_exact'] or not replay['png_pixels_exact'] or not all(replay['forward_fields_exact'].values()):
        raise ValueError('actual zero-adapter native replay failed')
    lock=(output/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=output/'checkpoint.pt'
    if (output/'report.json').exists():raise ValueError('preserve completed training; use the final export stage')
    if checkpoint.exists()!=resume:raise ValueError('explicit --resume required exactly when a committed checkpoint exists')
    previous=list(output.glob('attempt_*'));attempt=f'attempt_{len(previous)+1:03}'
    attempt_dir=output/attempt;attempt_dir.mkdir(exist_ok=False)
    phase='verifying_source';step=0;start_step=0;committed_step=0;history=[];started=time.perf_counter()
    def status(**extra):save_json(output/'status.json',dict(phase=phase,pid=None if phase in ('failed','completed') else os.getpid(),
        attempt=attempt,step=step,budget=CONFIG['updates'],general_ilm_goal_complete=False,**extra))
    def unchanged():
        if identity()!=code or sha(output/'training_pairs.jsonl')!=meta['training_manifest_sha256']:
            raise ValueError('bound implementation or training data changed')
    def interrupted(signum,frame):raise InterruptedError('training signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
    try:
        status()
        if sha(vs.ACQUISITION/'report.json')!=meta['source_file_manifest_sha256']:
            raise ValueError('registered release manifest changed')
        source=vs.verify_release()
        save_json(output/'source_verification.json',source)
        for name,key in (('manifest.json','condition_manifest_sha256'),('condition.safetensors','condition_payload_sha256')):
            if sha(CONDITION/name)!=meta[key]:raise ValueError('fixed compiled condition changed')
        with gpu_zero() as device:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler
            from ilm.image_edit.native_flow_step import (add_adapters,trainable,adapter_state,frozen_input_features,encode_target)
            from ilm.image_edit.visual_flow_learning import sample_flow,denoising_loss
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            phase='loading';status();pipe,loader=vs.load_pipeline(device)
            vs.install_compiled(pipe,CONDITION);removal=vs.remove_lexical_modules(pipe)
            torch.manual_seed(CONFIG['initialization_seed']);torch.cuda.manual_seed(CONFIG['initialization_seed'])
            layers=add_adapters(pipe,CONFIG['rank']);selected=trainable(pipe)
            if layers!=meta['adapter_layer_names'] or sum(p.numel() for p in selected.values())!=meta['adapter_trainable_parameters']:
                raise ValueError('actual trainable graph differs from registration')
            optimizer=torch.optim.AdamW(list(selected.values()),lr=CONFIG['learning_rate'],betas=tuple(CONFIG['betas']),
                weight_decay=CONFIG['weight_decay'],foreach=False)
            time_rng=torch.Generator(device='cpu').manual_seed(CONFIG['time_seed'])
            noise_rng=torch.Generator(device=device).manual_seed(CONFIG['noise_seed'])
            if resume:
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                step=validate_checkpoint(state,meta,plan);restore_adapters(pipe,state['adapters'])
                optimizer.load_state_dict(state['optimizer']);history=state['history']
                torch.set_rng_state(state['torch_cpu_rng']);torch.cuda.set_rng_state(state['torch_cuda_rng'],device)
                time_rng.set_state(state['time_rng']);noise_rng.set_state(state['noise_rng']);del state
            else:
                torch.manual_seed(CONFIG['codec_seed']);torch.cuda.manual_seed(CONFIG['codec_seed'])
            start_step=step;committed_step=step
            for part in ('transformer','connector_module'):getattr(pipe,part).train()
            pipe.transformer.enable_gradient_checkpointing();pipe.connector_module.connector.gradient_checkpointing=True
            frozen_versions={part+'.'+n:p._version for part in ('transformer','connector_module','lmm','vae')
                for n,p in getattr(pipe,part).named_parameters() if not p.requires_grad}
            save_json(attempt_dir/'initialization.json',dict(loader=loader,removal=removal,resumed=resume,
                checkpoint_start_step=step,adapter_parameters=meta['adapter_trainable_parameters']))
            training_scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(vs.SNAPSHOT/'scheduler'),local_files_only=True)
            previews=[r for r in rows if r['task']=='qa'][:2]
            by_id={r['id']:r for r in rows};phase='training';status()
            with (attempt_dir/'updates.jsonl').open('x') as log:
                for wanted in plan[step:]:
                    unchanged();row=by_id[wanted['id']];tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    for key in ('input','target'):
                        if sha(output/row[key])!=row[key+'_sha256']:raise ValueError('actual training pixels changed')
                    merged,reference,feature_meta=frozen_input_features(pipe,output/row['input'])
                    target=encode_target(pipe,output/row['target'])
                    pooled,sequence=pipe.connector_module.llm2dit(merged)
                    flow=sample_flow(target,training_scheduler,
                        time_generator=time_rng,noise_generator=noise_rng)
                    loss,prediction=denoising_loss(pipe.transformer,flow,sequence_condition=sequence[:1],
                        pooled_condition=pooled[:1],reference_latents=[[reference]])
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(selected.values()),CONFIG['clip_norm'],error_if_nonfinite=True)
                    optimizer.step();torch.cuda.synchronize();step=wanted['update']
                    record=dict(**wanted,loss=float(loss.detach()),gradient_norm=float(norm),input_sha256=row['input_sha256'],
                        target_sha256=row['target_sha256'],noise_level_index=flow.indices.tolist(),
                        timestep=flow.timesteps.detach().cpu().tolist(),sigma=flow.sigmas.detach().cpu().float().flatten().tolist(),
                        elapsed_seconds=time.perf_counter()-tick,task=row['task'],feature_metadata=feature_meta)
                    history.append(record);log.write(json.dumps(record)+'\n');log.flush()
                    del merged,reference,target,pooled,sequence,flow,prediction,loss
                    if step%CONFIG['preview_every']==0:
                        # These fixed training examples are monitoring, not
                        # validation or a rule for choosing a checkpoint.
                        import random
                        import numpy as np
                        states=(torch.get_rng_state(),torch.cuda.get_rng_state(device),random.getstate(),np.random.get_state())
                        for part in ('transformer','connector_module'):getattr(pipe,part).eval()
                        try:
                            for j,example in enumerate(previews):
                                if sha(output/example['input'])!=example['input_sha256']:raise ValueError('preview input changed')
                                value=vs.generate(pipe,output/example['input'],seed=20269490+j,
                                    width=1024,height=1024,steps=CONFIG['inference_steps'],guidance=CONFIG['guidance'])
                                preview_folder=attempt_dir/'previews'/f'{step:04}_{j:02}'
                                saved=vs.save_generation(value,preview_folder)
                                save_json(preview_folder/'record.json',dict(id=example['id'],after_update=step,
                                    seed=20269490+j,input_sha256=example['input_sha256'],result=saved,
                                    monitoring_only=True,checkpoint_selected_by_quality=False))
                                del value
                        finally:
                            torch.set_rng_state(states[0]);torch.cuda.set_rng_state(states[1],device)
                            random.setstate(states[2]);np.random.set_state(states[3])
                            for part in ('transformer','connector_module'):getattr(pipe,part).train()
                    if torch.cuda.max_memory_allocated()/1024**3>CONFIG['maximum_allocated_vram_gib']:
                        raise RuntimeError('registered allocated-memory limit exceeded; preserve last committed checkpoint')
                    if step%CONFIG['save_every']==0 or step==CONFIG['updates']:
                        versions={part+'.'+n:p._version for part in ('transformer','connector_module','lmm','vae')
                            for n,p in getattr(pipe,part).named_parameters() if not p.requires_grad}
                        if versions!=frozen_versions:raise ValueError('frozen parameter version changed')
                        adapters=adapter_state(pipe)
                        if not all(torch.isfinite(v).all() for v in adapters.values()):raise FloatingPointError('nonfinite adapter checkpoint')
                        value=checkpoint_state(adapters=adapters,optimizer=optimizer,step=step,history=history,metadata=meta,
                            time_rng=time_rng,noise_rng=noise_rng,cuda_rng=torch.cuda.get_rng_state(device))
                        validate_checkpoint(value,meta,plan);atomic_checkpoint(checkpoint,value,attempt);committed_step=step;del value,adapters
                    status(last_committed_step=committed_step)
                    print(json.dumps(record),flush=True)
            report=dict(stage='completed',updates=step,checkpoint_sha256=sha(checkpoint),metadata=meta,
                source=source,all_training_draws=len(history),actual_training_pairs=384,optimizer_restored=resume,
                elapsed_seconds_this_attempt=time.perf_counter()-started,new_updates_this_attempt=step-start_step,
                peak_allocated_vram_gib_this_attempt=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_vram_gib_this_attempt=torch.cuda.max_memory_reserved()/1024**3,
                runtime_parameters=vs.parameters(pipe),actual_trainable_parameters=sum(p.numel() for p in selected.values()),
                frozen_parameter_versions_unchanged=True,autonomous_assessment_completed=False,
                semantic_learning_validated=False,general_ilm_goal_complete=False)
            save_json(output/'report.json',report)
            save_json(attempt_dir/'completed.json',dict(start_step=start_step,end_step=step,
                elapsed_seconds_this_attempt=report['elapsed_seconds_this_attempt'],
                peak_allocated_vram_gib_this_attempt=report['peak_allocated_vram_gib_this_attempt']))
        phase='completed';status()
    except BaseException as error:
        phase='failed';status(exception=type(error).__name__,reason=str(error),committed_checkpoint_present=checkpoint.exists(),
            last_committed_step=committed_step)
        save_json(attempt_dir/'failed.json',dict(start_step=start_step,last_completed_in_memory_step=step,
            last_committed_step=committed_step,elapsed_seconds=time.perf_counter()-started,reason=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','train','export'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    {'prepare':lambda:prepare(args.output.resolve()),'train':lambda:train(args.output.resolve(),args.resume),
        'export':lambda:export_final(args.output.resolve())}[args.stage]()
