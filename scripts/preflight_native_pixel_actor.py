#!/usr/bin/env python3
"""Execute and differentiate the real full-native PNG-to-pixel candidate."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from scripts.train_conditional_pixel_density import atomic_tensor, equal_tree, numerical_checks, require

NATIVE=ROOT/'artifacts/native_chinese_v80_20261106'
RASTER=ROOT/'artifacts/pixel_density_long_fit_v1'
CASES=ROOT/'artifacts/conditional_readout_v69_20261105/attempt01'
TRAIN=ROOT/'data/general_instructions/complete_lines_train_20261105'
AUDIT=ROOT/'data/general_instructions/codec_language_audit_v68_20261105'
CONTROLS=ROOT/'artifacts/native_visual_transfer_v71_20261105/attempt01/structural_vectors.pt'
CODEC=ROOT/'artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt'
SEED=202611092
FILES=('scripts/preflight_native_pixel_actor.py','ilm/visual_lm/native_pixel_language.py',
    'scripts/audit_native_pixel_actor_preflight.py',
    'references/native_pixel_actor_preflight_protocol.md','ilm/visual_lm/conditional_pixel_density.py',
    'ilm/visual_lm/adaptive_image_continuation.py','ilm/visual_lm/native_image_continuation.py',
    'ilm/visual_lm/native_image_actor.py','ilm/visual_lm/image_context_interface.py',
    'ilm/visual_lm/semantic_image_memory.py','ilm/visual_lm/grounded_image_actor.py',
    'ilm/visual_lm/grounded_image_weights.py','ilm/visual_lm/codec_block_writer.py',
    'ilm/visual_lm/continuous_glyph_codec.py','ilm/visual_lm/gguf_visual_language.py',
    'ilm/visual_lm/gguf_language_trunk.py','ilm/visual_lm/semantic_image_actor.py',
    'ilm/visual_lm/clean_writing_flow.py','ilm/visual_lm/semantic_writing_model.py',
    'ilm/visual_lm/suffix_attention_mask.py','scripts/train_conditional_pixel_density.py')


def load(path):
    return json.loads(Path(path).read_text())


def identity():
    code={name:sha(ROOT/name) for name in FILES}
    for name,digest in code.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()==digest,
            'commit preflight source first: '+name)
    return code


def read_png(path):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as im:pixels=np.asarray(im.convert('L')).copy()
    require(bool(((pixels==0)|(pixels==255)).all()),'owning binary prompt PNG required')
    return torch.from_numpy(pixels).float().div(127.5).sub(1)[None,None].clone()


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA during preparation')
    code=identity()
    require(sha(NATIVE/'training/checkpoint_latest.pt')=='2e267b023118552e2462971c5f8e0dc49b57f4e401ef7ba58082c05fc85fc0f5',
        'completed native checkpoint changed')
    require(sha(RASTER/'execution/final.pt')=='4f91b4129d9eb0129f15dc012bce145ec4c84f3d0c7f386f135d7eedb64d0e02',
        'qualified physical decoder checkpoint changed')
    import numpy as np
    import torch
    from PIL import Image
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
    from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
    torch.set_num_threads(4)
    prior=load(CASES/'preparation.json')
    require(sha(CASES/'prepared.pt')==prior['prepared_sha256'],'owning source prompt cache changed')
    old=torch.load(CASES/'prepared.pt',map_location='cpu',weights_only=True)
    train=CompleteWritingImageData(TRAIN,purpose='train',expected_report_sha256='0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e')
    development=CodecLanguageAuditData(AUDIT)
    rows=[];bound={}
    def bind(path):bound[str(path.relative_to(ROOT))]=sha(path)
    for p in (NATIVE/'training/checkpoint_latest.pt',NATIVE/'training/training_report.json',
            NATIVE/'verification.json',RASTER/'execution/final.pt',RASTER/'execution/report.json',
            CASES/'preparation.json',CASES/'prepared.pt',TRAIN/'report.json',AUDIT/'report.json',CONTROLS,CODEC):bind(p)
    for index in (0,262,512,513,640):
        row=old[index]
        if index<512:
            j=train.names.index(row['source_file'])
            full=train.target(j)[...,:-128].clone()
            prompt=fold_writing_prompt(train.prompt(j)).pixels
            require(torch.equal(prompt,row['prompt']),'original training prompt differs')
            bind(TRAIN/row['source_file'])
        else:
            payload=development.payload(index-512)
            require(development.cases[index-512]['id']==row['id'],'development source order differs')
            full=payload['references'][0].float().div(127.5).sub(1)[None,None].clone()
            bind(AUDIT/row['source_file'])
        require(torch.equal(full[...,:64],row['reference']),'complete reference first block differs')
        blocks=full.shape[-1]//64
        require(2<=blocks<=8,'registered complete-answer length differs')
        images=torch.cat([full[...,j*64:(j+1)*64].clone() for j in range(blocks)])
        rows.append(dict(index=index,id=row['id'],source=row['source'],split=row['split'],
            prompt=row['prompt'].clone(),images=images,blocks=blocks))
    require(rows[0]['split']==rows[1]['split']=='train' and all(r['split']=='development' for r in rows[2:]),
        'only the first two cases may train')
    output.mkdir(parents=True,exist_ok=False);(output/'inputs').mkdir();(output/'references').mkdir()
    cases=[]
    for i,row in enumerate(rows):
        prompt=(row['prompt'][0,0].numpy()>0).astype('uint8')*255
        image=output/'inputs'/f'{i:02}.png';Image.fromarray(prompt).save(image)
        require(torch.equal(read_png(image),row['prompt']),'PNG decode changes original prompt pixels')
        target=np.concatenate([(v[0].numpy()>0).astype('uint8')*255 for v in row['images']],axis=1)
        ref=output/'references'/f'{i:02}.png';Image.fromarray(target).save(ref)
        cases.append({k:row[k] for k in ('index','id','source','split','blocks')} | dict(
            input=str(image.relative_to(output)),input_sha256=sha(image),
            reference=str(ref.relative_to(output)),reference_sha256=sha(ref)))
    atomic_tensor(output/'data.pt',rows)
    save_json(output/'cases.json',cases)
    save_json(output/'preparation.json',dict(code_sha256=code,bound_sha256=bound,
        data_sha256=sha(output/'data.pt'),cases_sha256=sha(output/'cases.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        cases=5,training_cases=2,development_cases=3,seed=SEED,logical_updates=2,
        actual_optimizer_calls=3,maximum_blocks=16,maximum_seconds=1200,maximum_allocated_vram_gib=22.,
        native_prompt_KV_retained=True,condition_width=4096,condition_projection_reset=True,
        training_reference_truncation=False,packages={n:version(n) for n in ('torch','torchvision','transformers','numpy')},
        general_ilm_goal_complete=False))
    print(json.dumps(dict(prepared=str(output),cases=cases),indent=2),flush=True)


def unchanged(output):
    meta=load(output/'preparation.json')
    require(identity()==meta['code_sha256'],'preflight implementation changed')
    for name,digest in meta['bound_sha256'].items():require(sha(ROOT/name)==digest,'bound source changed: '+name)
    require(sha(output/'data.pt')==meta['data_sha256'] and sha(output/'cases.json')==meta['cases_sha256'],
        'prepared complete-answer data changed')
    for row in load(output/'cases.json'):
        require(sha(output/row['input'])==row['input_sha256'] and sha(output/row['reference'])==row['reference_sha256'],
            'physical input/reference PNG changed')
    return meta


def prefix_identity(prefix):
    from ilm.visual_lm.clean_image_actor import tensor_digest
    return dict(length=prefix.length,initial_state=tensor_digest(prefix.initial_state),
        rotary_offsets=tensor_digest(prefix.rotary_offsets),blocked_before=prefix.blocked_before,
        key_values=[[tensor_digest(k),tensor_digest(v)] for k,v in prefix.key_values])


def save_output(folder,result):
    import numpy as np
    from PIL import Image
    folder.mkdir(parents=True,exist_ok=False)
    atomic_tensor(folder/'result.pt',result)
    images=result['images']
    empty=len(images)==0
    pixels=np.concatenate([(v[0].numpy()>0).astype('uint8')*255 for v in images],axis=1) if not empty else np.full((32,64),255,dtype='uint8')
    Image.fromarray(pixels).save(folder/'answer.png')
    return dict(blocks=len(images),empty_output_display_only=empty,
        stopped=result['stopped'],capped=result['capped'],result_sha256=sha(folder/'result.pt'),
        png_sha256=sha(folder/'answer.png'))


def run(output):
    require(os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID','explicit physical GPU ordering required')
    meta=unchanged(output);execution=output/'execution';execution.mkdir(exist_ok=False)
    started=time.perf_counter();completed=0
    def status(phase,**extra):
        save_json(execution/'status.json',dict(phase=phase,pid=os.getpid(),logical_updates=completed,
            elapsed_seconds=time.perf_counter()-started,**extra))
    def deadline(signum,frame):raise TimeoutError('registered1200-second preflight deadline')
    old_handler=signal.signal(signal.SIGALRM,deadline);signal.alarm(1200)
    try:
        with gpu_zero() as device:
            import gc
            import torch
            from torch.nn import functional as F
            from ilm.visual_lm.adaptive_image_continuation import adapter_parameters
            from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
            from ilm.visual_lm.native_pixel_language import NativePixelLanguageModel
            from ilm.visual_lm.conditional_pixel_density import pack_pixels
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            rows=torch.load(output/'data.pt',map_location='cpu',weights_only=True)
            status('loading_complete_native_actor')
            local=ROOT.parent/'LocalLLM/.local'
            actor,provenance=load_trained_native_image_actor(checkpoint=NATIVE/'training/checkpoint_latest.pt',
                controls=CONTROLS,codec=CODEC,ollama_root=local/'models/ollama',
                gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
            old=actor.writing
            warm={k:v.detach().cpu().clone() for k,v in old.state_dict().items()
                if k.startswith(('image_input.','stop.')) or k=='input_scale'}
            raster=torch.load(RASTER/'execution/final.pt',map_location='cpu',weights_only=True)['model']
            model=NativePixelLanguageModel(old.continuation.context,old.writer,
                input_scale=float(old.input_scale),query_seed=SEED+1,raster_state=raster).to(device)
            model.warm_image_interfaces(warm);actor.writing=model
            del old,raster,warm;gc.collect();torch.cuda.empty_cache();actor.eval()
            require(not hasattr(model,'flow'),'the image-code prediction flow must be absent')
            parameters=[p for p in model.parameters() if p.requires_grad]
            counts=dict(complete_parameters=sum(p.numel() for p in actor.parameters()),
                trained_parameters=sum(p.numel() for p in parameters),
                density_parameters=sum(p.numel() for p in model.density.parameters()),
                suffix_adapter_parameters=sum(p.numel() for _,p in adapter_parameters(model.continuation.context)),
                transferred_raster_tensors=model.transferred_raster_tensors)
            status('native_PNG_prefill',**counts)
            prefixes=[];prefix_hashes=[]
            for i,row in enumerate(rows):
                prefix,geometry=actor.prefill(read_png(output/f'inputs/{i:02}.png'))
                require(geometry['all_prompt_keys_retained'] and prefix.blocked_before==0,'full native keys required')
                prefix_hashes.append(prefix_identity(prefix));prefixes.append(prefix.to('cpu'))
                del prefix
            atomic_tensor(execution/'prefixes.pt',[dict(key_values=p.key_values,initial_state=p.initial_state,
                rotary_offsets=p.rotary_offsets,blocked_before=p.blocked_before,valid_lengths=p.valid_lengths)
                for p in prefixes[:2]])
            encoded=[model.image_codes(row['images']).cpu() for row in rows[:2]]
            # Prohibit any accidental use of the old codec image decoder.
            def no_decoder(*args,**kwargs):raise RuntimeError('old image-code decoder entered the output path')
            model.writer.predict=no_decoder;model.writer.codec.decode=no_decoder
            collector=None
            actual_encode=model.writer.encode_targets
            def tracked_encode(images):
                codes=actual_encode(images)
                if collector is not None:
                    collector.append(dict(images=images.detach().cpu().clone(),codes=codes.detach().cpu().clone()))
                return codes
            model.writer.encode_targets=tracked_encode
            checks={};evidence={}
            def causal_check(label):
                prefix=prefixes[0].to(device);codes=encoded[0].to(device)
                changed=codes.clone();images=rows[0]['images'].clone();images[1].neg_()
                changed[1]=model.image_codes(images[1:2])[0]
                with torch.no_grad():
                    full=model.conditions(prefix,codes[None],checkpoint_layers=False)
                    altered=model.conditions(prefix,changed[None],checkpoint_layers=False)
                    error=float((full[:,:2]-altered[:,:2]).abs().max())
                require(error==0.,'future-image query leakage')
                checks[label+'_future_image_exclusion_exact']=True
                return prefix,codes,full
            prefix,codes,full=causal_check('before')
            with torch.no_grad():
                start=model.start_prefix(prefix)
                actual=start.initial_state[:,0].float();expected=full[:,0].float()
                absolute=float((actual-expected).abs().max())
                relative=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
                require(relative<=1e-4,'native incremental/full-history normwise comparison failed')
                evidence['start_state_comparison']=dict(max_absolute=absolute,normwise_relative=relative,
                    emitted_pixels_exact=bool(torch.equal(model.emit_image(actual),model.emit_image(expected))))
                target=pack_pixels(rows[0]['images'][:2,0].to(device)<0)
                numerical=numerical_checks(model.density,model.normalized_conditions(full[0,:2]),target)
                checks['pixel_causality_and_cache']=True
                collector=[]
                first=model.emit_image(actual)
                following,feedback=model.append_emitted(start,first)
                second=model.emit_image(following.initial_state[:,0])
                require(following.length==start.length+2,'actual emitted feedback did not append two image positions')
                atomic_tensor(execution/'forced_feedback.pt',dict(images=torch.cat((first,second)).cpu(),
                    first_feedback_sha256=feedback,records=collector,
                    source='separately forced path check; not an autonomous answer'))
                collector=None
                checks['actual_generated_image_feedback']=True
            del prefix,codes,full,start,following,first,second
            fixed={n:(id(p),p._version) for n,p in actor.named_parameters() if not p.requires_grad}
            optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
            history=[]
            def update(i):
                actor.train();optimizer.zero_grad(set_to_none=True)
                loss,parts=model.loss_from_codes(prefixes[i].to(device),rows[i]['images'].to(device),encoded[i].to(device),writer_blocks=16)
                require(bool(torch.isfinite(loss)),'nonfinite actual native pixel loss')
                loss.backward()
                gradient={key:0. for key in ('density','image_input','start_query','suffix','stop')}
                for name,p in model.named_parameters():
                    if p.grad is None:continue
                    require(bool(torch.isfinite(p.grad).all()),'nonfinite gradient: '+name)
                    key=('suffix' if name.startswith('continuation.context.') else name.split('.')[0])
                    if key in gradient:gradient[key]+=float(p.grad.float().square().sum())
                gradient={k:v**.5 for k,v in gradient.items()}
                require(all(v>0 for v in gradient.values()),'a required native image learning path has zero gradient')
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                require(bool(torch.isfinite(norm)),'nonfinite clipped gradient norm')
                optimizer.step()
                require(all((id(p),p._version)==fixed[n] for n,p in actor.named_parameters() if n in fixed),
                    'a frozen source parameter changed')
                return dict(loss=float(loss),image_nll=float(parts['image_nll']),stop=float(parts['stop']),
                    gradient_norm=float(norm),gradient_components=gradient)
            status('actual_optimizer_updates')
            history.append(update(0));completed=1
            atomic_tensor(execution/'update0001.pt',dict(model=model.trainable_state_dict(),optimizer=optimizer.state_dict()))
            restored=torch.load(execution/'update0001.pt',map_location='cpu',weights_only=True)
            history.append(update(1));completed=2
            expected_model=model.trainable_state_dict()
            # CPU copies preserve optimizer step-counter device on restoration.
            expected_optimizer={}
            def cpu_tree(value):
                if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
                if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
                return value
            expected_optimizer=cpu_tree(optimizer.state_dict())
            model.load_trainable_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
            replay=update(1)
            require(replay==history[1] and equal_tree(expected_model,model.trainable_state_dict())
                and equal_tree(expected_optimizer,cpu_tree(optimizer.state_dict())),'actual native update replay differs')
            checks['actual_update_two_replay_bit_exact']=True
            del restored,expected_model,expected_optimizer
            actor.eval();causal_check('after')
            checks['frozen_source_parameters_unchanged']=True
            atomic_tensor(execution/'final.pt',dict(model=model.trainable_state_dict(),optimizer=cpu_tree(optimizer.state_dict()),logical_updates=2))
            save_json(execution/'updates.json',history)
            for i in range(5):
                prefix,_=actor.prefill(read_png(output/f'inputs/{i:02}.png'))
                require(prefix_identity(prefix)==prefix_hashes[i],'suffix learning changed source native prefill')
                del prefix
            checks['all_five_native_prefills_bit_exact_after_learning']=True
            status('actual_PNG_to_answer_generation')
            outputs=[];first_result=None
            for i in range(5):
                # Only owning prompt pixels enter the deployed actor.
                collector=[]
                result=actor.generate(read_png(output/f'inputs/{i:02}.png'),maximum_blocks=16)
                outputs.append(save_output(execution/f'answer_{i:02}',result))
                atomic_tensor(execution/f'answer_{i:02}/feedback.pt',collector);collector=None
                if i==0:first_result=result
                require(time.perf_counter()-started<meta['maximum_seconds'],'registered preflight time budget exceeded')
                require(torch.cuda.max_memory_allocated()/1024**3<=22.,'registered full-model VRAM budget exceeded')
                status('actual_PNG_to_answer_generation',completed_cases=i+1)
            collector=[]
            repeated=actor.generate(read_png(output/'inputs/00.png'),maximum_blocks=16)
            require(equal_tree(first_result,repeated),'full autonomous native PNG answer replay differs')
            outputs.append(save_output(execution/'answer_00_replay',repeated))
            atomic_tensor(execution/'answer_00_replay/feedback.pt',collector);collector=None
            checks['autonomous_answer_replay_exact']=True
            save_json(execution/'report.json',dict(code_sha256=meta['code_sha256'],counts=counts,checks=checks,
                prefix_identities=prefix_hashes,evidence=evidence,pixel_numerical_checks=numerical,
                outputs=outputs,logical_optimizer_updates=2,actual_optimizer_step_calls=3,
                final_checkpoint_sha256=sha(execution/'final.pt'),
                source=provenance,codec_decoder_calls=0,actual_complete_PNG_generations=6,
                forced_two_block_control_separate=True,elapsed_seconds=time.perf_counter()-started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                initialization_is_composite=True,language_training_qualified=False,general_ilm_goal_complete=False))
            status('completed')
    except BaseException as error:
        status('failed',reason=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        state=load(execution/'status.json') if (execution/'status.json').exists() else {}
        save_json(execution/'status.json',dict(state,pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.output.resolve())
