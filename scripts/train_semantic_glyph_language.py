#!/usr/bin/env python3
"""Train and evaluate next-glyph language generation through actual image feedback."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import diagnostic_gpu,host_resources,require,owner_gone
from scripts.train_semantic_glyph_bridge import atomic_tensor

CORPUS=ROOT/'artifacts/pixel_language_diversity_v1'
CONTROLS=ROOT/'artifacts/native_visual_transfer_v71_20261105/attempt01/structural_vectors.pt'
FILES=('scripts/train_semantic_glyph_language.py','references/semantic_glyph_language_protocol.md',
    'ilm/visual_lm/semantic_glyph_loop.py','ilm/visual_lm/image_context_interface.py',
    'ilm/visual_lm/semantic_image_memory.py','ilm/visual_lm/grounded_image_weights.py',
    'ilm/visual_lm/grounded_image_actor.py','scripts/score_pixel_language_first_blocks.py',
    'scripts/train_semantic_glyph_bridge.py')


def load(path): return json.loads(Path(path).read_text())
def relative(path): return str(Path(path).resolve().relative_to(ROOT))


def identity():
    values={n:sha(ROOT/n) for n in FILES}
    for n,digest in values.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==digest,
                'commit semantic language implementation: '+n)
    return values


def image(path):
    import torch
    return torch.from_numpy(np.asarray(Image.open(path).convert('L')).copy()).float()[None,None]/127.5-1


def prepare(output,primitive):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation requires CUDA hidden')
    require(not output.exists(),'preserve previous language attempts')
    sources=identity()
    meta=load(primitive/'preparation.json');report=load(primitive/'execution/report.json')
    require(load(primitive/'execution/status.json')['phase']=='completed','complete glyph primitive first')
    assessment=load(primitive/f"execution/assessment_{report['step']:05}.json")
    for route in ('semantic','image_roundtrip'):
        rows=[r for r in assessment['records'] if r['route']==route]
        require(sum(r['f1'] for r in rows)/len(rows)>.999 and min(r['f1'] for r in rows)>.95,
                'glyph reconstruction does not pass the declared primitive gate')
    require(sha(primitive/'execution/checkpoint_latest.pt')==report['checkpoint_sha256'],'primitive state changed')
    charmap={c:i for i,row in enumerate(load(primitive/'inventory.json')['records']) for c in row['chars']}
    source=ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    records={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    corpus=load(CORPUS/'manifest.json');jobs=[];unsupported=[]
    bindings={relative(source):sha(source),relative(CORPUS/'manifest.json'):sha(CORPUS/'manifest.json'),
        relative(CONTROLS):sha(CONTROLS),relative(primitive/'preparation.json'):sha(primitive/'preparation.json'),
        relative(primitive/'training.pt'):meta['training_sha256'],
        relative(primitive/'inventory.json'):meta['inventory_sha256'],
        relative(primitive/'execution/checkpoint_latest.pt'):report['checkpoint_sha256']}
    for row in corpus['train']:
        raw=records[Path(row['source']).stem]
        answer=raw['references'][0]['text']
        missing=sorted(set(answer)-charmap.keys())
        if missing:
            unsupported.append(dict(index=row['index'],source=row['source'],missing=missing))
            continue
        path=CORPUS/row['prompt']
        require(sha(path)==row['prompt_sha256'],'training prompt changed')
        bindings[relative(path)]=row['prompt_sha256']
        jobs.append(dict(index=row['index'],source=row['source'],prompt=relative(path),
            labels=[charmap[c] for c in answer],answer_sha256=hashlib.sha256(answer.encode()).hexdigest()))
    output.mkdir(parents=True)
    save_json(output/'preparation.json',dict(code_sha256=sources,bindings_sha256=bindings,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        primitive=relative(primitive),jobs=jobs,unsupported_training=unsupported,
        original_training_cases=len(corpus['train']),training_cases=len(jobs),
        development_training_allowed=False,external_teacher_at_runtime=False,
        seed=202611102,fit_steps=6000,batch_size=128,learning_rate=.0002,
        cache_maximum_seconds=3600.,fit_maximum_seconds=1200.,assessment_maximum_seconds=1200.,
        maximum_allocated_vram_gib=22.,maximum_pressure_wait_seconds=300.,
        maximum_generation_glyphs=64,minimum_runtime_ram_gib=24.,
        initial_state='unmodified original assistant state in FP32',
        cache_targets_enter_only_after_the_state_predicting_them=True))
    print(json.dumps(dict(prepared=True,training_cases=len(jobs),unsupported_training=len(unsupported),
                         glyphs=sum(len(j['labels']) for j in jobs))))


class Monitor:
    def __init__(self,path,seconds):
        self.path=path;self.seconds=seconds;self.began=time.perf_counter();self.waited=0.;self.samples=[]
        self.position={};self.admission=None
    def status(self,phase,**extra):
        save_json(self.path/'status.json',dict(phase=phase,pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            elapsed_seconds=time.perf_counter()-self.began,position=self.position,**extra))
    def guard(self):
        import torch
        require(time.perf_counter()-self.began<self.seconds,'semantic language stage time cap reached')
        require(torch.cuda.max_memory_allocated()/1024**3<=22.,'semantic language allocation cap reached')
        while True:
            value=host_resources();self.samples.append(value)
            require(value['available_gib']>=24.,'semantic language RAM reserve reached')
            if value['some_pressure_avg10']<=.5 and value['full_pressure_avg10']<=.1: break
            remaining=300.-self.waited
            require(remaining>0.,'semantic language pressure wait exhausted')
            self.status('waiting_for_memory_pressure',resources=value)
            start=time.perf_counter();time.sleep(min(10.,remaining));self.waited+=time.perf_counter()-start
    def receipt(self):
        import torch
        return dict(elapsed_seconds=time.perf_counter()-self.began,resource_wait_seconds=self.waited,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            minimum_available_gib=min(v['available_gib'] for v in self.samples),
            admission=self.admission)


def check(output):
    meta=load(output/'preparation.json')
    require(identity()==meta['code_sha256'],'language sources changed')
    for n,digest in meta['bindings_sha256'].items(): require(sha(ROOT/n)==digest,'bound input changed: '+n)
    return meta


def bridge_model(meta,device,checkpoint=None):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
    path=checkpoint or ROOT/meta['primitive']/'execution/checkpoint_latest.pt'
    bridge=SemanticGlyphBridge().to(device)
    bridge.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model'])
    return bridge.requires_grad_(False).eval()


def reader(device):
    import torch
    from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor
    from ilm.visual_lm.image_context_interface import ImageContextInterface
    local=ROOT.parent/'LocalLLM/.local'
    actor,provenance=load_grounded_image_actor(ollama_root=local/'models/ollama',
        gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
    actor.writer=torch.nn.Identity();actor.requires_grad_(False).eval()
    require(not any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith('.up')),
            'source contextual adapters changed')
    interface=ImageContextInterface(actor,**torch.load(CONTROLS,weights_only=True,map_location='cpu'))
    return interface,provenance


def numerics():
    import torch
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','set deterministic CUBLAS workspace')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def cache(output,meta,monitor):
    import torch
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    with diagnostic_gpu() as (device,admission),torch.no_grad():
        monitor.admission=admission;numerics();monitor.guard()
        bridge=bridge_model(meta,device);interface,provenance=reader(device);monitor.guard()
        data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
        all_images=(1-2*data['ink'].float()).to(device)
        embeddings=torch.cat([bridge.feedback_embedding(all_images[i:i+128])
                              for i in range(0,len(all_images),128)])
        file_records=[];causal_error=None
        for position,job in enumerate(meta['jobs']):
            monitor.position=dict(completed=position,total=len(meta['jobs']),index=job['index']);monitor.guard()
            arguments,geometry=interface.assemble(native_budget_image(image(ROOT/job['prompt'])))
            response=interface.actor.context(**arguments,use_cache=True)
            first=response.last_hidden_state[:,-1:].clone()
            past=response.past_key_values
            if position==0: independent=copy.deepcopy(past)
            prefix_length=past.get_seq_length();labels=torch.tensor(job['labels'],device=device)
            values=embeddings[labels][None]
            n=len(labels);offset=geometry['rotary_offset']
            physical=torch.arange(prefix_length,prefix_length+n,device=device)
            rotary=torch.arange(offset,offset+n,device=device)
            positions=torch.stack((physical,rotary,rotary,rotary))[:,None]
            del response,arguments
            response=interface.actor.context(inputs_embeds=values,position_ids=positions,
                                             past_key_values=past,use_cache=True)
            states=torch.cat((first,response.last_hidden_state),1)[0].cpu().clone()
            del response,past
            if position==0:
                # Same gold prefix processed one glyph at a time through the
                # deployment API. Future target images must not affect states.
                errors=[]
                for i in range(min(n,3)):
                    pos=torch.tensor([independent.get_seq_length(),offset+i,offset+i,offset+i],device=device).reshape(4,1,1)
                    probe=interface.actor.context(inputs_embeds=values[:,i:i+1],position_ids=pos,
                        past_key_values=independent,use_cache=True)
                    errors.append(float((probe.last_hidden_state[0,-1].cpu()-states[i+1]).abs().max()))
                    independent=probe.past_key_values
                    del probe
                causal_error=max(errors)
                require(causal_error<.002,'batched causal cache differs from sequential deployment')
                del independent
            name=f"case_{job['index']:04}.pt"
            value=dict(states=states,labels=torch.tensor(job['labels']+[-1]),
                prompt_sha256=meta['bindings_sha256'][job['prompt']],
                first_native_state_sha256=tensor_hash(first),geometry=geometry,
                feedback_from_gold_images=True,current_target_not_input_before_prediction=True,
                primitive_checkpoint_sha256=meta['bindings_sha256'][meta['primitive']+'/execution/checkpoint_latest.pt'])
            atomic_tensor(monitor.path/name,value)
            file_records.append(dict(index=job['index'],file=name,sha256=sha(monitor.path/name),states=len(states)))
            del first,values,states
            monitor.status('caching_gold_image_histories')
            if (position+1)%16==0: print(json.dumps(monitor.position),flush=True)
        save_json(monitor.path/'report.json',dict(files=file_records,provenance=provenance,
            sequential_cache_maximum_error=causal_error,optimizer_updates=0,
            vocabulary_parameters_materialized=0,development_training_allowed=False,**monitor.receipt()))


def fit(output,meta,monitor):
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    from scripts.train_semantic_glyph_bridge import bridge_gpu
    source=load(output/'cache/report.json')
    require(load(output/'cache/status.json')['phase']=='completed','complete and preserve causal cache first')
    # This stage trains only the small bridge; no pretrained body is loaded.
    limits=dict(minimum_entry_ram_gib=64.)
    with bridge_gpu(limits) as (device,admission):
        monitor.admission=admission
        torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        bridge=bridge_model(meta,device)
        bridge.readout.requires_grad_(True);bridge.stop.requires_grad_(True)
        parts=[];labels=[]
        for row in source['files']:
            path=output/'cache'/row['file'];require(sha(path)==row['sha256'],'cached states changed')
            value=torch.load(path,weights_only=True,map_location='cpu')
            require(value['states'].shape==(len(value['labels']),4096),'causal state count differs')
            parts.append(value['states']);labels.append(value['labels'])
        states=torch.cat(parts).to(device);labels=torch.cat(labels).to(device)
        data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
        images=(1-2*data['ink'].float()).to(device)
        targets=((data['semantic']-data['mean'])/data['scale']).to(device)
        params=[p for p in bridge.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=meta['learning_rate'],weight_decay=.01)
        for step in range(1,meta['fit_steps']+1):
            monitor.position=dict(step=step,total=meta['fit_steps'])
            if step%20==1:monitor.guard()
            indexes=torch.randint(len(states),(meta['batch_size'],),device=device)
            h=states[indexes];label=labels[indexes];keep=label>=0;end=(~keep).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                predicted=bridge.readout(h.float()).float();stop=bridge.stop(h.float()).flatten()
                code_loss=F.mse_loss(predicted[keep],targets[label[keep]])
                cosine_loss=1-F.cosine_similarity(predicted[keep],targets[label[keep]]).mean()
                logits=bridge.decode_logits(predicted[keep])
                pixel_loss=reconstruction_loss(logits,images[label[keep]])
                stop_loss=F.binary_cross_entropy_with_logits(stop.float(),end)
                loss=code_loss+.2*cosine_loss+pixel_loss+stop_loss
            require(bool(torch.isfinite(loss)),'nonfinite semantic language loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
            fraction=step/meta['fit_steps']
            for group in optimizer.param_groups:
                group['lr']=meta['learning_rate']*(.1+.9*.5*(1+__import__('math').cos(__import__('math').pi*fraction)))
            if step%100==0:
                row=dict(step=step,loss=float(loss),code_loss=float(code_loss),pixel_loss=float(pixel_loss),
                         stop_loss=float(stop_loss),seconds=time.perf_counter()-monitor.began)
                with (monitor.path/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                monitor.status('fitting_readout');print(json.dumps(row),flush=True)
        checkpoint=monitor.path/'checkpoint_latest.pt'
        atomic_tensor(checkpoint,dict(model={k:v.detach().cpu() for k,v in bridge.state_dict().items()},
            step=meta['fit_steps'],language_training_complete=True,
            preparation_sha256=sha(output/'preparation.json'),primitive_only=False,
            usable_language_demonstrated=False,optimizer=optimizer.state_dict()))
        save_json(monitor.path/'report.json',dict(steps=meta['fit_steps'],states=len(states),
            training_cases=len(source['files']),checkpoint_sha256=sha(checkpoint),
            trained_parameters=sum(p.numel() for p in params),usable_language_demonstrated=False,
            fit_finished_is_not_language_validation=True,**monitor.receipt()))


def assess(output,meta,monitor):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphLanguageActor
    with diagnostic_gpu() as (device,admission),torch.no_grad():
        monitor.admission=admission;numerics();monitor.guard()
        checkpoint=output/'fit/checkpoint_latest.pt'
        require(load(output/'fit/status.json')['phase']=='completed','finish fitting first')
        require(sha(checkpoint)==load(output/'fit/report.json')['checkpoint_sha256'],'language state changed')
        bridge=bridge_model(meta,device,checkpoint);interface,provenance=reader(device)
        actor=SemanticGlyphLanguageActor(interface,bridge,language_training_complete=True).eval()
        hook=interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
        corpus=load(CORPUS/'manifest.json');records=[]
        # Training controls first, then every Chinese development input and its
        # erased counterpart. No output is selected by appearance.
        cases=[('train',meta['jobs'][i]['index'],'normal') for i in range(8)]
        cases += [('development',i,route) for i in range(16) for route in ('normal','blank')]
        for position,(split,index,route) in enumerate(cases):
            monitor.position=dict(completed=position,total=len(cases),split=split,index=index,route=route)
            row=corpus['train' if split=='train' else 'development'][index]
            name=row['prompt'] if split=='train' else row[route]
            digest=row['prompt_sha256'] if split=='train' else row[route+'_sha256']
            require(sha(CORPUS/name)==digest,'original assessment image changed')
            value=actor.generate(image(CORPUS/name),maximum_glyphs=meta['maximum_generation_glyphs'])
            stem=f'{split}_{index:04}_{route}'
            atomic_tensor(monitor.path/(stem+'.pt'),value)
            # Display only the actual emitted glyphs, with no inferred text or font.
            tiles=value['images'];width=max(1,len(tiles))*32
            array=np.full((32,width),255,dtype=np.uint8)
            for i,tile in enumerate(tiles):array[:,i*32:i*32+32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
            Image.fromarray(array).save(monitor.path/(stem+'.png'))
            records.append(dict(split=split,index=index,route=route,glyphs=len(tiles),
                stopped=value['stopped'],capped=value['capped'],
                file=stem+'.pt',sha256=sha(monitor.path/(stem+'.pt')),
                png=stem+'.png',png_sha256=sha(monitor.path/(stem+'.png'))))
            monitor.status('generating_image_answers');print(json.dumps(records[-1]),flush=True)
        hook.remove()
        save_json(monitor.path/'report.json',dict(records=records,provenance=provenance,
            teacher_service=False,runtime_tokenizer=False,runtime_font=False,runtime_glyph_table=False,
            actual_emitted_pixels_reencoded=True,development_sealed=False,
            semantic_accuracy_not_yet_reviewed=True,**monitor.receipt()))


def run(output,stage):
    meta=check(output);owner_gone(ROOT/'artifacts/pixel_language_diversity_training_v2')
    execution=output/stage;execution.mkdir(exist_ok=False)
    monitor=Monitor(execution,meta[stage+'_maximum_seconds']);monitor.status('admitting')
    try:
        {'cache':cache,'fit':fit,'assessment':assess}[stage](output,meta,monitor)
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error));raise
    finally:
        if (execution/'status.json').exists():save_json(execution/'status.json',dict(load(execution/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','cache','fit','assessment'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--primitive',type=Path,default=ROOT/'artifacts/semantic_glyph_bridge_v1')
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.output.resolve(),args.primitive.resolve())
    else: run(args.output.resolve(),args.stage)
