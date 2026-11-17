#!/usr/bin/env python3
"""Equal-form codec comparison; preserve the completed natural-frequency controls."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
REG=ROOT/'references/evidence/balanced_codec_comparison_registration_20261118.json'
RUN=ROOT/'artifacts/balanced_codec_comparison_v1'
SOURCE=ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt'
PARTITION=ROOT/'artifacts/binary_codec_preparation_v1/partition.pt'
FILES=('ilm/visual_lm/form_balanced_codec_support.py','tests/test_form_balanced_codec_support.py',
       'references/evidence/form_balanced_codec_cpu_20261118.json',
       'references/evidence/binary_codec_comparison_result_20261118.json',
       'references/evidence/binary_plan_information_20261118.json','references/evidence/binary_codec_frequency_20261118.json',
       'references/form_balanced_codec_direction_20261118.md','scripts/train_balanced_plan_codec_comparison.py','ilm/visual_lm/binary_plan_glyph_density.py',
       'ilm/visual_lm/binary_plan_codec_learning.py','ilm/visual_lm/binary_codec_metrics.py',
       'ilm/visual_lm/continuous_glyph_codec.py','ilm/visual_lm/continuous_glyph_codec_training.py',
       'ilm/visual_lm/occupancy_glyph_density.py','ilm/visual_lm/conditional_pixel_density.py',
       'ilm/visual_lm/full_rank_visual_span.py','ilm/visual_lm/glyph_reader_monitor.py',
       'ilm/visual_lm/visual_context_deployment_resources.py','scripts/measure_reader_writing_consequence.py',
       'scripts/diagnose_visual_context_gradients.py','tests/test_binary_plan_codec_learning.py',
       'tests/test_binary_codec_metrics.py','references/balanced_codec_comparison_protocol_20261118.md',
       'references/evidence/binary_codec_learning_cpu_20261118.json','references/evidence/binary_codec_metrics_cpu_20261118.json',
       'references/evidence/binary_codec_preparation_registration_20261118.json',
       'references/evidence/binary_codec_preparation_result_20261118.json',
       'references/evidence/occupancy_density_structure_result_20261118.json',
       'artifacts/binary_codec_preparation_v1/partition.pt','artifacts/binary_codec_preparation_v1/result.json',
       str(SOURCE.relative_to(ROOT)))


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only registration')
    prepared=original.load(ROOT/'artifacts/binary_codec_preparation_v1/result.json')
    original.require(prepared['all_source_cells_retained'] and prepared['exact_physical_aliases_in_one_partition'],'Physical split must pass')
    original.require(original.sha(PARTITION)==prepared['partition_sha256'] and original.sha(SOURCE)==prepared['source_sha256'],'Prepared sources differ')
    original.write_new(REG,dict(schema='ilm-balanced-codec-comparison-v1',input_sha256={n:original.sha(ROOT/n) for n in FILES},
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy']},modes=['continuous','binary'],
        seed=202611195,order_seed=202611196,code_bits=64,channels=[32,64,96,128],groups=8,
        updates_per_arm=6000,batch_images=128,evaluation_batch_images=256,learning_rate=.0002,minimum_learning_rate=.00002,
        warmup_updates=250,adam_betas=[.9,.95],weight_decay=.01,gradient_clip=1.,replay_update=3,
        source_cells=68948,train_cells=prepared['train_cells'],held_cells=prepared['held_cells'],
        preflight='Every complete training image, weighted to equal form mass; one update, measure, exact reset',
        sampling='One first-source representative per exact physical form, fixed permutation cycled; same 768000 draws per arm',
        unchanged_control='references/evidence/binary_codec_comparison_result_20261118.json',
        quality=dict(train_f1=.98,held_f1=.98,held_macro_f1=.98,held_macro_exact=.80,
            maximum_f1_drop_from_continuous=.005,continuous_held_f1=.98,false_blank=0,false_ink=0,gain_over_natural_binary=.02,
            maximum_plan_correction_event_ratio=.60,old_direct_tree_mean_events=97.30248883216336),
        resources=dict(maximum_seconds=2400,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=8,maximum_process_rss_gib=8,
            minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        only_final_checkpoints_qualify=True,all_preflight_updates_reset=True,autonomous_language_measured=False,
        full_language_fit_authorized=False,goal_achieved=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG))),flush=True)


def nested_sha(value):
    """Exact model/optimizer replay digest including scalar tensor states."""
    import torch
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().cpu().contiguous();digest.update(repr((tuple(v.shape),str(v.dtype))).encode());digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):digest.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            digest.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:digest.update(repr(x).encode())
    visit(value);return digest.hexdigest()


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Deterministic physical GPU0 required')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in [str(REG.relative_to(ROOT)),*[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git','show','HEAD:'+name])==(ROOT/name).read_bytes(),'Commit every bound source')
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before execution')
    import numpy as np
    import torch
    from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanCodec,xor_correction,pack_plan
    from ilm.visual_lm.binary_plan_codec_learning import codec_forward,reconstruction_terms
    from ilm.visual_lm.form_balanced_codec_support import balanced_codec_support
    from ilm.visual_lm.binary_codec_metrics import codec_metrics
    from ilm.visual_lm.occupancy_glyph_density import encode_occupancy
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    RUN.mkdir();monitor=GlyphReaderMonitor(RUN,2400);monitor.pressure_wait_allowance_seconds=300;completed=[]
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External binary codec comparison deadline')))
    try:
        monitor.status('verifying_binary_codec_sources')
        for name,digest in reg['input_sha256'].items():original.require(original.sha(ROOT/name)==digest,'Bound source changed: '+name)
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            def guard():
                monitor.guard()
                original.require(torch.cuda.max_memory_allocated()/1024**3<=8,'Codec allocation ceiling')
                original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3,'Codec RSS ceiling')
            data=torch.load(SOURCE,weights_only=True,map_location='cpu');partition=torch.load(PARTITION,weights_only=True,map_location='cpu')
            packed=data['packed_ink'];original.require(len(packed)==reg['source_cells'],'Complete physical source differs')
            target=torch.from_numpy(np.unpackbits(packed.numpy(),axis=1,bitorder='big').reshape(-1,1,32,32).copy())
            images=target.to(device);train_indices=partition['train_index'];original.require(len(train_indices)==reg['train_cells'],'Training split differs')
            support=balanced_codec_support(partition,reg['order_seed']);order=support['order']
            original.require(support['training_forms']==8629 and support['training_cells']==64041,'Complete equal-form support differs')
            opath=RUN/'source_order.pt';torch.save(order,opath)
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(reg['seed']).get_state())
            prototype=BinaryPlanCodec(64);initial=copy.deepcopy(prototype.state_dict());initial_hash=tensor_state_sha(initial)
            torch.save(initial,RUN/'common_initial.pt')
            natural=original.load(ROOT/'references/evidence/binary_codec_comparison_result_20261118.json')
            original.require(all(a['initial_tensor_sha256']==initial_hash for a in natural['arms']),'Must preserve original natural-frequency initial tensors')
            def objective(model,mode,indices,backward=False,balance_source=False):
                totals={n:0. for n in ['loss','bce','edge','dice']}
                for start in range(0,len(indices),128):
                    guard();chunk=images[indices[start:start+128].to(device)];out=codec_forward(model,chunk,mode)
                    if mode=='binary':original.require(bool(((out['plan_values']==-1)|(out['plan_values']==1)).all()),'Soft values entered hard training path')
                    terms=reconstruction_terms(out['image_logits'],chunk)
                    if balance_source:
                        weights=support['complete_source_weights'][indices[start:start+128]].to(device=device,dtype=torch.float32)
                        terms={name:value*weights for name,value in terms.items()}
                    if backward:(terms['loss'].sum()/len(indices)).backward()
                    for name,value in terms.items():totals[name]+=float(value.detach().sum())/len(indices)
                return dict(positions=len(indices),**totals)
            def evaluate(model,mode,stage):
                monitor.position=dict(mode=mode,stage=stage);monitor.status('decoding_every_source_image')
                arrays=[];values=[];event_arrays=[];model.eval()
                with torch.no_grad():
                    for start in range(0,len(images),256):
                        guard();chunk=images[start:start+256]
                        if mode=='binary':
                            bits=model.encode_plan(chunk);logits=model.decode_logits(bits);values.append(pack_plan(bits).to(torch.uint8).cpu().numpy())
                        else:
                            out=codec_forward(model,chunk,mode);logits=out['image_logits'];values.append(out['plan_values'].cpu().numpy())
                        ink=(logits>=0).to(torch.uint8);residual=xor_correction(chunk,ink)
                        original.require(torch.equal(xor_correction(ink,residual),chunk),'Physical source recovery failed')
                        events=encode_occupancy(residual,'tree').valid.sum(1).to(torch.int16)
                        arrays.append(ink.cpu().numpy());event_arrays.append(events.cpu().numpy())
                ink=np.concatenate(arrays);events=np.concatenate(event_arrays);value=np.concatenate(values)
                path=RUN/f'{mode}_{stage}.npz';np.savez_compressed(path,ink=ink,plan=value,correction_events=events)
                metrics=codec_metrics(torch.from_numpy(ink),target,partition,torch.from_numpy(events),mode)
                model.train();return dict(metrics=metrics,raw_file=str(path.relative_to(ROOT)),raw_sha256=original.sha(path),
                    all_68948_source_images_preserved=True,all_xor_recoveries_exact=True)
            for mode in reg['modes']:
                guard();model=copy.deepcopy(prototype).to(device).train();original.require(tensor_state_sha(model.state_dict())==initial_hash,'Common initial tensors differ')
                monitor.position=dict(mode=mode);monitor.status('probing_complete_training_source')
                model.zero_grad(set_to_none=True);before=objective(model,mode,train_indices,backward=True,balance_source=True)
                original.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Missing/nonfinite complete-source gradient')
                encoder_gradient=float(model.encoder[-1].weight.grad.norm());decoder_gradient=float(model.expand.weight.grad.norm())
                original.require(encoder_gradient>0 and decoder_gradient>0,'Zero bottleneck/drawing gradient')
                trial=torch.optim.AdamW(model.parameters(),lr=.0002,betas=(.9,.95),weight_decay=.01)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);trial.step();trial_hash=tensor_state_sha(model.state_dict())
                original.require(trial_hash!=initial_hash,'Trial update did not change weights')
                with torch.no_grad():after=objective(model,mode,train_indices,balance_source=True)
                model.load_state_dict(initial);model.zero_grad(set_to_none=True);del trial
                original.require(tensor_state_sha(model.state_dict())==initial_hash,'Complete-source trial reset failed')
                preflight=dict(before=before,after=after,encoder_gradient_norm=encoder_gradient,decoder_gradient_norm=decoder_gradient,
                    total_gradient_norm=float(norm),trial_tensor_sha256=trial_hash,exact_reset=True,retained_updates=0)
                initial_evaluation=evaluate(model,mode,'initial')
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,betas=(.9,.95),weight_decay=.01)
                ledger_path=RUN/f'{mode}_ledger.jsonl';replay=None;updates_done=0;curve=[]
                with ledger_path.open('x') as ledger:
                    for update in range(1,6001):
                        guard();monitor.position=dict(mode=mode,update=update)
                        if update==1 or update%100==0:monitor.status('fitting_matched_binary_codec')
                        indices=order[torch.arange((update-1)*128,update*128)%len(order)]
                        lr=.0002*update/250 if update<=250 else .00002+(.0002-.00002)*.5*(1+math.cos(math.pi*(update-250)/(6000-250)))
                        for group in optimizer.param_groups:group['lr']=lr
                        saved_model=copy.deepcopy(model.state_dict()) if update==3 else None
                        saved_optimizer=copy.deepcopy(optimizer.state_dict()) if update==3 else None
                        def step():
                            optimizer.zero_grad(set_to_none=True);terms=objective(model,mode,indices,backward=True)
                            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                            return dict(**terms,gradient_norm=float(norm))
                        terms=step()
                        if update==3:
                            first_model=tensor_state_sha(model.state_dict());first_optimizer=nested_sha(optimizer.state_dict())
                            model.load_state_dict(saved_model);optimizer.load_state_dict(saved_optimizer);other=step()
                            original.require(terms==other and first_model==tensor_state_sha(model.state_dict()) and first_optimizer==nested_sha(optimizer.state_dict()),'Actual optimizer replay differs')
                            replay=dict(update=3,exact_loss_gradient_model_optimizer=True,extra_trial_images=128,retained_extra_updates=0)
                            del saved_model,saved_optimizer
                        row=dict(update=update,batch_index_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),learning_rate=lr,**terms)
                        ledger.write(json.dumps(row)+'\n');updates_done=update
                        if update==1 or update%100==0:ledger.flush();curve.append(row)
                final_evaluation=evaluate(model,mode,'final');final_hash=tensor_state_sha(model.state_dict())
                checkpoint=RUN/f'{mode}_final.pt';torch.save(dict(mode=mode,updates=6000,model=model.state_dict(),initial_tensor_sha256=initial_hash),checkpoint)
                arm=dict(mode=mode,parameter_count=sum(p.numel() for p in model.parameters()),initial_tensor_sha256=initial_hash,
                    final_tensor_sha256=final_hash,preflight=preflight,optimizer_replay=replay,updates=6000,retained_training_draws=768000,
                    initial=initial_evaluation,final=final_evaluation,ledger_file=str(ledger_path.relative_to(ROOT)),ledger_sha256=original.sha(ledger_path),
                    training_curve=curve,checkpoint_file=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=original.sha(checkpoint))
                completed.append(arm);original.write_new(RUN/f'{mode}.json',arm);del model,optimizer;torch.cuda.empty_cache()
            control,candidate=completed;c=control['final']['metrics'];b=candidate['final']['metrics'];q=reg['quality']
            natural_binary=next(a for a in natural['arms'] if a['mode']=='binary')['final']['metrics']['held']['pixels']['f1']
            checks=dict(gain_over_natural_binary=b['held']['pixels']['f1']-natural_binary>=q['gain_over_natural_binary'],train_f1=b['train']['pixels']['f1']>=q['train_f1'],held_f1=b['held']['pixels']['f1']>=q['held_f1'],
                held_macro_f1=b['held']['macro_form_f1']>=q['held_macro_f1'],held_macro_exact=b['held']['macro_form_exact_rate']>=q['held_macro_exact'],
                retention=b['held']['pixels']['f1']>=c['held']['pixels']['f1']-q['maximum_f1_drop_from_continuous'],
                common_codec_ceiling=c['held']['pixels']['f1']>=q['continuous_held_f1'],false_blank=b['all']['pixels']['false_blank']==0,
                false_ink=b['all']['pixels']['false_ink']==0,event_reduction=b['all']['mean_plan_and_correction_events']<=q['maximum_plan_correction_event_ratio']*q['old_direct_tree_mean_events'])
            for name,digest in reg['input_sha256'].items():original.require(original.sha(ROOT/name)==digest,'Frozen source changed: '+name)
            result=dict(schema='ilm-balanced-codec-comparison-result-v1',registration_sha256=original.sha(REG),source_commit=commit,arms=completed,
                support_checks=checks,support=all(checks.values()),unchanged_natural_control_sha256=original.sha(ROOT/'references/evidence/binary_codec_comparison_result_20261118.json'),
                unchanged_control_reexecuted=False,uniform_physical_forms=8629,all_original_source_cells_retained=True,source_order_file=str(opath.relative_to(ROOT)),source_order_sha256=original.sha(opath),
                source_files_unchanged=True,complete_preflight_updates_reset=True,resources=monitor.receipt(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                scope='Equal-form training distribution, same 64-bit/continuous codecs and fixed original mask validation; reused physical development, no contextual actor/language',
                autonomous_language_measured=False,full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);monitor.status('complete_balanced_codec_comparison')
            print(json.dumps(dict(support=result['support'],checks=checks,continuous_held=c['held'],binary_held=b['held'],binary_all=b['all'])),flush=True)
    except BaseException as error:
        if 'model' in locals():
            torch.save(dict(mode=mode,model=model.state_dict(),updates=locals().get('updates_done',0)),RUN/'failure_state.pt')
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),completed_arms=len(completed),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_balanced_codec_comparison',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
