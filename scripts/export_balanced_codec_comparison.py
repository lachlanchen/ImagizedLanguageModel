#!/usr/bin/env python3
"""Recount the equal-form comparison and preserve natural-frequency controls."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/balanced_codec_comparison_v1'
REG=ROOT/'references/evidence/balanced_codec_comparison_registration_20261118.json'
OUT=ROOT/'references/evidence/balanced_codec_comparison_20261118'
PUBLIC=ROOT/'references/evidence/balanced_codec_comparison_result_20261118.json'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only')
    original.require(not OUT.exists() and not PUBLIC.exists(),'Preserve every export')
    terminal=original.load(ROOT/'artifacts/balanced_codec_comparison_v1.exit.json')
    original.require(terminal['phase']=='terminal' and terminal['exit_code']==0,'Neural comparison not complete')
    import fcntl
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from ilm.visual_lm.binary_codec_metrics import codec_metrics
    from ilm.visual_lm.form_balanced_codec_support import balanced_codec_support
    from ilm.visual_lm.binary_plan_glyph_density import xor_correction
    from ilm.visual_lm.occupancy_glyph_density import encode_occupancy
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2);began=time.perf_counter();samples=[]
    def guard():
        r=host_resources();samples.append(r)
        original.require(r['available_gib']>=24 and time.perf_counter()-began<240,'CPU codec audit reserve/deadline')
        original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*1024**3,'CPU codec audit RSS cap')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources();original.require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,'CPU codec audit admission')
        reg=original.load(REG);raw=original.load(RUN/'result.json')
        original.require(raw['registration_sha256']==original.sha(REG),'Registration changed')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
        paths=[REG,RUN/'result.json',ROOT/'artifacts/balanced_codec_comparison_v1.exit.json']
        inputs={str(p.relative_to(ROOT)):original.sha(p) for p in paths}
        data=torch.load(ROOT/'artifacts/nonlinear_feedback_alignment_v1/prepared.pt',weights_only=True,map_location='cpu')
        partition=torch.load(ROOT/'artifacts/binary_codec_preparation_v1/partition.pt',weights_only=True,map_location='cpu')
        target=torch.from_numpy(np.unpackbits(data['packed_ink'].numpy(),axis=1,bitorder='big').reshape(-1,1,32,32).copy())
        orderpath=ROOT/raw['source_order_file'];original.require(original.sha(orderpath)==raw['source_order_sha256'],'Source order changed')
        inputs[str(orderpath.relative_to(ROOT))]=original.sha(orderpath);order=torch.load(orderpath,weights_only=True,map_location='cpu')
        expected=balanced_codec_support(partition,reg['order_seed'])['order']
        original.require(torch.equal(order,expected),'Actual complete training order differs')
        initial=torch.load(RUN/'common_initial.pt',weights_only=True,map_location='cpu');initial_hash=tensor_state_sha(initial)
        inputs['artifacts/balanced_codec_comparison_v1/common_initial.pt']=original.sha(RUN/'common_initial.pt')
        arrays={};audits=[]
        structure=original.load(ROOT/'references/evidence/occupancy_density_structure_result_20261118.json')
        oldregpath=ROOT/'references/evidence/occupancy_density_structure_registration_20261118.json';oldreg=original.load(oldregpath)
        original.require(original.sha(oldregpath)==structure['registration_sha256'],'Completed structural evidence changed')
        source_name='artifacts/nonlinear_feedback_alignment_v1/prepared.pt'
        original.require(oldreg['input_sha256'][source_name]==reg['input_sha256'][source_name],'Old event denominator uses another source')
        original.require(structure['modes']['tree']['mean_events_per_cell']==reg['quality']['old_direct_tree_mean_events'],'Reuse the completed source-event control')
        for arm in raw['arms']:
            mode=arm['mode'];guard()
            original.require(arm['initial_tensor_sha256']==initial_hash and arm['preflight']['exact_reset'],'Common tensors or trial reset differs')
            original.require(arm['preflight']['before']['positions']==arm['preflight']['after']['positions']==reg['train_cells'],'Incomplete real support preflight')
            original.require(arm['optimizer_replay']==dict(update=3,exact_loss_gradient_model_optimizer=True,extra_trial_images=128,retained_extra_updates=0),'Optimizer replay not complete')
            checkpoint=ROOT/arm['checkpoint_file'];original.require(original.sha(checkpoint)==arm['checkpoint_sha256'],'Final checkpoint bytes changed')
            inputs[str(checkpoint.relative_to(ROOT))]=original.sha(checkpoint)
            saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
            original.require(saved['updates']==6000 and saved['mode']==mode and tensor_state_sha(saved['model'])==arm['final_tensor_sha256'],'Final checkpoint metadata/tensors differ')
            original.require(sum(v.numel() for v in saved['model'].values())==arm['parameter_count']==2030465,'Actual codec parameter count differs')
            ledger=ROOT/arm['ledger_file'];original.require(original.sha(ledger)==arm['ledger_sha256'],'Source/update ledger changed')
            inputs[str(ledger.relative_to(ROOT))]=original.sha(ledger);rows=[json.loads(line) for line in ledger.read_text().splitlines()]
            original.require(len(rows)==arm['updates']==6000 and arm['retained_training_draws']==768000,'Fixed complete fit differs')
            for update,row in enumerate(rows,1):
                idx=order[torch.arange((update-1)*128,update*128)%len(order)]
                lr=.0002*update/250 if update<=250 else .00002+(.0002-.00002)*.5*(1+math.cos(math.pi*(update-250)/(6000-250)))
                original.require(row['update']==update and row['positions']==128 and row['learning_rate']==lr,'Fixed optimizer schedule differs')
                original.require(row['batch_index_sha256']==hashlib.sha256(idx.numpy().tobytes()).hexdigest(),'Actual training image order differs')
                original.require(all(math.isfinite(row[k]) for k in ['loss','bce','edge','dice','gradient_norm']),'Nonfinite retained training value')
            original.require(arm['training_curve']==[r for r in rows if r['update']==1 or r['update']%100==0],'Training curve changed')
            for stage in ['initial','final']:
                guard();report=arm[stage];path=ROOT/report['raw_file'];original.require(original.sha(path)==report['raw_sha256'],'Actual reconstructions changed')
                inputs[str(path.relative_to(ROOT))]=original.sha(path)
                with np.load(path) as a:
                    ink=a['ink'];values=a['plan'];events=a['correction_events']
                original.require(ink.dtype==np.uint8 and ink.shape==(68948,1,32,32) and np.isin(ink,[0,1]).all(),'Every actual binary reconstruction required')
                if mode=='binary':original.require(values.dtype==np.uint8 and values.shape==(68948,8),'Every hard 64-bit plan required')
                else:original.require(values.dtype==np.float32 and values.shape==(68948,64) and np.isfinite(values).all() and (np.abs(values)<=1).all(),'Every bounded continuous code required')
                original.require(events.shape==(68948,),'Every correction count required')
                actual_events=[]
                for start in range(0,68948,256):
                    guard();a=torch.from_numpy(ink[start:start+256]);b=target[start:start+256]
                    residual=xor_correction(a,b);original.require(torch.equal(xor_correction(a,residual),b),'Exact physical recovery failed')
                    actual_events.append(encode_occupancy(residual,'tree').valid.sum(1))
                actual_events=torch.cat(actual_events)
                original.require(np.array_equal(actual_events.numpy(),events),'Actual correction event recount differs')
                measured=codec_metrics(torch.from_numpy(ink),target,partition,actual_events,mode)
                original.require(measured==report['metrics'],'All-source physical metric recount differs')
                audits.append(dict(mode=mode,stage=stage,positions=68948,all_pixels_recounted=True,all_xor_and_event_counts_exact=True,
                    metrics=measured))
                if stage=='final':arrays[mode]=ink
        c=raw['arms'][0]['final']['metrics'];b=raw['arms'][1]['final']['metrics'];q=reg['quality']
        natural_path=ROOT/'references/evidence/binary_codec_comparison_result_20261118.json';natural=original.load(natural_path)
        original.require(original.sha(natural_path)==raw['unchanged_natural_control_sha256'] and not raw['unchanged_control_reexecuted'],'Preserve unchanged natural-frequency controls')
        original.require(all(a['initial_tensor_sha256']==initial_hash for a in natural['arms']),'Original initial tensors differ')
        natural_binary=next(a for a in natural['arms'] if a['mode']=='binary')['final']['metrics']['held']['pixels']['f1']
        checks=dict(gain_over_natural_binary=b['held']['pixels']['f1']-natural_binary>=q['gain_over_natural_binary'],train_f1=b['train']['pixels']['f1']>=q['train_f1'],held_f1=b['held']['pixels']['f1']>=q['held_f1'],
            held_macro_f1=b['held']['macro_form_f1']>=q['held_macro_f1'],held_macro_exact=b['held']['macro_form_exact_rate']>=q['held_macro_exact'],
            retention=b['held']['pixels']['f1']>=c['held']['pixels']['f1']-q['maximum_f1_drop_from_continuous'],common_codec_ceiling=c['held']['pixels']['f1']>=q['continuous_held_f1'],
            false_blank=b['all']['pixels']['false_blank']==0,false_ink=b['all']['pixels']['false_ink']==0,
            event_reduction=b['all']['mean_plan_and_correction_events']<=q['maximum_plan_correction_event_ratio']*q['old_direct_tree_mean_events'])
        original.require(checks==raw['support_checks'] and all(checks.values())==raw['support'],'Qualification verdict differs')
        def first_distinct(held,number,nonempty=False):
            chosen=[];seen=set()
            for idx in (partition['held']==held).nonzero().flatten().tolist():
                if nonempty and not bool(target[idx].any()):continue
                g=int(partition['group'][idx])
                if g not in seen:chosen.append(idx);seen.add(g)
                if len(chosen)==number:return chosen
            raise ValueError('Not enough fixed distinct physical examples')
        held_indices=first_distinct(True,64);train_indices=first_distinct(False,15,True)
        empty=int((~target.bool().flatten(1).any(1)).nonzero()[0]);train_indices=sorted([empty,*train_indices])
        OUT.mkdir();exports=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
        for name,indices in [('held',held_indices),('train',train_indices)]:
            width=4*292;height=((len(indices)+3)//4)*102+28;canvas=Image.new('L',(width,height),255);draw=ImageDraw.Draw(canvas)
            draw.text((4,3),'Source | continuous | binary | XOR correction (2x nearest pixel enlargement)',font=font,fill=0)
            tiles=[]
            for n,idx in enumerate(indices):
                x=(n%4)*292;y=(n//4)*102+28;draw.text((x+3,y),f'{name} source-cell {idx}',font=font,fill=0)
                source=target[idx,0].numpy();continuous=arrays['continuous'][idx,0];binary=arrays['binary'][idx,0]
                for j,tile in enumerate([source,continuous,binary,source^binary]):
                    raster=255*(1-tile);canvas.paste(Image.fromarray(raster).resize((64,64),Image.Resampling.NEAREST),(x+3+72*j,y+18));tiles.append((x+3+72*j,y+18,raster))
            path=OUT/f'{name}_fixed_reconstructions.png';canvas.save(path);back=np.asarray(Image.open(path))
            for x,y,raster in tiles:original.require(np.array_equal(back[y:y+64:2,x:x+64:2],raster),'Export changed an original image pixel')
            exports.append(dict(file=str(path.relative_to(ROOT)),sha256=original.sha(path),source_indices=indices,columns='source, continuous final, binary final, target XOR binary final',scale=2))
        result=dict(raw);result['resources']=dict(raw['resources']);result['resources'].pop('admission',None)
        result.update(input_sha256=inputs,audits=audits,exports=exports,all_275792_reconstructions_recounted=True,
            all_12000_training_updates_and_order_checked=True,unchanged_control_bytes_verified=True,
            natural_control_comparison={a['mode']:dict(natural_held_f1=a['final']['metrics']['held']['pixels']['f1'],
                balanced_held_f1=next(v for v in raw['arms'] if v['mode']==a['mode'])['final']['metrics']['held']['pixels']['f1']) for a in natural['arms']},exporter_sha256=original.sha(Path(__file__)),
            audit_resources=dict(elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                minimum_available_host_gib=min(v['available_gib'] for v in samples),cpu_threads=2),
            independent_recount_scope='Persisted pixels, exact corrections/events, metrics, checkpoints and ledgers; no repeated neural training or decoding')
        original.write_new(PUBLIC,result)
        print(json.dumps(dict(public=str(PUBLIC.relative_to(ROOT)),sha256=original.sha(PUBLIC),support=raw['support'],images=275792,exports=len(exports))),flush=True)


if __name__=='__main__':main()
