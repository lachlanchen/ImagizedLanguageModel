#!/usr/bin/env python3
"""Qualify a fixed-body writer using exact cached states and actual full-reader copies."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/qualify_balanced_glyph_writer.py'


def load(path):return json.loads(Path(path).read_text())


def main(run):
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'commit writer qualification before execution')
    from scripts.train_balanced_glyph_writer import verify
    from scripts.assess_glyph_tail_adapter import candidate,boards,f1,identity
    m=verify(run);_,checkpoint=candidate(run);code=identity();code[OWN]=sha(ROOT/OWN)
    cache=load(run/'cache/report.json')
    require(cache['actual_parent_training_pixels_and_stop_counts_reproduced']
            and sha(run/'cache/pools.pt')==cache['pool_sha256'],'actual fixed-body cache required')
    import torch
    from scripts.train_semantic_glyph_language import numerics,image,atomic_tensor
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.infer_glyph_tail_language import load_actor
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    folder=run/'qualify';folder.mkdir(exist_ok=False)
    monitor=GlyphReaderMonitor(folder,1800.);monitor.status('admitting')
    try:
        p=torch.load(run/'cache/pools.pt',weights_only=True,map_location='cpu')
        gold=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'][:,0].numpy()
        expected_first=[j['labels'][0] for j in m['training_jobs']]
        expected_later=[i for j in m['training_jobs'] for i in j['labels'][1:]]
        require(p['first_labels'].tolist()==expected_first and p['later_labels'].tolist()==expected_later
                and len(p['terminal'])==len(expected_first),'training pool mapping differs')
        monitor.guard()
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard()
            actor,provenance=load_actor(checkpoint,device)
            predicted={};stop_values={}
            for name in ['first','later','terminal']:
                ink=[];stops=[]
                for start in range(0,len(p[name]),128):
                    monitor.position=dict(route='actual_fixed_body_training_states',pool=name,completed=start,total=len(p[name]))
                    monitor.guard();h=p[name][start:start+128].to(device)
                    if name=='terminal':s=actor.bridge.stop(h).flatten()
                    else:
                        logits,_,s=actor.bridge.predict(h);ink.append((logits>=0).to(torch.uint8)[:,0].cpu().numpy())
                    stops.extend(s.cpu().tolist())
                if ink:predicted[name]=np.concatenate(ink)
                stop_values[name]=np.array(stops)
            sampled=set(np.linspace(0,len(m['training_jobs'])-1,16,dtype=int).tolist())
            rows=[];images=[];labels=[];preview=[];offset=0
            for index,job in enumerate(m['training_jobs']):
                n=len(job['labels']);actual=np.concatenate([predicted['first'][index:index+1],predicted['later'][offset:offset+n-1]])
                offset+=n-1;target=gold[job['labels']]
                rows.extend(dict(case=index,position=j,label=label,f1=f1(a,g),exact=bool(np.array_equal(a,g)))
                    for j,(a,g,label) in enumerate(zip(actual,target,job['labels'])))
                images.extend(actual);labels.extend(job['labels'])
                if index in sampled:preview.append(dict(index=index,actual=actual,gold=target))
            require(offset==len(predicted['later']) and len(rows)==23262,'complete training qualification required')
            terminal_ok=int((stop_values['terminal']>=0).sum())
            continue_ok=sum(int((stop_values[name]<0).sum()) for name in ['first','later'])
            # The new copy check cannot reuse the unadapted feature cache.
            genuine=ROOT/m['genuine'];held=[r for r in load(genuine/'preparation.json')['records'] if r['split']=='held_template']
            copy_rows=[];copy_images=[];copy_labels=[];copy_preview=[]
            sample_copy=set(np.linspace(0,len(held)-1,16,dtype=int).tolist())
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            for index,row in enumerate(held):
                monitor.position=dict(route='actual_full_adapted_reader_copy',completed=index,total=len(held));monitor.guard()
                prompt=genuine/row['png'];require(sha(prompt)==row['png_sha256'],'held copy prompt changed')
                arguments,_=actor.interface.assemble(native_budget_image(image(prompt)))
                h=actor.interface.actor.context(**arguments,use_cache=False).last_hidden_state[:,-1]
                logits,_,stop=actor.bridge.predict(h);actual=(logits>=0).to(torch.uint8)[0,0].cpu().numpy()
                target=gold[row['label']]
                copy_rows.append(dict(index=index,label=row['label'],f1=f1(actual,target),
                    exact=bool(np.array_equal(actual,target)),nonstop=bool(stop[0]<0)))
                copy_images.append(actual);copy_labels.append(row['label'])
                if index in sample_copy:copy_preview.append(dict(index=index,actual=actual[None],gold=target[None]))
            hook.remove();require(len(held)==256,'complete held-copy cohort required')
            train_f1=float(np.mean([r['f1'] for r in rows]));copy_f1=float(np.mean([r['f1'] for r in copy_rows]))
            fraction=float(np.mean([r['f1']>=.95 for r in copy_rows]));nonstop=float(np.mean([r['nonstop'] for r in copy_rows]))
            gate=train_f1>=.98 and copy_f1>=.95 and fraction>=.8 and terminal_ok/3176>=.95 and continue_ok/23262>=.95 and nonstop>=.95
            raw={}
            for name,values,y in [('training',images,labels),('copy_held',copy_images,copy_labels)]:
                path=folder/(name+'_outputs.pt');atomic_tensor(path,dict(ink=torch.from_numpy(np.stack(values)[:,None]),labels=torch.tensor(y)))
                raw[path.name]=sha(path)
            names=boards(folder,'training',preview)+boards(folder,'copy_held',copy_preview)
            save_json(folder/'records.json',dict(training=rows,copy_held=copy_rows))
            save_json(folder/'report.json',dict(code_sha256=code,checkpoint_sha256=sha(checkpoint),
                training_mean_f1=train_f1,copy_held_mean_f1=copy_f1,copy_held_fraction095=fraction,
                terminal_correct=terminal_ok,terminal_cases=3176,continue_correct=continue_ok,continue_cases=23262,
                copy_nonstop_fraction=nonstop,readout_gate_passed=bool(gate),
                copy_inputs_run_through_actual_adapted_full_body=True,
                training_states_cached_from_unchanged_adapted_body=True,
                cached_parent_training_pixels_and_stop_counts_previously_reproduced=True,
                training_cache_report_sha256=sha(run/'cache/report.json'),raw_outputs_sha256=raw,boards=names,
                autonomous_language_accuracy=False,provenance=provenance,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('artifacts/balanced_glyph_writer_v1'))
    main(p.parse_args().run.resolve())
