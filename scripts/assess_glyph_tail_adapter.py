#!/usr/bin/env python3
"""Physical qualification and complete image generation after body adaptation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/assess_glyph_tail_adapter.py'


def load(path):
    return json.loads(Path(path).read_text())


def identity():
    names = [OWN, 'references/glyph_tail_assessment_protocol.md',
             'scripts/infer_glyph_tail_language.py', 'ilm/visual_lm/glyph_tail_adapter.py',
             'ilm/visual_lm/glyph_runtime_resources.py',
             'ilm/visual_lm/semantic_glyph_loop.py']
    for name in names:
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit assessment code first')
    return {name: sha(ROOT / name) for name in names}


def candidate(run):
    m = load(run / 'preparation.json')
    for name, digest in m['code_sha256'].items():
        require(sha(ROOT/name)==digest,'registered training source changed: '+name)
    require(load(run / 'fit/status.json')['phase'] == 'completed', 'complete main adaptation first')
    report = load(run / 'fit/report.json')
    require(report['actual_optimizer_updates'] == m['fit_steps'] and report['all_checks_passed'], 'incomplete fit')
    checkpoint = run / 'fit/checkpoint_latest.pt'
    require(sha(checkpoint) == report['checkpoint_sha256'], 'candidate changed')
    return m, checkpoint


def tail_view(context):
    """Reference existing layer tensors; allocate no second body copy."""
    import torch
    from torch import nn
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    with torch.device('meta'):
        view = Qwen3VLTextModel(context.config)
    view.embed_tokens = None
    view.layers = nn.ModuleList(list(context.layers)[-4:])
    view.norm, view.rotary_emb = context.norm, context.rotary_emb
    return view.eval()


def f1(actual, gold):
    a, g = actual.astype(bool), gold.astype(bool)
    denominator = int(a.sum() + g.sum())
    return 2 * int((a & g).sum()) / denominator if denominator else 1.


def boards(folder, name, rows):
    names = []
    for start in range(0, len(rows), 4):
        group = rows[start:start+4]
        width = max(max(1,len(r['gold']))*32 for r in group)
        canvas = Image.new('L', (max(640,width+12), len(group)*100), 245)
        draw = ImageDraw.Draw(canvas)
        for j,row in enumerate(group):
            draw.text((6,j*100+3),str(row['index'])+' target / actual',fill=0)
            for k,key in enumerate(['gold','actual']):
                for i,glyph in enumerate(row[key]):
                    canvas.paste(Image.fromarray((255*(1-glyph)).astype(np.uint8)),(6+i*32,j*100+24+k*32))
        path = folder / f'{name}_{start//4:02}.png'; canvas.save(path); names.append(path.name)
    return names


def qualify(run):
    code=identity(); m, checkpoint=candidate(run)
    import torch
    from scripts.train_semantic_glyph_language import Monitor, numerics, image, atomic_tensor
    from scripts.infer_glyph_tail_language import load_actor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    folder=run/'qualify'; folder.mkdir(exist_ok=False)
    monitor=Monitor(folder,3600.); monitor.status('admitting')
    try:
        source=load(run/'cache/report.json')['files']
        for row in source: require(sha(ROOT/row['file'])==row['sha256'],'frozen prefix changed')
        primitive_path=ROOT/m['primitive']/'training.pt'
        require(sha(primitive_path)==m['bindings_sha256'][str(primitive_path.relative_to(ROOT))],
                'physical target primitive changed')
        primitive=torch.load(primitive_path,weights_only=True,map_location='cpu')
        ink=primitive['ink'][:,0].numpy()
        samples=set(np.linspace(0,len(source)-1,16,dtype=int).tolist())
        tf_records=[]; tf_images=[]; tf_labels=[]; tf_preview=[]
        terminals=continues=terminal_ok=continue_ok=0
        with full_reader_gpu() as (device,admission), torch.no_grad():
            monitor.admission=admission; numerics(); monitor.guard()
            actor,provenance=load_actor(checkpoint,device); tail=tail_view(actor.interface.actor.context)
            for index,row in enumerate(source):
                monitor.position=dict(route='training_predictors',completed=index,total=len(source)); monitor.guard()
                value=torch.load(ROOT/row['file'],weights_only=True,map_location='cpu',mmap=True)
                h=tail(inputs_embeds=value['lower'].to(device),position_ids=value['position_ids'].to(device),
                       use_cache=False).last_hidden_state[0,value['boundary']-1:]
                logits,_,stop=actor.bridge.predict(h)
                actual=(logits[:-1]>=0).to(torch.uint8)[:,0].cpu().numpy()
                labels=value['labels'][:-1].numpy(); gold=ink[labels]
                probabilities=stop.sigmoid().cpu()
                terminal_ok+=int(probabilities[-1]>=.5); terminals+=1
                continue_ok+=int((probabilities[:-1]<.5).sum()); continues+=len(labels)
                tf_records.extend(dict(case=index,position=i,label=int(label),f1=f1(a,g),exact=bool(np.array_equal(a,g)))
                                  for i,(a,g,label) in enumerate(zip(actual,gold,labels)))
                tf_images.extend(actual); tf_labels.extend(labels.tolist())
                if index in samples: tf_preview.append(dict(index=index,actual=actual,gold=gold))
            del tail
            genuine=ROOT/m['genuine']; gm=load(genuine/'preparation.json')
            held=[r for r in gm['records'] if r['split']=='held_template']
            copy_records=[]; copy_images=[]; copy_labels=[]; copy_preview=[]
            sample_copy=set(np.linspace(0,len(held)-1,16,dtype=int).tolist())
            for index,row in enumerate(held):
                monitor.position=dict(route='actual_adapted_body_copy',completed=index,total=len(held)); monitor.guard()
                prompt=genuine/row['png']; require(sha(prompt)==row['png_sha256'],'copy input changed')
                args,_=actor.interface.assemble(native_budget_image(image(prompt)))
                h=actor.interface.actor.context(**args,use_cache=False).last_hidden_state[:,-1]
                logits,_,stop=actor.bridge.predict(h)
                actual=(logits>=0).to(torch.uint8)[0,0].cpu().numpy(); gold=ink[row['label']]
                copy_records.append(dict(index=index,label=row['label'],f1=f1(actual,gold),
                    exact=bool(np.array_equal(actual,gold)),nonstop=bool(stop.sigmoid()[0]<.5)))
                copy_images.append(actual); copy_labels.append(row['label'])
                if index in sample_copy: copy_preview.append(dict(index=index,actual=actual[None],gold=gold[None]))
            tf_mean=float(np.mean([r['f1'] for r in tf_records])); copy_mean=float(np.mean([r['f1'] for r in copy_records]))
            fraction=float(np.mean([r['f1']>=.95 for r in copy_records]))
            gate=(tf_mean>=.98 and copy_mean>=.95 and fraction>=.8
                  and terminal_ok/terminals>=.95 and continue_ok/continues>=.95
                  and np.mean([r['nonstop'] for r in copy_records])>=.95)
            raw={}
            for name,images,labels in [('training',tf_images,tf_labels),('copy_held',copy_images,copy_labels)]:
                path=folder/(name+'_outputs.pt')
                atomic_tensor(path,dict(ink=torch.from_numpy(np.stack(images)[:,None]),labels=torch.tensor(labels)))
                raw[path.name]=sha(path)
            names=boards(folder,'training',tf_preview)+boards(folder,'copy_held',copy_preview)
            save_json(folder/'records.json',dict(training=tf_records,copy_held=copy_records))
            save_json(folder/'report.json',dict(code_sha256=code,checkpoint_sha256=sha(checkpoint),
                training_mean_f1=tf_mean,copy_held_mean_f1=copy_mean,copy_held_fraction095=fraction,
                terminal_correct=terminal_ok,terminal_cases=terminals,
                continue_correct=continue_ok,continue_cases=continues,
                copy_nonstop_fraction=float(np.mean([r['nonstop'] for r in copy_records])),
                readout_gate_passed=bool(gate),copy_inputs_run_through_actual_adapted_full_body=True,
                raw_outputs_sha256=raw,boards=names,autonomous_language_accuracy=False,
                provenance=provenance,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error)); raise
    finally:
        save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


def assess(run):
    code=identity(); m,checkpoint=candidate(run)
    report=load(run/'qualify/report.json'); review=load(run/'qualify/visual_review.json')
    require(review['report_sha256']==sha(run/'qualify/report.json')
            and review['all8_qualification_boards_actually_viewed'],'actually review physical qualification first')
    for name in report['boards']: require(sha(run/'qualify'/name)==review['boards_sha256'][name],'reviewed board changed')
    import torch
    from scripts.train_semantic_glyph_language import Monitor,image,numerics,atomic_tensor
    from scripts.infer_glyph_tail_language import load_actor,output_png
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    old=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    controls=load(ROOT/'artifacts/semantic_glyph_language_v1/preparation.json')['jobs'][:8]
    jobs=[dict(key=f'train/{i:04}/normal',prompt=r['prompt']) for i,r in enumerate(controls)]
    for i,row in enumerate(old['development'][:16]):
        for route in ['normal','blank']:
            jobs.append(dict(key=f'development/{i:04}/{route}',prompt='artifacts/pixel_language_diversity_v1/'+row[route]))
    jobs.extend(dict(key=f"fresh/{r['index']:04}/normal",prompt=r['prompt']) for r in m['fresh_cases'])
    require(len(jobs)==72,'fixed complete-stream cohort differs')
    folder=run/'assessment'; folder.mkdir(exist_ok=False)
    monitor=Monitor(folder,1800.); monitor.status('admitting')
    try:
        records=[]
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission; numerics(); monitor.guard()
            actor,provenance=load_actor(checkpoint,device)
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            for index,job in enumerate(jobs):
                monitor.position=dict(completed=index,total=72,key=job['key']); monitor.guard()
                prompt=ROOT/job['prompt']
                value=actor.generate(image(prompt),maximum_glyphs=64)
                stem=job['key'].replace('/','_')
                atomic_tensor(folder/(stem+'.pt'),value); output_png(value,folder/(stem+'.png'))
                records.append(dict(**job,input_sha256=sha(prompt),file=stem+'.pt',sha256=sha(folder/(stem+'.pt')),
                    png=stem+'.png',png_sha256=sha(folder/(stem+'.png')),glyphs=len(value['images']),
                    stopped=value['stopped'],capped=value['capped']))
                monitor.status('generating_actual_image_answers'); print(json.dumps(records[-1]),flush=True)
            hook.remove()
            save_json(folder/'report.json',dict(records=records,code_sha256=code,
                checkpoint_sha256=sha(checkpoint),provenance=provenance,references_read=False,
                fixed_physical_gate_passed=report['readout_gate_passed'],
                assessment_does_not_override_failed_component_gate=True,
                source_lexical_head=False,tokenizer=False,font=False,glyph_bank=False,
                actual_emitted_pixel_feedback=True,quality_not_yet_reviewed=True,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error)); raise
    finally:
        save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['qualify','assess'],required=True)
    parser.add_argument('--run',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    args=parser.parse_args(); {'qualify':qualify,'assess':assess}[args.stage](args.run.resolve())
