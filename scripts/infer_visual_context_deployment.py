#!/usr/bin/env python3
"""Standalone raw-image answers from complete learned-context checkpoints."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    import torch
    from ilm.image_edit.baseline import sha,save_json
    from ilm.visual_lm.trainable_context_checkpoint import load_deploy
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from scripts.infer_compact_visual_span import input_cases,generate_case
    args.normal_only=args.profile=='normal'
    cases=input_cases(args)
    if args.profile=='long_books':cases=[c for c in cases if c[0].startswith('book_')]
    if not cases or args.profile=='long_books' and len(cases)!=24:
        raise ValueError('Keep all normal and prefix-erased book inputs')
    args.output.mkdir(parents=True,exist_ok=False)
    monitor=GlyphReaderMonitor(args.output,args.seconds);monitor.status('admitting_visual_context_inference')
    monitor.pressure_wait_allowance_seconds=1800
    identity=dict(checkpoint_sha256=sha(args.checkpoint),
        inputs=[dict(id=k,glyphs=n,sha256=d) for k,_,n,d in cases],profile=args.profile,
        seed=args.seed,flow_steps=args.flow_steps,maximum_glyphs=args.maximum_glyphs,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    save_json(args.output/'request.json',identity)
    try:
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2);monitor.guard()
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            start=time.perf_counter()
            model,updates,provenance=load_deploy(args.checkpoint,args.local,args.codec,device)
            load_seconds=time.perf_counter()-start
            model.context.register_forward_pre_hook(lambda _module,_args:monitor.guard())
            rows=[]
            for case in cases:
                monitor.position=dict(case=case[0],completed=len(rows),total=len(cases));monitor.status('generating_own_pixels')
                monitor.guard()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    row=generate_case(model,device,args,case,args.output)
                rows.append(row)
                print(json.dumps({k:row[k] for k in ('id','emitted_glyphs','stopped','elapsed_seconds')}),flush=True)
            provenance.get('codec',{}).pop('path',None)
            save_json(args.output/'report.json',dict(request_sha256=sha(args.output/'request.json'),
                optimizer_updates=updates,provenance=provenance,checkpoint_load_seconds=load_seconds,
                total_generation_seconds=sum(r['elapsed_seconds'] for r in rows),records=rows,
                reference_inputs=False,output_repaired=False,live_teacher_used=False,
                runtime_lexical_output_head=False,resource_receipt=monitor.receipt(),goal_achieved=False))
            monitor.status('complete_visual_context_inference')
    except BaseException as error:
        monitor.status('failed_visual_context_inference',error=repr(error),automatic_retry=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image',type=Path);g.add_argument('--manifest',type=Path)
    p.add_argument('--glyph-count',type=int);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',choices=('all','normal','long_books'),default='all')
    p.add_argument('--local',type=Path,default=ROOT.parent/'LocalLLM')
    p.add_argument('--codec',type=Path,default=ROOT/'artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt')
    p.add_argument('--maximum-glyphs',type=int,default=128);p.add_argument('--flow-steps',type=int,default=8)
    p.add_argument('--seed',type=int,default=202611153);p.add_argument('--seconds',type=float,default=7200)
    a=p.parse_args()
    if a.manifest and a.glyph_count is not None or a.image and a.profile!='all':p.error('Use profiles with manifest inputs only')
    if min(a.maximum_glyphs,a.flow_steps,a.seconds)<=0:p.error('Positive generation budgets required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        p.error('Require physical GPU0 and the registered CUDA workspace')
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    run(a)
