#!/usr/bin/env python3
"""Image-only inference from final adapter exports; no training data is read."""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT,gpu_zero,save_json,sha
from ilm.image_edit.native_flow_run import CONDITION,identity,restore_adapters
from ilm.image_edit import visual_semantic as vs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--initial-source',action='store_true',help='matched source control with zero adapters; load no learned adapter weights')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    info=json.loads((args.deployment/'deployment.json').read_text());weights=args.deployment/'adapters.safetensors'
    code=identity()
    if info['format']!='native_visual_flow_adapter_v1' or info['source_revision']!=vs.REVISION or info['code_sha256']!=code:
        raise ValueError('deployment source/format differs')
    if sha(weights)!=info['weights_sha256']:raise ValueError('final exported adapter weights changed')
    for file,key in [('manifest.json','condition_manifest_sha256'),('condition.safetensors','condition_payload_sha256')]:
        if sha(CONDITION/file)!=info[key]:raise ValueError('fixed condition changed')
    rows=json.loads(args.inputs.read_text());inputs_sha=sha(args.inputs)
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('unique nonempty input-image index required')
    for r in rows:
        if set(r)!={'id','input','input_sha256','seed'} or sha(args.inputs.parent/r['input'])!=r['input_sha256']:
            raise ValueError('inference index must contain only bound input images and seeds')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    save_json(args.output/'status.json',dict(phase='starting',pid=os.getpid()))
    try:
        source=vs.verify_release()
        if source!=info['source_sha256']:raise ValueError('deployed base release changed')
        with gpu_zero() as device:
            import torch
            from safetensors.torch import load_file
            from ilm.image_edit.native_flow_step import add_adapters,trainable
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            pipe,loader=vs.load_pipeline(device);vs.install_compiled(pipe,CONDITION);removal=vs.remove_lexical_modules(pipe)
            layers=add_adapters(pipe,info['rank'])
            if layers!=info['adapter_layer_names']:raise ValueError('deployed adapter graph changed')
            if not args.initial_source:
                restore_adapters(pipe,load_file(str(weights),device='cpu'))
            for name in ('transformer','connector_module'):getattr(pipe,name).eval()
            outputs=[]
            for i,r in enumerate(rows):
                if identity()!=code or sha(args.inputs.parent/r['input'])!=r['input_sha256']:raise ValueError('input or source changed while live')
                value=vs.generate(pipe,args.inputs.parent/r['input'],seed=r['seed'],width=info['output_canvas'][0],
                    height=info['output_canvas'][1],steps=info['steps'],guidance=info['guidance'])
                result=vs.save_generation(value,args.output/f'{i:03}')
                outputs.append(dict(id=r['id'],path=f'{i:03}/answer.png',sha256=result['sha256'],result=result,
                    input_sha256=r['input_sha256'],seed=r['seed']))
                save_json(args.output/'outputs.json',outputs)
                save_json(args.output/'status.json',dict(phase='generating',pid=os.getpid(),completed=len(outputs),total=len(rows)))
                print(f'Generated {len(outputs)}/{len(rows)}',flush=True)
            if identity()!=code or sha(args.inputs)!=inputs_sha or sha(weights)!=info['weights_sha256']:
                raise ValueError('inference source/model/index changed')
            save_json(args.output/'report.json',dict(source=source,loader=loader,removal=removal,
                deployment_sha256=sha(args.deployment/'deployment.json'),
                weights_sha256=None if args.initial_source else sha(weights),
                exported_weights_file_sha256=sha(weights),learned_adapter_weights_loaded=not args.initial_source,
                arm='initial_source' if args.initial_source else 'final',inputs_sha256=inputs_sha,
                complete_generations=len(outputs),code_sha256=code,runtime_parameters=vs.parameters(pipe),
                adapter_parameters=sum(p.numel() for p in trainable(pipe).values()),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.perf_counter()-started,
                training_manifest_loaded=False,optimizer_loaded=False,target_or_transcript_used=False,
                changing_text_condition=False,output_repair=False,external_teacher_or_OCR=False,
                language_quality_qualified=False,general_ilm_goal_complete=False))
        save_json(args.output/'status.json',dict(phase='completed',pid=None))
    except BaseException as error:
        save_json(args.output/'status.json',dict(phase='failed',pid=None,exception=type(error).__name__,reason=str(error)));raise


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1');main()
