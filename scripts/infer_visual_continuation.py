#!/usr/bin/env python3
"""Standalone physical-image inference for a registered continuation checkpoint."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,sha
from ilm.visual_lm.continuation_training import build_actor,restore_trainable
from scripts.infer_glyph_tail_language import prepare_prompt
from scripts.infer_compact_visual_span import save_grid


def load_model(checkpoint,device,allow_intermediate=False):
    from scripts.preflight_visual_continuation import runtime
    value=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if (value.get('schema')!='visual-continuation-training-checkpoint-v1' or value.get('preflight_only',True)
            or (not allow_intermediate and not value['completed_training'])):
        raise ValueError('a completed full fit is required; explicitly allow registered intermediate snapshots for research')
    if value['runtime']!=runtime():raise ValueError('model runtime differs from its training registration')
    from scripts.compare_balanced_writer_matched import CHECKPOINT,CHECKPOINT_SHA
    if sha(CHECKPOINT)!=value['initial_bridge_sha256'] or value['initial_bridge_sha256']!=CHECKPOINT_SHA:
        raise ValueError('inherited image interface weights differ')
    for name,digest in value['inference_source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('inference implementation differs: '+name)
    actor,provenance=build_actor(device,value['seed']);restore_trainable(actor,value['trainable_weights']);actor.eval()
    return actor,dict(provenance,step=value['step'],completed_training=value['completed_training'],
                      checkpoint_sha256=sha(checkpoint),registration_sha256=value['registration_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maximum-glyphs',type=int,default=128);p.add_argument('--allow-intermediate',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    from scripts.infer_glyph_context_motor_image import standalone_reader_gpu
    from scripts.train_semantic_glyph_language import numerics
    from scripts.train_semantic_glyph_bridge import atomic_tensor
    from ilm.visual_lm.continuation_training import TrainingMonitor
    monitor=TrainingMonitor(a.output,1800,300);monitor.status('admitting')
    try:
        with standalone_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();actor,provenance=load_model(a.checkpoint,device,a.allow_intermediate)
            handles=[layer.register_forward_pre_hook(lambda _m,_a:monitor.guard()) for layer in actor.interface.actor.context.layers]
            canvas,observation=prepare_prompt(a.image);prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
            began=time.perf_counter();value=actor.generate(prompt,maximum_glyphs=a.maximum_glyphs)
            torch.cuda.synchronize();elapsed=time.perf_counter()-began;atomic_tensor(a.output/'output.pt',value)
            ink=((1-value['images'])/2).to(torch.uint8);np.savez_compressed(a.output/'ink.npz',ink=ink.numpy());save_grid(ink,a.output/'answer.png')
            for h in handles:h.remove()
            from scripts.train_visual_continuation import atomic_json
            atomic_json(a.output/'report.json',dict(provenance=provenance,input=observation,emitted_glyphs=len(ink),
                stopped=value['stopped'],elapsed_generation_seconds=elapsed,tensor_sha256=sha(a.output/'output.pt'),
                raw_sha256=sha(a.output/'ink.npz'),image_sha256=sha(a.output/'answer.png'),
                actual_emitted_pixel_feedback=True,OCR=False,font_lookup=False,external_teacher=False,**monitor.receipt()))
            monitor.status('complete')
    finally:
        from ilm.visual_lm.continuation_training import load
        from scripts.train_visual_continuation import atomic_json
        atomic_json(a.output/'status.json',dict(load(a.output/'status.json'),pid=None))


if __name__=='__main__':main()
