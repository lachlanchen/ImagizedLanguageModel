#!/usr/bin/env python3
"""Image-only inference for completed natural-first writer checkpoints.

This entry point accepts a genuine completed 36,000-update candidate. It also
accepts the preserved 18,000-update parent via its unchanged original loader.
Qualification is separate; accepting weights does not certify useful language.
"""
import argparse
from pathlib import Path
import time

from ilm.image_edit.baseline import sha,save_json
from scripts.infer_glyph_tail_language import prepare_prompt,output_png,CONTROL_SHA


def load_actor(checkpoint,device):
    import torch
    payload=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if payload.get('step')==18000:
        del payload
        from scripts.infer_glyph_tail_language import load_actor as original
        return original(checkpoint,device)
    if (payload.get('step')!=36000 or payload.get('new_optimizer_updates')!=36000
            or payload.get('writer_training_family')!='natural_first_image_v1'
            or payload.get('tail_start_layer')!=32 or not payload.get('language_training_complete')):
        raise ValueError('a completed registered natural-first checkpoint is required')
    from scripts.train_semantic_glyph_language import reader,CONTROLS
    from ilm.visual_lm.glyph_tail_adapter import apply_tail_adapters
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge,SemanticGlyphLanguageActor
    if sha(CONTROLS)!=CONTROL_SHA:raise ValueError('fixed numeric task controls changed')
    interface,provenance=reader(device)
    if len(interface.actor.context.layers)!=36:raise ValueError('source body identity differs')
    apply_tail_adapters(interface,payload['body_tail_adapters'])
    bridge=SemanticGlyphBridge().to(device);bridge.load_state_dict(payload['model'],strict=True)
    actor=SemanticGlyphLanguageActor(interface,bridge,language_training_complete=True).eval().requires_grad_(False)
    return actor,dict(provenance,fixed_controls_sha256=CONTROL_SHA,source_derived_fixed_task_controls=True,
        writer_training_family=payload['writer_training_family'],new_writer_optimizer_updates=36000,
        image_feedback_runtime_unchanged=True,language_qualification_not_implied=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--maximum-glyphs',type=int,default=64)
    a=p.parse_args()
    if not 1<=a.maximum_glyphs<=256:p.error('--maximum-glyphs must be between 1 and 256')
    import torch
    import numpy as np
    from scripts.train_semantic_glyph_language import numerics,atomic_tensor
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    a.output.mkdir(parents=True,exist_ok=False);monitor=GlyphReaderMonitor(a.output,1800.);monitor.status('admitting')
    try:
        canvas,observed=prepare_prompt(a.image);canvas.save(a.output/'observed_prompt.png')
        pixels=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
        monitor.guard()
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard()
            began=time.perf_counter();actor,provenance=load_actor(a.checkpoint,device)
            torch.cuda.synchronize();load_seconds=time.perf_counter()-began
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            torch.cuda.synchronize();began=time.perf_counter();waited=monitor.waited
            value=actor.generate(pixels,maximum_glyphs=a.maximum_glyphs)
            torch.cuda.synchronize();generation_seconds=time.perf_counter()-began;generation_wait=monitor.waited-waited
            hook.remove();atomic_tensor(a.output/'output.pt',value);output_png(value,a.output/'output.png')
            save_json(a.output/'report.json',dict(**observed,checkpoint_sha256=sha(a.checkpoint),
                observed_prompt_sha256=sha(a.output/'observed_prompt.png'),
                output_sha256=sha(a.output/'output.pt'),glyphs=len(value['images']),stopped=value['stopped'],
                capped=value['capped'],source=provenance,actual_emitted_pixel_feedback=True,
                actor_loading_wall_seconds=load_seconds,generation_wall_seconds=generation_seconds,
                recorded_generation_pressure_wait_seconds=generation_wait,
                generation_wall_excluding_pressure_wait_seconds=generation_seconds-generation_wait,
                generation_timing_excludes_actor_loading_prompt_read_and_output_save=True,
                tokenizer=False,font=False,glyph_bank=False,source_lexical_head=False,
                training_or_evaluation_corpus_opened=False,useful_language_not_certified_by_loading=True,
                **monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:
        import json
        save_json(a.output/'status.json',dict(json.loads((a.output/'status.json').read_text()),pid=None))


if __name__=='__main__':main()
