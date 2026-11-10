#!/usr/bin/env python3
"""Image-only inference with learned contextual deltas and a glyph writer."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

CONTROL_SHA = 'e53cb8d13892aa5de283d55138101ca45527356f73b40a9a965cee77c67167c1'


def load_actor(checkpoint, device):
    import torch
    from ilm.visual_lm.glyph_tail_adapter import apply_tail_adapters
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphLanguageActor
    from scripts.train_semantic_glyph_language import reader, bridge_model, CONTROLS
    from ilm.image_edit.baseline import sha
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if (not payload['language_training_complete'] or payload['tail_start_layer'] != 32
            or payload['step'] != 18000):
        raise ValueError('a completed contextual-adaptation checkpoint is required')
    if sha(CONTROLS) != CONTROL_SHA:
        raise ValueError('fixed numeric task controls changed')
    interface, provenance = reader(device)
    if len(interface.actor.context.layers) != 36:
        raise ValueError('source body identity differs')
    apply_tail_adapters(interface, payload['body_tail_adapters'])
    bridge = bridge_model({}, device, checkpoint)
    actor = SemanticGlyphLanguageActor(interface, bridge, language_training_complete=True).eval()
    actor.requires_grad_(False)
    return actor, dict(provenance, fixed_controls_sha256=CONTROL_SHA, source_derived_fixed_task_controls=True)


def prepare_prompt(path):
    """Prepare a physical image without reading, recognizing or rendering text."""
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        original_size = list(source.size)
        observed = ImageOps.exif_transpose(source).convert('RGBA')
        white = Image.new('RGBA', observed.size, 'white'); white.alpha_composite(observed)
        observed = white.convert('L')
    if observed.width*observed.height > 1048576:
        ratio = (1048576/(observed.width*observed.height))**.5
        observed = observed.resize((max(1, int(observed.width*ratio)), max(1, int(observed.height*ratio))),
                                   Image.Resampling.BILINEAR)
    canvas = Image.new('L', (max(32, (observed.width+31)//32*32),
                             max(32, (observed.height+31)//32*32)), 255)
    canvas.paste(observed, (0, 0))
    return canvas, dict(input_sha256=hashlib.sha256(raw).hexdigest(), original_size=original_size,
                        observed_size=list(canvas.size), host_preparation_OCR_or_font=False,
                        host_preparation='EXIF orientation, white alpha composite, grayscale, bounded resize, right/bottom white padding')


def output_png(value, path):
    a = np.full((32, max(1, len(value['images'])) * 32), 255, np.uint8)
    for i, tile in enumerate(value['images']):
        a[:, i*32:(i+1)*32] = ((tile[0].numpy() + 1) * 127.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-glyphs', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.maximum_glyphs <= 256:
        parser.error('--maximum-glyphs must be between 1 and 256')
    import torch
    from ilm.image_edit.baseline import save_json, sha
    from scripts.train_semantic_glyph_language import numerics, atomic_tensor
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor as Monitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    args.output.mkdir(parents=True, exist_ok=False)
    monitor=Monitor(args.output,1800.)
    monitor.status('admitting')
    try:
        canvas, observed = prepare_prompt(args.image)
        canvas.save(args.output/'observed_prompt.png')
        prompt_pixels=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
        monitor.guard()
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission=admission
            numerics()
            monitor.guard()
            loading_started=time.perf_counter()
            actor, provenance = load_actor(args.checkpoint, device)
            torch.cuda.synchronize(); loading_seconds=time.perf_counter()-loading_started
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            torch.cuda.synchronize(); started=time.perf_counter(); waited_before=monitor.waited
            value = actor.generate(prompt_pixels, maximum_glyphs=args.maximum_glyphs)
            torch.cuda.synchronize(); generation_seconds=time.perf_counter()-started
            generation_wait=monitor.waited-waited_before
            hook.remove()
            atomic_tensor(args.output / 'output.pt', value)
            output_png(value, args.output / 'output.png')
            save_json(args.output / 'report.json', dict(**observed,
                observed_prompt_sha256=sha(args.output/'observed_prompt.png'),
                checkpoint_sha256=sha(args.checkpoint), output_sha256=sha(args.output / 'output.pt'),
                glyphs=len(value['images']), stopped=value['stopped'], capped=value['capped'],
                source=provenance, actual_emitted_pixel_feedback=True,
                actor_loading_wall_seconds=loading_seconds,generation_wall_seconds=generation_seconds,
                recorded_generation_pressure_wait_seconds=generation_wait,
                generation_wall_excluding_recorded_pressure_wait_seconds=generation_seconds-generation_wait,
                generation_timing_includes_image_prefill_pixel_feedback_and_stop=True,
                generation_timing_excludes_prompt_file_read_output_serialization_and_actor_loading=True,
                tokenizer=False, font=False, glyph_bank=False, source_lexical_head=False,
                training_or_evaluation_corpus_opened=False,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error))
        raise
    finally:
        status=json.loads((args.output/'status.json').read_text())
        save_json(args.output/'status.json',dict(status,pid=None))


if __name__ == '__main__':
    main()
