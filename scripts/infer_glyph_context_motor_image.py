#!/usr/bin/env python3
"""Standalone image-to-image motor inference; no training manifest or glyph bank."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow, ContextMotorBridge
from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
from scripts.infer_glyph_tail_language import load_actor, prepare_prompt
from scripts.infer_compact_visual_span import save_grid
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.train_semantic_glyph_language import numerics
from scripts.score_pixel_language_first_blocks import require

PARENT_SHA = '5c434a9a9b0c6ddc0c3fe4005265e4f8ca5b7d0add9a74a80ed371806fa71085'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--motor-checkpoint', type=Path, required=True)
    parser.add_argument('--motor-sha256', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--maximum-glyphs', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(1 <= args.maximum_glyphs <= 256 and 0 <= args.seed < 2**63, 'Invalid cap or seed')
    require(sha(args.parent_checkpoint) == PARENT_SHA and sha(args.motor_checkpoint) == args.motor_sha256,
            'Explicit standalone model identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    monitor = GlyphReaderMonitor(args.output, 3600)
    monitor.pressure_wait_allowance_seconds = 600
    monitor.status('admitting')
    try:
        canvas, observation = prepare_prompt(args.image)
        prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None, None]/127.5-1
        canvas.save(args.output/'observed_prompt.png')
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            began = time.perf_counter()
            actor, provenance = load_actor(args.parent_checkpoint, device)
            payload = torch.load(args.motor_checkpoint, map_location='cpu', weights_only=True)
            require(payload['training_complete'] and payload['step'] == 12000
                    and payload['parent_checkpoint_sha256'] == PARENT_SHA, 'Completed registered head required')
            flow = ContextMotorFlow().to(device).eval().requires_grad_(False)
            flow.load_state_dict(payload['ema'])
            actor.bridge = ContextMotorBridge(actor.bridge, flow, payload['motor_mean'].to(device),
                                             payload['motor_scale'].to(device)).eval().requires_grad_(False)
            del payload
            frozen = [(p, p._version) for p in actor.parameters()]
            require(not any(p.requires_grad for p in actor.parameters()), 'Frozen runtime required')
            torch.cuda.synchronize(); loading = time.perf_counter()-began
            hook = actor.interface.actor.context.register_forward_pre_hook(lambda _m, _a:monitor.guard())
            actor.bridge.begin_case(args.seed)
            torch.cuda.synchronize(); began = time.perf_counter()
            with torch.inference_mode():
                value = actor.generate(prompt, maximum_glyphs=args.maximum_glyphs)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-began
            hook.remove()
            value.update(context_motor_flow=True, motor_trace=actor.bridge.motor_trace,
                         case_seed=args.seed, sampling_candidates=1, sampling_intervals=32)
            require(all(p._version == version for p, version in frozen), 'Frozen runtime parameter changed')
            atomic_tensor(args.output/'output.pt', value)
            ink = ((1-value['images'])/2).to(torch.uint8)
            save_grid(ink, args.output/'output.png')
            save_json(args.output/'report.json', dict(**observation,
                parent_checkpoint_sha256=PARENT_SHA, motor_checkpoint_sha256=args.motor_sha256,
                observed_prompt_sha256=sha(args.output/'observed_prompt.png'),
                output_sha256=sha(args.output/'output.pt'), output_PNG_sha256=sha(args.output/'output.png'),
                glyphs=len(ink), stopped=value['stopped'], capped=value['capped'], seed=args.seed,
                actor_loading_seconds=loading, complete_generation_seconds=elapsed,
                actual_emitted_pixels_used_as_history=True, final_EMA_only=True, source=provenance,
                tokenizer=False, lexical_head=False, glyph_bank=False, OCR=False, teacher_service=False,
                training_manifest_opened=False, no_runtime_access_trace_claim_until_cold_audit=True,
                all_frozen_parameter_versions_unchanged=True, language_quality_not_measured=True,
                **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed', error=repr(error)); raise
    finally:
        from json import loads
        save_json(args.output/'status.json', dict(loads((args.output/'status.json').read_text()), pid=None))


if __name__ == '__main__':
    main()
