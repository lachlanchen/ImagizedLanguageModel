#!/usr/bin/env python3
"""Image-only inference with learned contextual deltas and a glyph writer."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_actor(checkpoint, device):
    import torch
    from ilm.visual_lm.glyph_tail_adapter import apply_tail_adapters
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphLanguageActor
    from scripts.train_semantic_glyph_language import reader, bridge_model
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if (not payload['language_training_complete'] or payload['tail_start_layer'] != 32
            or payload['step'] != 18000):
        raise ValueError('a completed contextual-adaptation checkpoint is required')
    interface, provenance = reader(device)
    if len(interface.actor.context.layers) != 36:
        raise ValueError('source body identity differs')
    apply_tail_adapters(interface, payload['body_tail_adapters'])
    bridge = bridge_model({}, device, checkpoint)
    actor = SemanticGlyphLanguageActor(interface, bridge, language_training_complete=True).eval()
    actor.requires_grad_(False)
    return actor, provenance


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
    import torch
    from ilm.image_edit.baseline import save_json, sha
    from scripts.train_semantic_glyph_language import image, numerics, atomic_tensor
    from scripts.score_pixel_language_first_blocks import diagnostic_gpu
    args.output.mkdir(parents=True, exist_ok=False)
    with diagnostic_gpu() as (device, admission), torch.no_grad():
        numerics()
        actor, provenance = load_actor(args.checkpoint, device)
        value = actor.generate(image(args.image), maximum_glyphs=args.maximum_glyphs)
        atomic_tensor(args.output / 'output.pt', value)
        output_png(value, args.output / 'output.png')
        save_json(args.output / 'report.json', dict(input_sha256=sha(args.image),
            checkpoint_sha256=sha(args.checkpoint), output_sha256=sha(args.output / 'output.pt'),
            glyphs=len(value['images']), stopped=value['stopped'], capped=value['capped'],
            source=provenance, actual_emitted_pixel_feedback=True,
            tokenizer=False, font=False, glyph_bank=False, source_lexical_head=False,
            training_or_evaluation_corpus_opened=False))


if __name__ == '__main__':
    main()
