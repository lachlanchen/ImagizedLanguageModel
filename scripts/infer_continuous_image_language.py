#!/usr/bin/env python3
"""Standalone research checkpoint: prompt image/text -> actual answer pixels.

Typed prompts are rendered only at the input boundary. The checkpoint contains
all neural codec/reader/writer weights. No corpus, teacher or output font is read.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path


def image_cells(path, glyphs=None):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        value = np.asarray(image.convert('L')).copy()
    h, w = value.shape
    if h % 32 or w % 32:
        raise ValueError('Prompt image must be a grid of 32-by-32 cells')
    cells = (value < 128).reshape(h // 32, 32, w // 32, 32).transpose(0, 2, 1, 3).reshape(-1, 1, 32, 32)
    if glyphs is not None:
        if not 1 <= glyphs <= len(cells):
            raise ValueError('Input glyph count must describe existing prompt cells')
        cells = cells[:glyphs]
    return cells.astype(np.uint8).copy()


def text_cells(text, font_path=None):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scripts.prepare_visual_sequence_corpus import FONT, Rasterizer
    if not text:
        raise ValueError('Typed input needs at least one character')
    if font_path is None or (FONT.is_file() and Path(font_path).samefile(FONT)):
        renderer = Rasterizer()
        packed = b''.join(renderer.tile(char) for char in text)
        return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder='big').reshape(-1, 1, 32, 32)
    font = ImageFont.truetype(str(font_path), 26)
    cells = []
    for char in text:
        image = Image.new('L', (32, 32), 255)
        if char not in '\r\n':
            box = font.getbbox(char)
            x = (32 - (box[2] - box[0])) // 2 - box[0]
            y = (32 - (box[3] - box[1])) // 2 - box[1]
            ImageDraw.Draw(image).text((x, y), char, font=font, fill=0)
        cells.append((np.asarray(image) < 128)[None])
    return np.stack(cells).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--text')
    parser.add_argument('--font', type=Path, help='Optional input font; default uses the training font and layout')
    parser.add_argument('--input-glyphs', type=int, help='Optional count excluding final grid padding')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--maximum-new', type=int, default=128)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--seed', type=int, default=202611200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing generated output')
    if not 1 <= args.maximum_new <= 640:
        raise ValueError('Use a finite output cap between 1 and 640')
    cells = image_cells(args.image, args.input_glyphs) if args.image else text_cells(args.text, args.font)
    if not 1 <= len(cells) <= 2048:
        raise ValueError('Prompt must contain 1 to 2048 aligned writing cells')
    import numpy as np
    import torch
    from PIL import Image
    from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
    from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
    if args.device == 'cuda:0':
        from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
        scope = full_reader_gpu()
    else:
        scope = nullcontext((torch.device('cpu'), dict(mode='cpu_standalone')))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with scope as (device, admission):
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        if checkpoint.get('components') not in (1, 8) or 'model' not in checkpoint:
            raise ValueError('Expected a complete continuous image-language checkpoint')
        codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
        model = ContinuousImageLanguage(codec, components=checkpoint['components']).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        output, receipt = model.generate(torch.from_numpy(cells).to(device), maximum_new=args.maximum_new,
            stochastic=args.sample, generator=generator)
        pixels = output.cpu().numpy()
        count = len(pixels); cols = min(32, max(1, count)); rows = max(1, (count + cols - 1) // cols)
        page = np.full((rows * 32, cols * 32), 255, dtype=np.uint8)
        for i, tile in enumerate(pixels):
            y, x = divmod(i, cols)
            page[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32] = (1 - tile[0]) * 255
        args.output.mkdir(parents=True)
        Image.fromarray(page).save(args.output / 'answer.png')
        np.savez_compressed(args.output / 'actual_pixels.npz', ink=pixels)
        digest = hashlib.sha256()
        with args.checkpoint.open('rb') as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(part)
        result = dict(checkpoint_sha256=digest.hexdigest(), checkpoint_update=checkpoint.get('update'),
            components=checkpoint['components'], prompt_cells=len(cells), typed_input_rendering=bool(args.text),
            prompt_ink_sha256=hashlib.sha256(cells.tobytes()).hexdigest(),
            stochastic=args.sample, seed=args.seed, output_fonts_used=False,
            external_teacher_or_corpus_used=False, language_quality_established=False, **receipt)
        (args.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
