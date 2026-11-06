"""Host raster preparation for text, image and mixed native ILM prompts.

Only the owning physical pixel tensor reaches the image actor. Text shaping
is an optional host operation, outside the neural language interface.
"""
import hashlib
import math

import torch

from .causal_visual_prefix import fold_writing_prompt
from .codec_image_interface import MAXIMUM_VISUAL_CELLS, PreparedImagePrompt, pixel_sha
from .grounded_image_actor import validate_image


def _prepared(pixels, receipt):
    pixels = pixels.contiguous().clone()
    validate_image(pixels)
    cells = pixels.shape[-2] * pixels.shape[-1] // 32**2
    if cells > MAXIMUM_VISUAL_CELLS:
        raise ValueError("prompt exceeds 512 visual cells; split the prompt or crop explicitly")
    return PreparedImagePrompt(pixels, dict(receipt, actor_shape=list(pixels.shape),
        actor_pixels_sha256=pixel_sha(pixels), visual_cells=cells,
        ocr_calls=0, character_segmentation=False, neural_input="physical_pixels_only"))


def prepare_text_prompt(text):
    """Render complete visible lines, folding long lines at white columns.

    Explicit line breaks remain explicit. Empty lines remain white rows.
    Missing glyphs or insufficient capacity fail rather than inserting tofu,
    shrinking the font, truncating text or asking an OCR service.
    """
    from .full_text_renderer import CompleteLineRenderer
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a nonempty typed prompt is required")
    original_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    original_characters = len(text)
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if len(lines) > MAXIMUM_VISUAL_CELLS:
        raise ValueError("too many explicit prompt lines for this input capacity")
    renderer = CompleteLineRenderer()
    pages, records = [], []
    for index, line in enumerate(lines):
        if not line:
            page = torch.ones((1,1,32,64), dtype=torch.float32)
            records.append(dict(line=index, empty=True, height=32, width=64))
        else:
            raw, metadata = renderer.render(line, maximum_blocks=240)
            strip = torch.from_numpy(raw.copy()).float().div(127.5).sub(1)[None,None]
            folded = fold_writing_prompt(strip)
            if not torch.equal(folded.unfold(), strip):
                raise ValueError("host text folding changed visible pixels")
            page = folded.pixels
            records.append(dict(line=index, empty=False, render=metadata,
                source_ranges=folded.source_ranges, height=page.shape[-2], width=page.shape[-1]))
        pages.append(page)
    height, width = sum(p.shape[-2] for p in pages), max(p.shape[-1] for p in pages)
    if height*width//32**2 > MAXIMUM_VISUAL_CELLS:
        raise ValueError("complete typed prompt exceeds 512 visual cells")
    pixels = torch.ones((1,1,height,width), dtype=torch.float32)
    offset = 0
    for page in pages:
        pixels[...,offset:offset+page.shape[-2],:page.shape[-1]] = page
        offset += page.shape[-2]
    return _prepared(pixels, dict(input_layout="rendered_text_lines", host_text_adapter=True,
        source_text_sha256=original_sha, source_characters=original_characters,
        normalized_characters=len(text), crlf_normalized=True, explicit_lines=len(lines),
        line_layout=records, renderer=renderer.provenance, resized=False,
        missing_glyph_substitution=False, input_truncated=False))


def prepare_instruction_image(instruction, image):
    """Place a visible instruction above an unchanged prepared image page."""
    if not isinstance(image, PreparedImagePrompt):
        raise TypeError("an explicitly prepared image prompt is required")
    text = prepare_text_prompt(instruction)
    upper, lower = text.pixels, image.pixels
    gap = 32
    height, width = upper.shape[-2]+gap+lower.shape[-2], max(upper.shape[-1],lower.shape[-1])
    if height*width//32**2 > MAXIMUM_VISUAL_CELLS:
        raise ValueError("instruction and image together exceed 512 visual cells")
    pixels = torch.ones((1,1,height,width), dtype=torch.float32)
    pixels[...,:upper.shape[-2],:upper.shape[-1]] = upper
    pixels[...,upper.shape[-2]+gap:,:lower.shape[-1]] = lower
    return _prepared(pixels, dict(input_layout="instruction_above_image", host_text_adapter=True,
        text_input=text.receipt, image_input=image.receipt, instruction_image_gap_px=gap,
        image_origin_xy=[0,upper.shape[-2]+gap], resized=False, input_truncated=False))


def validate_native_answer(answer, maximum_blocks):
    images = answer["images"]
    if (images.device.type!="cpu" or images.dtype!=torch.float32 or images.ndim!=4
            or images.shape[1:]!=(1,32,64) or len(images)>maximum_blocks
            or not bool(((images==-1)|(images==1)).all())):
        raise ValueError("unaltered physical native answer blocks required")
    stopped, capped = answer["stopped"], answer["capped"]
    probabilities = answer["stop_probabilities"]
    if (type(stopped) is not bool or type(capped) is not bool or stopped==capped
            or (capped and len(images)!=maximum_blocks) or len(probabilities)!=len(images)+1
            or not all(math.isfinite(v) and 0<=v<=1 for v in probabilities)
            or any(v>=.5 for v in probabilities[:-1]) or (probabilities[-1]>=.5)!=stopped
            or not answer["native_context_kv_retained"] or not answer["feedback_from_emitted_pixels"]
            or not answer["image_input_image_output"] or len(answer["feedback_sha256"])!=len(images)
            or any(answer[k] for k in ("tokenizer","language_vocabulary","ocr","teacher_service"))):
        raise ValueError("native image feedback or learned stopping trace differs")
