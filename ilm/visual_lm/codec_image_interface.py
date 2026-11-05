"""Image-file input and lossless output for the V68 image-language actor.

This boundary owns its pixel tensors. It neither recognizes writing nor loads
references. Page layout is preserved; strip folding is an explicit choice.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .causal_visual_prefix import fold_writing_prompt
from .grounded_image_actor import validate_image

MAXIMUM_VISUAL_CELLS = 512


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha(pixels):
    return hashlib.sha256(pixels.numpy().tobytes()).hexdigest()


def json_file(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def image_bytes(pixels):
    return ((pixels[0, 0] + 1) * 127.5).round().to(torch.uint8).numpy()


@dataclass(frozen=True)
class PreparedImagePrompt:
    pixels: torch.Tensor
    receipt: dict


def prepare_image_prompt(path, *, layout="page", maximum_width=1024):
    """Normalize a single image without OCR, resizing or inferred layout.

    Pages retain grayscale levels and receive white right/bottom padding to a
    32-pixel boundary. Explicit binary-strip mode uses the training fold and
    wraps only at wholly white columns. Neither path segments characters.
    """
    if layout not in ("page", "strip"):
        raise ValueError("layout must explicitly be page or strip")
    path = Path(path)
    source_sha = sha(path)
    with Image.open(path) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("provide one explicit page; multi-frame files are ambiguous")
        if original.width * original.height > 8 * 1024**2:
            raise ValueError("image exceeds the file-input budget; crop explicitly first")
        dimensions = [original.width, original.height]
        source_mode = original.mode
        oriented = ImageOps.exif_transpose(original).convert("RGBA")
        white = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
        gray = np.array(Image.alpha_composite(white, oriented).convert("L"), copy=True)
    height, width = gray.shape
    pixels = torch.from_numpy(gray).to(torch.float32).div(127.5).sub(1)[None, None].clone()
    ranges = None
    if layout == "strip":
        folded = fold_writing_prompt(pixels, maximum_width=maximum_width)
        if not torch.equal(folded.unfold(), pixels):
            raise ValueError("strip folding changed the observed pixels")
        pixels, ranges = folded.pixels, folded.source_ranges
    else:
        padded_height, padded_width = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
        page = torch.ones((1, 1, padded_height, padded_width), dtype=torch.float32)
        page[..., :height, :width] = pixels
        pixels = page
    validate_image(pixels)
    cells = pixels.shape[-2] * pixels.shape[-1] // 32**2
    if cells > MAXIMUM_VISUAL_CELLS:
        raise ValueError("prompt exceeds 512 visual cells; explicitly crop or split the image")
    if pixels._base is not None or pixels.untyped_storage().nbytes() != pixels.numel() * 4:
        raise ValueError("the actor must receive owning pixel storage")
    if source_sha != sha(path):
        raise ValueError("input file changed while being read")
    return PreparedImagePrompt(pixels, dict(source_sha256=source_sha,
        source_dimensions_wh=dimensions, source_mode=source_mode,
        oriented_grayscale_shape_hw=[height, width], input_layout=layout,
        actor_shape=list(pixels.shape), visual_cells=cells,
        actor_pixels_sha256=pixel_sha(pixels), strip_source_ranges=ranges,
        orientation="EXIF applied", alpha_background="white", color="grayscale",
        resized=False, ocr_calls=0, character_segmentation=False))


def save_prepared_prompt(prepared, out):
    """Create a new immutable input record, retaining precisely the actor input."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(prepared.pixels, out / "prompt.pt")
    Image.fromarray(image_bytes(prepared.pixels)).save(out / "prompt.png")
    receipt = dict(prepared.receipt, prompt_tensor_sha256=sha(out / "prompt.pt"),
        prompt_png_sha256=sha(out / "prompt.png"))
    json_file(out / "input_receipt.json", receipt)
    return receipt


def validate_generation(result, *, maximum_blocks):
    """Validate actual physical blocks and every learned stop event."""
    if type(maximum_blocks) is not int or not 1 <= maximum_blocks <= 128:
        raise ValueError("output capacity must be between 1 and 128 physical blocks")
    if set(result) != {"images", "events", "stopped", "capped"}:
        raise ValueError("unexpected actor result fields")
    count = len(result["images"])
    if (type(result["stopped"]) is not bool or type(result["capped"]) is not bool
            or result["stopped"] == result["capped"] or count > maximum_blocks
            or (result["capped"] and count != maximum_blocks)
            or len(result["events"]) != count + int(result["stopped"])):
        raise ValueError("physical stream length or terminal state differs")
    for index, event in enumerate(result["events"]):
        stopping = result["stopped"] and index == count
        probability = event["stop_probability"]
        if (event["block"] != index or type(event["stopped"]) is not bool
                or event["stopped"] != stopping or not math.isfinite(probability)
                or not 0 <= probability <= 1 or stopping != (probability >= .5)):
            raise ValueError("learned stop event differs from its probability")
    for block in result["images"]:
        validate_image(block)
        if block.shape != (1, 1, 32, 64) or not bool(((block == -1) | (block == 1)).all()):
            raise ValueError("unmodified binary 32-by-64 actor blocks required")
    return count


def save_generation(result, out, *, maximum_blocks):
    """Save raw output and exact PNGs; an empty answer has no placeholder PNG."""
    count = validate_generation(result, maximum_blocks=maximum_blocks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(result, out / "raw_generation.pt")
    files = {"raw_generation.pt": sha(out / "raw_generation.pt")}
    if count:
        pixels = torch.cat(result["images"], dim=-1)
        Image.fromarray(image_bytes(pixels)).save(out / "answer.png")
        files["answer.png"] = sha(out / "answer.png")
        (out / "blocks").mkdir()
        for index, block in enumerate(result["images"]):
            relative = f"blocks/{index:03d}.png"
            Image.fromarray(image_bytes(block)).save(out / relative)
            files[relative] = sha(out / relative)
    receipt = dict(blocks=count, physical_shape_hw=[32, count * 64],
        stopped=result["stopped"], capped=result["capped"], maximum_blocks=maximum_blocks,
        events=result["events"], files_sha256=files, output_repair=False,
        ocr_calls=0, reference_images_used=False, language_quality_claim=False)
    json_file(out / "output_receipt.json", receipt)
    return receipt
