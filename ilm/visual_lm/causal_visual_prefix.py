"""Prepare causal, owning image inputs without a language alphabet or reader.

The page adapter is for the repository's binary 32-pixel writing strips.
Native scanned pages can use a separate direct image interface; they must not
be interpreted as flattened strips by guessing their layout.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FoldedWritingPrompt:
    pixels: torch.Tensor
    source_ranges: tuple[tuple[int, int], ...]

    def unfold(self):
        return torch.cat([self.pixels[..., row*32:(row+1)*32, :end-start]
                          for row, (start, end) in enumerate(self.source_ranges)], dim=-1).clone()


@dataclass(frozen=True)
class CausalVisualPrefix:
    prompt: FoldedWritingPrompt
    completed_blocks: tuple[torch.Tensor, ...]


def _binary_strip(pixels):
    if (pixels.device.type != "cpu" or pixels.dtype != torch.float32
            or pixels.ndim != 4 or pixels.shape[:3] != (1, 1, 32)
            or pixels.shape[-1] < 1):
        raise ValueError("CPU float32 [1,1,32,width] writing pixels required")
    if not bool(((pixels == -1) | (pixels == 1)).all()):
        raise ValueError("lossless binary writing pixels required")


def fold_writing_prompt(pixels, *, maximum_width=1024):
    """Wrap only at wholly white columns; retain every original pixel.

    `source_ranges` describe physical offsets for inversion only. The encoder
    receives `pixels`; it does not need the ranges or any source text.
    """
    _binary_strip(pixels)
    if type(maximum_width) is not int or maximum_width < 64 or maximum_width % 32:
        raise ValueError("page width must be a multiple of 32, at least 64")
    width = min(maximum_width, ((pixels.shape[-1]+31)//32)*32)
    white = (pixels[0, 0] == 1).all(dim=0)
    ranges, start = [], 0
    while start < pixels.shape[-1]:
        end = min(start+width, pixels.shape[-1])
        if end < pixels.shape[-1]:
            lower = start+width//2
            candidates = torch.nonzero(white[lower:end], as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise ValueError("no safe blank-column wrap within the page width")
            end = lower+int(candidates[-1])+1
        ranges.append((start, end))
        start = end
    page = torch.ones((1, 1, 32*len(ranges), width), dtype=torch.float32)
    for row, (start, end) in enumerate(ranges):
        page[..., row*32:(row+1)*32, :end-start] = pixels[..., start:end]
    return FoldedWritingPrompt(page, tuple(ranges))


def prepare_teacher_forced_prefix(image, *, prompt_blocks, completed_blocks, maximum_width=1024):
    """Expose only an observed prompt and a specified completed image prefix.

    Future pixels are never inspected, including for binary/termination
    validation. Returned tensors have fresh storage that cannot alias a full
    target row. The caller supplies completed history length; no future target
    length or content is returned. Encode history blocks separately.
    """
    if (type(prompt_blocks) is not int or prompt_blocks < 1
            or type(completed_blocks) is not int or completed_blocks < 0):
        raise ValueError("positive prompt and nonnegative completed block counts required")
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or image.device.type != "cpu" or image.dtype != torch.float32
            or image.shape[:3] != (1, 1, 32)
            or image.shape[-1] < (prompt_blocks+completed_blocks)*64):
        raise ValueError("physical source prefix is unavailable")
    visible = image[..., :(prompt_blocks+completed_blocks)*64].clone()
    _binary_strip(visible)
    prompt = fold_writing_prompt(visible[..., :prompt_blocks*64], maximum_width=maximum_width)
    history = tuple(visible[..., (prompt_blocks+i)*64:(prompt_blocks+i+1)*64].clone()
                    for i in range(completed_blocks))
    return CausalVisualPrefix(prompt, history)
