"""Utilities for English glyph tiling representations."""

from .grid_renderer import (
    GridSpec,
    SPEC_2X2_64,
    SPEC_3X3_40,
    SPEC_4X4_32,
    SPEC_8X8_16,
    DEFAULT_SPEC,
    DEFAULT_RULES,
    select_spec,
    render_char_patch,
    render_word_tile_tensor,
    render_word_tile_image,
    render_words,
)

__all__ = [
    "GridSpec",
    "SPEC_2X2_64",
    "SPEC_3X3_40",
    "SPEC_4X4_32",
    "SPEC_8X8_16",
    "DEFAULT_SPEC",
    "DEFAULT_RULES",
    "select_spec",
    "render_char_patch",
    "render_word_tile_tensor",
    "render_word_tile_image",
    "render_words",
]
