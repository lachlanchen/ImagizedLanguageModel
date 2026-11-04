"""Host-only complete-line writing renderer; the actor consumes only its pixels."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import unicodedata

import numpy as np


def native_bindings():
    if sys.version_info[:2]!=(3,12):
        raise RuntimeError("use the existing native Python 3.12 rendering environment")
    if "libglib-2.0.so" not in os.environ.get("LD_PRELOAD",""):
        raise RuntimeError("the documented system GLib/GObject/GIO preload is required")
    if "pixar_fontconfig_" in os.environ.get("FONTCONFIG_FILE",""):
        raise RuntimeError("use a fresh process: the released renderer replaces the system font map")
    sys.path.append("/usr/lib/python3/dist-packages")
    import cairo
    import gi
    foreign=Path(__file__).resolve().parents[2]/"artifacts/external/native_bindings/usr/lib/python3/dist-packages/gi"
    if foreign.is_dir() and str(foreign) not in gi.__path__:gi.__path__.append(str(foreign))
    gi.require_foreign("cairo")
    gi.require_version("Pango","1.0");gi.require_version("PangoCairo","1.0")
    from gi.repository import Pango,PangoCairo
    return cairo,Pango,PangoCairo


class CompleteLineRenderer:
    """Render every source mark, or fail before producing a training image.

    Strings and Unicode are used by this host adapter only. The deployed
    image-language actor needs neither this renderer nor a character table.
    """

    def __init__(self,font_family="Noto Sans CJK JP",font_size=24,height=32,padding=2):
        if height!=32 or not 8<=font_size<=28 or padding<2:
            raise ValueError("expected 32-pixel writing height and a fitting font")
        self.cairo,self.Pango,self.PangoCairo=native_bindings()
        self.height,self.padding=height,padding
        self.font_family,self.font_size=font_family,font_size
        available={family.get_name() for family in self.PangoCairo.font_map_get_default().list_families()}
        if font_family not in available:raise ValueError(f"requested font family unavailable: {font_family}")
        self.font=self.Pango.font_description_from_string(f"{font_family} {font_size}px")
        self.surface=self.cairo.ImageSurface(self.cairo.FORMAT_A8,32,32)
        self.context=self.cairo.Context(self.surface)
        primary=Path(subprocess.check_output(["fc-match","-f","%{file}",font_family],text=True))
        self.provenance=dict(font_family=font_family,font_size_px=font_size,height_px=height,
            padding_px=padding,primary_font_path=str(primary),
            primary_font_sha256=hashlib.sha256(primary.read_bytes()).hexdigest(),
            pango_version=self.Pango.version_string(),fallback_selected_fonts_inspected=False,
            source_policy="complete explicit single line, no cropping or font shrinking")

    def _layout(self,text):
        if not isinstance(text,str) or not text or not text.strip():raise ValueError("visible source writing required")
        if any(unicodedata.category(c) in {"Cc","Cf","Cs","Zl","Zp"} for c in text):
            raise ValueError("controls or line breaks require an explicit separate layout")
        layout=self.PangoCairo.create_layout(self.context)
        layout.set_font_description(self.font);layout.set_text(text,-1)
        unknown=layout.get_unknown_glyphs_count()
        if unknown:raise ValueError(f"source has {unknown} unknown font glyphs")
        ink,logical=layout.get_pixel_extents()
        if not ink.width or not ink.height:raise ValueError("source has no visible ink")
        if ink.height>self.height-2*self.padding:raise ValueError("full source ink exceeds the vertical canvas")
        left=min(0,ink.x);right=max(logical.width,ink.x+ink.width)
        width=64*math.ceil((right-left+2*self.padding)/64)
        metadata=dict(source_characters=len(text),logical_width_px=logical.width,
            full_ink_width_px=ink.width,full_ink_height_px=ink.height,width_px=width,blocks=width//64,
            draw_x=self.padding-left,draw_y=(self.height-ink.height)//2-ink.y,
            unknown_glyphs=0,complete_source=True)
        return layout,metadata

    def measure(self,text,maximum_blocks=256):
        _layout,metadata=self._layout(text)
        if metadata["blocks"]>maximum_blocks:raise OverflowError("complete source exceeds maximum image width")
        return metadata

    def render(self,text,maximum_blocks=256):
        layout,metadata=self._layout(text)
        if metadata["blocks"]>maximum_blocks:raise OverflowError("complete source exceeds maximum image width")
        surface=self.cairo.ImageSurface(self.cairo.FORMAT_RGB24,metadata["width_px"],self.height)
        context=self.cairo.Context(surface);context.set_source_rgb(1,1,1);context.paint()
        context.set_source_rgb(0,0,0);context.move_to(metadata["draw_x"],metadata["draw_y"])
        self.PangoCairo.show_layout(context,layout);surface.flush()
        rgba=np.frombuffer(surface.get_data(),np.uint8).reshape(self.height,surface.get_stride()//4,4)
        gray=rgba[:,:metadata["width_px"],:3].mean(-1)
        image=np.where(gray>=128,255,0).astype(np.uint8)
        if not (np.all(image[[0,-1]]==255) and np.all(image[:,[0,-1]]==255)):
            raise ValueError("rendered ink reaches the protected white frame")
        return image,metadata
