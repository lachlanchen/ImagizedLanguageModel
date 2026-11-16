"""Offline complete-page shaping and lossless physical tiling; no model imports.

Unicode and fonts are preparation tools only. Deployed generation must not import
this module or consume its text/font metadata. A tile is an image patch, not a
character identity. Rendering qualification does not qualify a language model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NATIVE_HELPER = ROOT / "ilm/visual_lm/full_text_renderer.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class MissingGlyphError(ValueError):
    pass


def page_to_tiles(page):
    page = np.asarray(page)
    if page.dtype != np.uint8 or page.ndim not in (2, 3) or (page.ndim==3 and page.shape[2]<=0):
        raise ValueError("Expected uint8 grayscale or channel-last image")
    h, w = page.shape[:2]
    if not h or not w or h % 32 or w % 32:
        raise ValueError("Complete page dimensions must be positive multiples of32")
    tail = page.shape[2:]
    axes = (0, 2, 1, 3) + tuple(range(4, page.ndim+2))
    return page.reshape(h//32,32,w//32,32,*tail).transpose(axes).reshape(-1,32,32,*tail).copy()


def tiles_to_page(tiles, shape):
    tiles = np.asarray(tiles)
    if len(shape) not in (2,3) or tiles.dtype != np.uint8 or (len(shape)==3 and shape[2]<=0):
        raise ValueError("Expected original grayscale/RGB shape and uint8 tiles")
    h,w = shape[:2]
    if not h or not w or h % 32 or w % 32 or tiles.shape != (h//32*(w//32),32,32,*shape[2:]):
        raise ValueError("Tile count, shape or page dimensions differ")
    axes = (0, 2, 1, 3) + tuple(range(4, len(shape)+2))
    return tiles.reshape(h//32,w//32,32,32,*shape[2:]).transpose(axes).reshape(shape).copy()


class ShapedPageRenderer:
    def __init__(self, width=1024, font_size=26, padding=8, max_pixels=16777216):
        if width != 1024 or font_size != 26 or padding != 8 or max_pixels != 16777216:
            raise ValueError("Use the registered source-page geometry")
        if sys.byteorder != "little":
            raise RuntimeError("Registered Cairo RGB24 extraction is little-endian")
        # Load only the existing host binding helper, avoiding visual_lm's neural
        # package initializer. The helper's source is bound by the preflight.
        spec = importlib.util.spec_from_file_location("ilm_source_native_bindings",NATIVE_HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.cairo,self.Pango,self.PangoCairo = module.native_bindings()
        if self.Pango.version_string() != "1.52.1":
            raise RuntimeError("Use registered Pango1.52.1; serialized debug format is version-specific")
        self.width,self.padding,self.max_pixels = width,padding,max_pixels
        self.surface = self.cairo.ImageSurface(self.cairo.FORMAT_RGB24,32,32)
        self.context = self.cairo.Context(self.surface)
        self.font = self.Pango.font_description_from_string("Noto Sans CJK SC 26px")
        self.font_cache = {}

    def _mapped_fonts(self):
        fonts = {}
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) != 6:
                continue
            path = Path(parts[5])
            if path.suffix.lower() not in {".ttf", ".ttc", ".otf"} or not path.is_file():
                continue
            stat = path.stat()
            key = (str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in self.font_cache:
                self.font_cache[key] = sha(path)
            fonts[self.font_cache[key]] = str(path)
        return fonts

    def _draw(self, layout, width, height, x, y):
        surface = self.cairo.ImageSurface(self.cairo.FORMAT_RGB24,width,height)
        context = self.cairo.Context(surface)
        context.set_source_rgb(1,1,1)
        context.paint()
        context.set_source_rgb(0,0,0)
        context.move_to(x,y)
        self.PangoCairo.show_layout(context,layout)
        surface.flush()
        bgra = np.frombuffer(surface.get_data(),np.uint8).reshape(height,surface.get_stride()//4,4)
        return bgra[:,:width,2::-1].copy()

    def render(self, text, verify_expanded=False):
        if not isinstance(text,str) or "\0" in text:
            raise ValueError("Source must be an unmodified string without embedded NUL")
        layout = self.PangoCairo.create_layout(self.context)
        layout.set_font_description(self.font)
        layout.set_width((self.width-4*self.padding)*self.Pango.SCALE)
        layout.set_wrap(self.Pango.WrapMode.WORD_CHAR)
        layout.set_ellipsize(self.Pango.EllipsizeMode.NONE)
        layout.set_single_paragraph_mode(False)
        layout.set_auto_dir(True)
        layout.set_text(text,-1)
        if layout.get_text() != text:
            raise ValueError("Native layout altered source text")
        unknown = layout.get_unknown_glyphs_count()
        if unknown:
            raise MissingGlyphError(f"No complete rendering: {unknown} unknown font glyphs")
        ink,logical = layout.get_pixel_extents()
        left = min(0,ink.x,logical.x)
        right = max(0,ink.x+ink.width,logical.x+logical.width)
        top = min(0,ink.y,logical.y)
        bottom = max(0,ink.y+ink.height,logical.y+logical.height)
        if right-left+2*self.padding > self.width:
            raise OverflowError("Complete shaped source exceeds registered width")
        height = max(32,32*((bottom-top+2*self.padding+31)//32))
        if height*self.width > self.max_pixels:
            raise OverflowError("Complete source exceeds pixel ceiling; no truncation")
        x,y = self.padding-left,self.padding-top
        debug_bytes = bytes(layout.serialize(self.Pango.LayoutSerializeFlags.OUTPUT).get_data())
        debug = json.loads(debug_bytes)
        if debug["text"] != text or debug["output"]["is-ellipsized"]:
            raise ValueError("Serialized layout omits or alters source")
        mapped = self._mapped_fonts()
        used = {}
        bidi_levels = set()
        for line in debug["output"]["lines"]:
            for run in line.get("runs",[]):
                value = run["font"]
                checksum = value["checksum"]
                if checksum not in mapped:
                    raise RuntimeError("Cannot authenticate actual selected font bytes")
                identity = json.dumps(value,sort_keys=True)
                used[identity] = dict(value,file=mapped[checksum])
                bidi_levels.add(run["bidi-level"])
        rgb = self._draw(layout,self.width,height,x,y)
        if np.any(rgb[[0,-1]] != 255) or np.any(rgb[:,[0,-1]] != 255):
            raise ValueError("Source ink touches protected page frame")
        if verify_expanded:
            larger = self._draw(layout,self.width+64,height+64,x+32,y+32)
            if not np.array_equal(rgb,larger[32:height+32,32:self.width+32]):
                raise ValueError("Expanded source differs from page crop")
            larger[32:height+32,32:self.width+32] = 255
            if np.any(larger != 255):
                raise ValueError("Clipped ink found outside source page")
        # Preserve color source bytes. Binary projection is explicit and needs
        # separate legibility qualification before use by the current writer.
        binary = np.where(rgb.mean(-1) < 128,0,255).astype(np.uint8)
        metadata = dict(source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_codepoints=len(text),source_bytes=len(text.encode()),page_shape=list(rgb.shape),
            lines=layout.get_line_count(),unknown_glyphs=0,source_text_unchanged=True,
            source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            binary_projection_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),
            binary_projection_rule="arithmetic RGB mean below128 is black; others white",
            binary_projection_legibility_qualified=False,
            selected_fonts=list(used.values()),bidi_levels=sorted(bidi_levels),
            layout_debug_sha256=hashlib.sha256(debug_bytes).hexdigest(),
            complete_source_ink_inside_page=True,expanded_canvas_verified=verify_expanded,
            color_source_preserved=True,tile_size=32,pango_version=self.Pango.version_string())
        return rgb,binary,metadata,debug_bytes
