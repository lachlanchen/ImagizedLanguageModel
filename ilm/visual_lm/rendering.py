from __future__ import annotations

import hashlib
import io
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageFilter


DEFAULT_GLYPH_ROOTS = [
    Path("/home/lachlan/ProjectsLFS/incoder/data/historic/glyphs"),
    Path("/home/lachlan/Projects/incoder/data/historic/glyphs"),
    Path("data/historic/glyphs"),
]

STAGES = ("oracle", "bronze", "seal", "liushutong")
STAGE_LABELS = {
    "oracle": "Oracle",
    "bronze": "Bronze",
    "seal": "Seal",
    "liushutong": "Liushutong",
    "modern": "Modern",
}

ENGLISH_HINTS = {
    "言": "speech, language, saying",
    "中": "center, middle",
    "水": "water, river",
    "日": "sun, day",
    "月": "moon, month",
    "人": "person, human",
    "山": "mountain",
    "火": "fire",
    "木": "tree, wood",
    "口": "mouth, opening",
    "学": "learning, study",
    "車": "cart, vehicle",
    "车": "cart, vehicle",
}


@dataclass(frozen=True)
class RenderConfig:
    image_size: int = 384
    margin: int = 24
    background: str = "#fbf7ec"
    ink: str = "#2b2318"
    line: str = "#d9c796"
    accent: str = "#7c2d12"

    def __post_init__(self) -> None:
        if self.image_size < 384:
            raise ValueError("RenderConfig.image_size must be at least 384 for page-style ILM-V samples.")


@dataclass(frozen=True)
class GlyphExample:
    char: str
    stage: str
    path: Path

    @property
    def label(self) -> str:
        return self.path.stem


def resolve_glyph_root(path: str | Path | None = None) -> Path | None:
    if path:
        p = Path(path).expanduser()
        return p if p.exists() else None
    for p in DEFAULT_GLYPH_ROOTS:
        if p.exists():
            return p
    return None


def font(size: int, *, bold: bool = False, serif: bool = False, cjk: bool = False) -> ImageFont.FreeTypeFont:
    cjk_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc" if serif else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc" if serif else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    latin_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold and serif else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf" if serif else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for candidate in (cjk_candidates + latin_candidates if cjk else latin_candidates + cjk_candidates):
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def contains_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width_px: int,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    *,
    line_spacing: int = 6,
) -> int:
    x, y = xy
    approx = max(4, int(width_px / max(8, getattr(fnt, "size", 18) * 0.56)))
    lines: list[str] = []
    for raw in text.splitlines():
        if contains_cjk(raw):
            buf = ""
            for ch in raw:
                if draw.textlength(buf + ch, font=fnt) > width_px and buf:
                    lines.append(buf)
                    buf = ch
                else:
                    buf += ch
            if buf:
                lines.append(buf)
        else:
            lines.extend(textwrap.wrap(raw, width=approx) or [""])
    line_h = getattr(fnt, "size", 18) + line_spacing
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h
    return y


def svg_to_image(path: Path, size: int) -> Image.Image:
    try:
        import cairosvg

        png = cairosvg.svg2png(url=str(path), output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGBA")
    except Exception:
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        d = ImageDraw.Draw(canvas)
        d.rectangle((12, 12, size - 12, size - 12), outline="#111827", width=3)
        d.text((size // 4, size // 3), path.stem[:2], font=font(max(12, size // 6), bold=True), fill="#111827")
        return canvas


def load_glyph_image(path: Path, size: int) -> Image.Image:
    if path.suffix.lower() == ".svg":
        return svg_to_image(path, size)
    img = Image.open(path).convert("RGBA")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def paste_contain(base: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    work = img.copy()
    work.thumbnail((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    px = x1 + (x2 - x1 - work.width) // 2
    py = y1 + (y2 - y1 - work.height) // 2
    base.paste(work, (px, py), work)


def draw_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    glyph: Image.Image | None,
    subtitle: str,
    cfg: RenderConfig,
) -> None:
    d = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box
    d.rounded_rectangle(box, radius=10, fill="#fffdf7", outline="#d0b886", width=2)
    title_font = font(15, bold=True)
    if d.textlength(title, font=title_font) > (x2 - x1 - 14):
        title_font = font(12, bold=True)
    d.text((x1 + 7, y1 + 8), title, font=title_font, fill="#1f2937")
    if glyph is not None:
        paste_contain(canvas, glyph, (x1 + 12, y1 + 36, x2 - 12, y2 - 34))
    d.text((x1 + 10, y2 - 25), subtitle, font=font(12), fill="#475467")


class GlyphCorpus:
    def __init__(self, root: str | Path | None = None, *, characters: Sequence[str] | None = None):
        self.root = resolve_glyph_root(root)
        if self.root is None:
            raise FileNotFoundError("No historic glyph root found. Pass --glyph-root or ingest data first.")
        self.characters = list(characters) if characters else self.discover_characters(max_chars=256)

    def discover_characters(self, *, max_chars: int = 256, min_stages: int = 2) -> list[str]:
        chars: list[str] = []
        for p in sorted(self.root.iterdir(), key=lambda x: x.name):
            if not p.is_dir():
                continue
            stages = [s for s in STAGES if (p / s).exists() and any((p / s).iterdir())]
            if len(stages) >= min_stages:
                chars.append(p.name)
            if len(chars) >= max_chars:
                break
        return chars

    def examples_for(self, char: str, *, rng: random.Random | None = None) -> list[GlyphExample]:
        rng = rng or random
        out: list[GlyphExample] = []
        cdir = self.root / char
        for stage in STAGES:
            sdir = cdir / stage
            if not sdir.exists():
                continue
            files = [p for p in sorted(sdir.iterdir()) if p.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg"}]
            if files:
                out.append(GlyphExample(char=char, stage=stage, path=rng.choice(files)))
        return out

    def sample_char(self, rng: random.Random) -> str:
        return rng.choice(self.characters)


def page_hash_seed(*parts: object) -> int:
    h = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def render_prompt_page(char: str, cfg: RenderConfig, *, variant: int = 0) -> Image.Image:
    rng = random.Random(page_hash_seed("prompt", char, variant))
    img = Image.new("RGB", (cfg.image_size, cfg.image_size), cfg.background)
    d = ImageDraw.Draw(img)
    m = cfg.margin
    d.rounded_rectangle((m, m, cfg.image_size - m, cfg.image_size - m), radius=18, fill="#fffaf0", outline="#d8c697", width=2)
    d.rectangle((m + 18, m + 18, cfg.image_size - m - 18, m + 62), fill="#efe2bd")
    d.text((m + 28, m + 26), "Visual prompt image", font=font(20, bold=True), fill="#3f2b12")
    big = font(max(54, cfg.image_size // 4), cjk=True, serif=True)
    bbox = d.textbbox((0, 0), char, font=big)
    d.text(((cfg.image_size - (bbox[2] - bbox[0])) // 2, m + 80), char, font=big, fill="#111827")
    y = m + 88 + (bbox[3] - bbox[1]) + 18
    prompt_lines = [
        f"Explain the evolution of {char}.",
        "Use English, Chinese, and classical-style Chinese.",
        "Show historical forms as image glyph panels.",
    ]
    if rng.random() < 0.5:
        prompt_lines.append("The answer must be a rendered page image.")
    for line in prompt_lines:
        y = draw_wrapped(d, (m + 30, y), line, cfg.image_size - 2 * m - 60, font(16), cfg.ink)
        y += 4
    y += 8
    for yy in range(y, cfg.image_size - m - 24, max(18, cfg.image_size // 22)):
        d.line((m + 30, yy, cfg.image_size - m - 30, yy), fill="#e8dcb8", width=1)
    return img


def render_answer_page(
    char: str,
    glyphs: Sequence[GlyphExample],
    cfg: RenderConfig,
    *,
    variant: int = 0,
) -> Image.Image:
    img = Image.new("RGB", (cfg.image_size, cfg.image_size), cfg.background)
    d = ImageDraw.Draw(img)
    m = cfg.margin
    d.rounded_rectangle((m, m, cfg.image_size - m, cfg.image_size - m), radius=18, fill="#fffaf0", outline="#d8c697", width=2)
    d.rectangle((m + 18, m + 16, cfg.image_size - m - 18, m + 58), fill="#efe2bd")
    sense = ENGLISH_HINTS.get(char, "historical character form")
    d.text((m + 26, m + 24), f"{char}  {sense}", font=font(19, cjk=True, serif=True), fill="#3f2b12")

    y = m + 76
    body_f = font(max(12, cfg.image_size // 24))
    cjk_f = font(max(13, cfg.image_size // 22), cjk=True, serif=True)
    y = draw_wrapped(
        d,
        (m + 28, y),
        f"English: {char} is treated as a visible written form. The answer is generated as a page image, not as a token string.",
        cfg.image_size - 2 * m - 56,
        body_f,
        cfg.ink,
        line_spacing=5,
    )
    y += 4
    y = draw_wrapped(
        d,
        (m + 28, y),
        f"中文：{char} 的歷史形體可由甲骨、金文、篆書等圖像證據連成脈絡。",
        cfg.image_size - 2 * m - 56,
        cjk_f,
        cfg.ink,
        line_spacing=4,
    )
    y += 4
    y = draw_wrapped(
        d,
        (m + 28, y),
        f"古文式：觀其形，可以知其義之所自來。",
        cfg.image_size - 2 * m - 56,
        cjk_f,
        cfg.ink,
        line_spacing=4,
    )
    y += 10
    d.line((m + 28, y, cfg.image_size - m - 28, y), fill=cfg.line, width=2)
    y += 14

    panel_count = min(4, len(glyphs) + 1)
    gap = max(8, cfg.image_size // 48)
    panel_w = (cfg.image_size - 2 * m - 56 - gap * (panel_count - 1)) // panel_count
    panel_h = max(88, cfg.image_size // 3)
    x = m + 28
    for idx, ex in enumerate(glyphs[: panel_count - 1]):
        gimg = load_glyph_image(ex.path, panel_h - 42)
        draw_panel(img, (x, y, x + panel_w, y + panel_h), STAGE_LABELS[ex.stage], gimg, ex.label, cfg)
        x += panel_w + gap
    modern_img = Image.new("RGBA", (panel_h, panel_h), (255, 255, 255, 0))
    md = ImageDraw.Draw(modern_img)
    mf = font(panel_h - 58, cjk=True, serif=True)
    bbox = md.textbbox((0, 0), char, font=mf)
    md.text(((panel_h - (bbox[2] - bbox[0])) // 2, 15), char, font=mf, fill="#111827")
    draw_panel(img, (x, y, x + panel_w, y + panel_h), "Modern", modern_img, f"U+{ord(char):04X}", cfg)
    y += panel_h + 16
    source = " · ".join([f"{STAGE_LABELS[g.stage]} {g.label}" for g in glyphs[:3]])
    draw_wrapped(d, (m + 28, y), f"Provenance: {source}", cfg.image_size - 2 * m - 56, font(12), "#6b4f1d")
    return img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=110, threshold=3))


def make_triptych(prompt: Image.Image, pred: Image.Image, target: Image.Image) -> Image.Image:
    w, h = prompt.size
    out = Image.new("RGB", (w * 3, h + 34), "#f8fafc")
    d = ImageDraw.Draw(out)
    for i, (label, im) in enumerate([("prompt image", prompt), ("model output image", pred), ("target image", target)]):
        x = i * w
        out.paste(im.convert("RGB"), (x, 34))
        d.text((x + 12, 8), label, font=font(18, bold=True), fill="#0f172a")
    return out
