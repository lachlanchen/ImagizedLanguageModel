#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def cjk_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], fill: str, outline: str, radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def centered_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, fnt, fill: str = "#18212f") -> None:
    lines = text.split("\n")
    line_h = fnt.size + 6 if hasattr(fnt, "size") else 20
    total_h = line_h * len(lines)
    y = box[1] + (box[3] - box[1] - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h


def arrow(draw: ImageDraw.ImageDraw, p1: Tuple[int, int], p2: Tuple[int, int], color: str = "#344054") -> None:
    draw.line([p1, p2], fill=color, width=4)
    x1, y1 = p1
    x2, y2 = p2
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)]
    else:
        pts = [(x2, y2), (x2 + 14, y2 - 8), (x2 + 14, y2 + 8)]
    draw.polygon(pts, fill=color)


def architecture() -> None:
    img = Image.new("RGB", (1800, 980), "#f7f9fc")
    d = ImageDraw.Draw(img)
    title = font(48, True)
    body = font(27)
    small = font(22)
    d.text((70, 45), "Image-Native Language Model (ILM-V)", font=title, fill="#101828")

    boxes = [
        ((80, 190, 360, 380), "Input canvas\nbook page / glyph\noracle / cuneiform", "#e0f2fe", "#0284c7"),
        ((455, 190, 735, 380), "Visual encoder\npatches + strokes\nlayout pyramid", "#ecfdf3", "#16a34a"),
        ((830, 190, 1110, 380), "Visual memory\n2D transformer\ncross-panel state", "#fff7ed", "#f97316"),
        ((1205, 190, 1485, 380), "Latent generator\nmasked diffusion\ninpainting prior", "#f5f3ff", "#7c3aed"),
        ((1580, 190, 1760, 380), "Output\nimage", "#fef2f2", "#dc2626"),
    ]
    for box, label, fill, outline in boxes:
        rounded(d, box, fill, outline)
        centered_text(d, box, label, body)

    for x in [360, 735, 1110, 1485]:
        arrow(d, (x + 15, 285), (x + 80, 285))

    # Bottom objectives.
    objectives = [
        ((110, 565, 455, 760), "Masked visual LM\npredict missing strokes\nand page patches"),
        ((535, 565, 880, 760), "Image instruction tuning\nquestion image ->\nanswer image"),
        ((960, 565, 1305, 760), "Glyph evolution\nmodern <-> oracle\nbronze / seal"),
        ((1385, 565, 1730, 760), "Readability critics\nOCR only as evaluator\nnot the main path"),
    ]
    for box, label in objectives:
        rounded(d, box, "#ffffff", "#98a2b3", radius=14)
        centered_text(d, box, label, small, "#344054")

    d.text((70, 875), "Key idea: patches/latents are visual units, not BPE or word tokens.", font=font(26, True), fill="#475467")
    img.save(FIG / "architecture_overview.png")


def curriculum() -> None:
    img = Image.new("RGB", (1800, 1050), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "3090-Friendly Training Curriculum", font=font(46, True), fill="#101828")

    phases = [
        ("0", "Data pipeline", "glyph DB, rendered pages,\nimage instruction pairs", "#e0f2fe"),
        ("1", "Glyph autoencoder", "128-256 px glyph tiles\nVAE / VQ-VAE", "#dcfce7"),
        ("2", "Evolution generator", "modern -> oracle/bronze/seal\nconditional latent diffusion", "#fef3c7"),
        ("3", "Visual page LM", "line/page image continuation\nmasked page denoising", "#f3e8ff"),
        ("4", "Instruction images", "prompt image -> answer image\nlayout-aware generation", "#ffe4e6"),
        ("5", "Multiscript expansion", "Chinese, books, cuneiform,\nhieroglyphic-style forms", "#e0e7ff"),
    ]
    x0 = 80
    y = 190
    w = 260
    h = 240
    gap = 35
    for i, (num, name, desc, fill) in enumerate(phases):
        x = x0 + i * (w + gap)
        rounded(d, (x, y, x + w, y + h), fill, "#667085", radius=18)
        d.ellipse((x + 18, y + 18, x + 70, y + 70), fill="#101828")
        centered_text(d, (x + 18, y + 18, x + 70, y + 70), num, font(28, True), "#ffffff")
        centered_text(d, (x + 10, y + 78, x + w - 10, y + 135), name, font(25, True))
        centered_text(d, (x + 12, y + 140, x + w - 12, y + h - 12), desc, font(19), "#344054")
        if i < len(phases) - 1:
            arrow(d, (x + w + 5, y + h // 2), (x + w + gap - 8, y + h // 2), "#475467")

    lanes = [
        ("1 GPU", "Phase 1-2: autoencoder and glyph evolution, 30M-250M params, bf16, checkpointing."),
        ("2 GPUs", "Phase 3-4: visual page LM, 200M-600M params, latent crops, gradient accumulation."),
        ("Always", "Use OCR only for evaluation/auxiliary loss; keep the primary path image-to-image."),
    ]
    y2 = 555
    for i, (tag, text) in enumerate(lanes):
        top = y2 + i * 120
        rounded(d, (120, top, 1680, top + 88), "#f8fafc", "#cbd5e1", radius=12)
        d.text((155, top + 26), tag, font=font(28, True), fill="#0f172a")
        d.text((320, top + 29), text, font=font(25), fill="#334155")

    img.save(FIG / "training_curriculum.png")


def load_svg(path: Path, size: int = 190) -> Image.Image:
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(url=str(path), output_width=size, output_height=size)
        import io

        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        fallback = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        dr = ImageDraw.Draw(fallback)
        dr.rectangle((20, 20, size - 20, size - 20), outline="#111827", width=5)
        dr.line((size // 2, 15, size // 2, size - 15), fill="#111827", width=8)
        return fallback


def zhong_evolution() -> None:
    data_root = Path("/home/lachlan/ProjectsLFS/incoder/data/historic/glyphs/中")
    samples = [
        ("Oracle", data_root / "oracle" / "J00886.svg"),
        ("Bronze", data_root / "bronze" / "B00610.svg"),
        ("Seal", data_root / "seal" / "S00273.svg"),
        ("Liushutong", data_root / "liushutong" / "L00585.svg"),
        ("Modern", None),
    ]
    img = Image.new("RGB", (1800, 800), "#fbfbfd")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "Example Output: Evolution of Chinese Character Zhong (中)", font=font(42, True), fill="#101828")
    d.text((72, 105), "A target answer is itself an image: explanation plus historical forms.", font=font(25), fill="#475467")
    card_w = 295
    x0 = 90
    y0 = 210
    for idx, (stage, path) in enumerate(samples):
        x = x0 + idx * (card_w + 40)
        rounded(d, (x, y0, x + card_w, y0 + 430), "#ffffff", "#d0d5dd", radius=20)
        centered_text(d, (x, y0 + 25, x + card_w, y0 + 75), stage, font(26, True))
        if path is not None and path.exists():
            glyph = load_svg(path, 205)
            img.paste(Image.new("RGB", glyph.size, "#ffffff"), (x + 45, y0 + 95))
            img.paste(glyph, (x + 45, y0 + 95), glyph)
            label = path.stem
        else:
            f = cjk_font(185)
            d.text((x + 70, y0 + 105), "中", font=f, fill="#111827")
            label = "Unicode U+4E2D"
        centered_text(d, (x + 25, y0 + 325, x + card_w - 25, y0 + 375), label, font(22), "#344054")
        if idx < len(samples) - 1:
            arrow(d, (x + card_w + 8, y0 + 205), (x + card_w + 34, y0 + 205), "#667085")

    d.text((90, 700), "Figure uses local hanziyuan-derived glyph files when available; generation targets should retrieve/cite real exemplars.", font=font(22), fill="#667085")
    img.save(FIG / "zhong_evolution_example.png")


def aginti_loop() -> None:
    img = Image.new("RGB", (1800, 900), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "AgInTi-Assisted Research Artifact Loop", font=font(44, True), fill="#101828")
    nodes = [
        ((110, 210, 430, 390), "Research prompt\nand paper outline", "#e0f2fe"),
        ((560, 210, 880, 390), "AgInTi figure brief\nlayout + labels", "#fef3c7"),
        ((1010, 210, 1330, 390), "Deterministic PNG\nscripted diagrams", "#dcfce7"),
        ((1460, 210, 1690, 390), "LaTeX paper\nfigures embedded", "#f3e8ff"),
    ]
    for box, text, fill in nodes:
        rounded(d, box, fill, "#667085")
        centered_text(d, box, text, font(25, True))
    for a, b in [((430, 300), (560, 300)), ((880, 300), (1010, 300)), ((1330, 300), (1460, 300))]:
        arrow(d, a, b)

    rounded(d, (180, 560, 1620, 735), "#ffffff", "#cbd5e1", radius=18)
    d.text((230, 600), "Why scripted PNGs now?", font=font(30, True), fill="#0f172a")
    d.text((230, 650), "They are reproducible, editable, and compile cleanly in the paper while leaving room for later AgInTi/image-model refinements.", font=font(25), fill="#334155")
    img.save(FIG / "aginti_artifact_loop.png")


def main() -> None:
    architecture()
    curriculum()
    zhong_evolution()
    aginti_loop()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
