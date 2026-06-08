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


def draw_glow(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int, color: str) -> None:
    x, y = center
    for i in range(radius, 0, -10):
        alpha = i / radius
        # PIL RGB draw has no alpha compositing here; use progressively lighter outlines.
        draw.ellipse((x - i, y - i, x + i, y + i), outline=color, width=1)


def paste_svg_on_card(
    base: Image.Image,
    path: Path | None,
    *,
    char: str | None,
    box: Tuple[int, int, int, int],
    label: str,
    sublabel: str,
    fill_text: str = "#f8fafc",
) -> None:
    d = ImageDraw.Draw(base)
    x1, y1, x2, y2 = box
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(box, radius=24, fill=(8, 16, 30, 178), outline=(220, 185, 120, 220), width=2)
    base.alpha_composite(overlay)

    glyph_size = min(x2 - x1 - 70, y2 - y1 - 110)
    gx = x1 + (x2 - x1 - glyph_size) // 2
    gy = y1 + 48
    if path is not None and path.exists():
        glyph = load_svg(path, glyph_size)
        # Invert black SVG strokes to warm ivory for dark cards.
        bg = Image.new("RGBA", glyph.size, (0, 0, 0, 0))
        pix = glyph.load()
        for yy in range(glyph.height):
            for xx in range(glyph.width):
                r, g, b, a = pix[xx, yy]
                if a and (r + g + b) < 520:
                    pix[xx, yy] = (250, 230, 185, a)
                elif a:
                    pix[xx, yy] = (255, 255, 255, 0)
        base.alpha_composite(glyph, (gx, gy))
    elif char:
        f = cjk_font(glyph_size)
        bbox = d.textbbox((0, 0), char, font=f)
        d.text((x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2, gy - 12), char, font=f, fill="#fde68a")

    centered_text(d, (x1 + 10, y2 - 76, x2 - 10, y2 - 42), label, font(24, True), fill_text)
    centered_text(d, (x1 + 10, y2 - 42, x2 - 10, y2 - 12), sublabel, font(18), "#cbd5e1")


def yan_cover_hero() -> None:
    bg_path = FIG / "aginti_yan_background.png"
    if bg_path.exists():
        bg = Image.open(bg_path).convert("RGB")
    else:
        bg = Image.new("RGB", (2048, 1536), "#07111f")
        bd = ImageDraw.Draw(bg)
        for x in range(0, 2048, 60):
            bd.line((x, 0, x + 700, 1536), fill="#0e7490", width=1)
    bg = bg.resize((1800, 1200), Image.LANCZOS).crop((0, 90, 1800, 1090)).convert("RGBA")

    # Darken and vignette to make publication labels readable.
    shade = Image.new("RGBA", bg.size, (0, 0, 0, 70))
    bg.alpha_composite(shade)
    d = ImageDraw.Draw(bg)
    W, H = bg.size

    # Top title block.
    title_box = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(title_box)
    td.rounded_rectangle((58, 48, 910, 205), radius=28, fill=(3, 7, 18, 178), outline=(245, 196, 106, 190), width=2)
    bg.alpha_composite(title_box)
    d.text((88, 72), "Image-Native Language Model", font=font(52, True), fill="#fff7ed")
    d.text((91, 143), "reading and writing language as images, not text tokens", font=font(25), fill="#dbeafe")

    # Flow boxes.
    flow_y = 275
    flow = [
        ((90, flow_y, 400, flow_y + 135), "IMAGE INPUT", "book page / oracle shard"),
        ((735, flow_y - 25, 1085, flow_y + 170), "ILM-V", "visual memory + latent diffusion"),
        ((1365, flow_y, 1710, flow_y + 135), "IMAGE OUTPUT", "answer sheet with glyph forms"),
    ]
    for box, head, sub in flow:
        x1, y1, x2, y2 = box
        ov = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rounded_rectangle(box, radius=24, fill=(5, 13, 27, 184), outline=(125, 211, 252, 220), width=2)
        bg.alpha_composite(ov)
        centered_text(d, (x1 + 8, y1 + 20, x2 - 8, y1 + 68), head, font(25, True), "#ecfeff")
        centered_text(d, (x1 + 12, y1 + 68, x2 - 12, y2 - 16), sub, font(20), "#bae6fd")
    arrow(d, (420, flow_y + 67), (718, flow_y + 67), "#67e8f9")
    arrow(d, (1100, flow_y + 67), (1345, flow_y + 67), "#67e8f9")

    # Central model chip.
    chip = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(chip)
    cd.rounded_rectangle((690, 485, 1125, 800), radius=36, fill=(10, 24, 46, 210), outline=(103, 232, 249, 235), width=3)
    for i in range(13):
        x = 710 + i * 32
        cd.line((x, 455, x, 485), fill=(103, 232, 249, 180), width=3)
        cd.line((x, 800, x, 835), fill=(251, 191, 36, 170), width=3)
    for i in range(8):
        y = 510 + i * 34
        cd.line((655, y, 690, y), fill=(103, 232, 249, 180), width=3)
        cd.line((1125, y, 1165, y), fill=(251, 191, 36, 170), width=3)
    bg.alpha_composite(chip)
    centered_text(d, (710, 520, 1105, 592), "ILM-V", font(48, True), "#ecfeff")
    centered_text(d, (720, 610, 1095, 710), "visual canvas encoder\n2D memory transformer\nmasked image generator", font(22), "#dbeafe")
    centered_text(d, (720, 730, 1095, 775), "patches are visual units", font(21, True), "#fde68a")

    # Real Yan evolution panels.
    data_root = Path("/home/lachlan/ProjectsLFS/incoder/data/historic/glyphs/言")
    glyphs = [
        ("Oracle", "J04903", data_root / "oracle" / "J04903.svg"),
        ("Bronze", "B02975", data_root / "bronze" / "B02975.svg"),
        ("Seal", "S01648", data_root / "seal" / "S01648.svg"),
        ("Modern", "U+8A00", None),
    ]
    x = 92
    y = 610
    card_w = 255
    for i, (stage, sub, path) in enumerate(glyphs):
        box = (x + i * (card_w + 24), y, x + i * (card_w + 24) + card_w, y + 310)
        paste_svg_on_card(bg, path, char="言" if path is None else None, box=box, label=stage, sublabel=sub)
        if i < len(glyphs) - 1:
            arrow(d, (box[2] + 4, y + 155), (box[2] + 22, y + 155), "#fbbf24")

    # Output answer card on right.
    answer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    ad = ImageDraw.Draw(answer)
    ad.rounded_rectangle((1240, 545, 1718, 905), radius=28, fill=(248, 250, 252, 235), outline=(186, 230, 253, 255), width=3)
    ad.rounded_rectangle((1272, 585, 1688, 705), radius=18, fill=(15, 23, 42, 230))
    bg.alpha_composite(answer)
    d.text((1288, 604), "Prompt: evolution of YAN", font=font(26, True), fill="#e0f2fe")
    d.text((1288, 652), "Answer is an image, not token text.", font=font(21), fill="#bae6fd")
    d.text((1275, 735), "YAN means speech / language.", font=font(25, True), fill="#0f172a")
    d.text((1275, 780), "The model retrieves real historical forms,\nplaces them in a readable explanation,\nand renders the whole answer as pixels.", font=font(22), fill="#334155")
    d.text((1275, 872), "Oracle → Bronze → Seal → Modern", font=font(22, True), fill="#92400e")

    # Footer provenance.
    d.text((72, 958), "Real glyph exemplars from local hanziyuan-derived ziyuan data for YAN (U+8A00): oracle, bronze, seal, modern.", font=font(18), fill="#e5e7eb")
    bg.convert("RGB").save(FIG / "ilm_v_yan_readme_hero.png", quality=95)


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
    yan_cover_hero()
    aginti_loop()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
