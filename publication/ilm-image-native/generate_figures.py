#!/usr/bin/env python3
from __future__ import annotations

import math
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


def arrow_any(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: str = "#344054",
    width: int = 4,
) -> None:
    draw.line([p1, p2], fill=color, width=width)
    x1, y1 = p1
    x2, y2 = p2
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 16
    spread = 0.48
    pts = [
        (x2, y2),
        (int(x2 - head * math.cos(angle - spread)), int(y2 - head * math.sin(angle - spread))),
        (int(x2 - head * math.cos(angle + spread)), int(y2 - head * math.sin(angle + spread))),
    ]
    draw.polygon(pts, fill=color)


def line_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = "#18212f",
    spacing: int = 8,
) -> int:
    x, y = xy
    line_h = getattr(fnt, "size", 22) + spacing
    for line in text.split("\n"):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h
    return y


def mini_page(
    base: Image.Image,
    box: Tuple[int, int, int, int],
    title: str,
    lines: Iterable[str],
    *,
    tint: str = "#fffaf0",
    outline: str = "#d6c7a1",
) -> None:
    d = ImageDraw.Draw(base)
    x1, y1, x2, y2 = box
    d.rounded_rectangle(box, radius=12, fill=tint, outline=outline, width=2)
    d.rectangle((x1 + 16, y1 + 14, x2 - 16, y1 + 46), fill="#f1e7ca")
    d.text((x1 + 26, y1 + 20), title, font=font(18, True), fill="#4a3418")
    y = y1 + 66
    for line in lines:
        fnt = cjk_font(24) if any(ord(ch) > 127 for ch in line) else font(21)
        d.text((x1 + 24, y), line, font=fnt, fill="#2c2418")
        y += 34
    for yy in range(y + 6, y2 - 20, 24):
        d.line((x1 + 24, yy, x2 - 24, yy), fill="#eadfbe", width=2)


def light_glyph(
    base: Image.Image,
    box: Tuple[int, int, int, int],
    label: str,
    path: Path | None,
    *,
    char: str | None = None,
    sublabel: str = "",
) -> None:
    d = ImageDraw.Draw(base)
    x1, y1, x2, y2 = box
    d.rounded_rectangle(box, radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((x1 + 18, y1 + 16), label, font=font(19, True), fill="#0f172a")
    glyph_size = min(x2 - x1 - 52, y2 - y1 - 94)
    gx = x1 + (x2 - x1 - glyph_size) // 2
    gy = y1 + 54
    if path is not None and path.exists():
        glyph = load_svg(path, glyph_size)
        base.paste(glyph, (gx, gy), glyph)
    elif char:
        fnt = cjk_font(glyph_size)
        bbox = d.textbbox((0, 0), char, font=fnt)
        d.text((x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2, gy - 12), char, font=fnt, fill="#111827")
    if sublabel:
        centered_text(d, (x1 + 8, y2 - 36, x2 - 8, y2 - 8), sublabel, font(16), "#475467")


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


def retinal_flow_paradigm() -> None:
    """Render the implemented read-predict-write-reread paradigm and its receipt."""
    img = Image.new("RGB", (2200, 1320), "#f8fafc")
    d = ImageDraw.Draw(img)
    navy = "#102a43"
    blue = "#0b6fa4"
    green = "#16835b"
    amber = "#b45f06"
    red = "#b42318"
    grey = "#475467"

    d.text((80, 45), "Retinal Flow Language Model", font=font(52, True), fill=navy)
    d.text(
        (82, 112),
        "Read visible fixations, predict a visual distribution, write ink, then read the model's own ink.",
        font=font(27),
        fill=grey,
    )

    # Strict boundary banner.
    d.rounded_rectangle((80, 170, 2120, 235), radius=8, fill="#e8f4f8", outline="#8ec9d6", width=2)
    centered_text(
        d,
        (95, 172, 2105, 233),
        "STUDENT BOUNDARY   pixels in -> continuous visual state -> pixels out   |   no tokenizer, Unicode IDs, OCR, codebook, or external LM",
        font(22, True),
        navy,
    )

    # Reading path.
    d.text((80, 278), "A  READ AND PREDICT", font=font(28, True), fill=blue)
    rounded(d, (80, 330, 450, 650), "#ffffff", "#8ebbd2", radius=8)
    d.text((105, 352), "Ordered image fixations", font=font(24, True), fill=navy)
    glyphs = ["天", "地", "玄", "黃", "宇", "宙"]
    for i, glyph in enumerate(glyphs):
        row, col = divmod(i, 3)
        x = 108 + col * 105
        y = 420 + row * 105
        d.rounded_rectangle((x, y, x + 82, y + 82), radius=5, fill="#fffdf8", outline="#94a3b8", width=2)
        centered_text(d, (x, y - 2, x + 82, y + 82), glyph, cjk_font(50), "#111827")
    d.text((108, 615), "x_t in [0,1]^(32 x 32)", font=font(18), fill=grey)

    rounded(d, (560, 330, 900, 650), "#ecfdf3", "#58a889", radius=8)
    d.text((595, 352), "Foveal retina", font=font(24, True), fill=navy)
    for r in range(4):
        for c in range(4):
            shade = 65 + 25 * ((r + c) % 3)
            fill = (shade, 130 + shade // 3, 150 + shade // 4)
            d.rectangle((615 + c * 52, 430 + r * 42, 655 + c * 52, 460 + r * 42), fill=fill)
    d.text((615, 610), "z_t = R(x_t), 192-D", font=font(19), fill=grey)

    rounded(d, (1010, 330, 1370, 650), "#fff7ed", "#d99a52", radius=8)
    d.text((1044, 352), "Recurrent visual field", font=font(24, True), fill=navy)
    for i in range(3):
        d.rounded_rectangle((1065 + i * 65, 435, 1165 + i * 65, 545), radius=8, fill="#ffffff", outline="#d97706", width=3)
        centered_text(d, (1065 + i * 65, 435, 1165 + i * 65, 545), f"GRU\n{i + 1}", font(20, True), amber)
    d.text((1060, 610), "h_t = G(h_(t-1), z_t), 384-D", font=font(18), fill=grey)

    rounded(d, (1480, 330, 2120, 650), "#f0f9ff", "#3b91b4", radius=8)
    d.text((1515, 352), "Visual compatibility energy", font=font(24, True), fill=navy)
    d.text((1518, 410), "E(h_t, z_t, R(candidate image))", font=font(22), fill=blue)
    d.text((1518, 462), "Scores arbitrary image candidates", font=font(21), fill=grey)
    d.text((1518, 502), "Multi-positive visual NCE", font=font(21), fill=grey)
    d.text((1518, 542), "Cross-font views share a neighborhood", font=font(21), fill=grey)
    d.text((1518, 600), "No finite character output table", font=font(21, True), fill=green)

    arrow_any(d, (450, 490), (555, 490), blue)
    arrow_any(d, (900, 490), (1005, 490), blue)
    arrow_any(d, (1370, 490), (1475, 490), blue)

    # Writing and rereading path.
    d.text((80, 715), "B  WRITE, REREAD, AND CLOSE THE LOOP", font=font(28, True), fill=green)
    rounded(d, (80, 770, 530, 1060), "#ffffff", "#8ebbd2", radius=8)
    d.text((113, 793), "Conditional rectified flow", font=font(24, True), fill=navy)
    d.text((112, 855), "y_tau = (1-tau) x_(t+1) + tau noise", font=font(20), fill=grey)
    d.text((112, 900), "v = F(y_tau, tau, h_t, z_t)", font=font(22, True), fill=green)
    d.text((112, 950), "32 x 32 continuous ink", font=font(21), fill=grey)
    d.text((112, 990), "high-noise training prevents copying", font=font(20), fill=grey)

    rounded(d, (650, 770, 1000, 1060), "#fff7ed", "#d99a52", radius=8)
    d.text((686, 793), "Candidate ink images", font=font(24, True), fill=navy)
    for i in range(4):
        x = 692 + (i % 2) * 130
        y = 858 + (i // 2) * 88
        d.rounded_rectangle((x, y, x + 88, y + 72), radius=5, fill="#ffffff", outline="#94a3b8", width=2)
        # Abstract continuous strokes, deliberately not symbolic labels.
        d.line((x + 18, y + 51, x + 45, y + 18, x + 68, y + 50), fill="#1f2937", width=5 + i)
        d.line((x + 28, y + 35, x + 66, y + 35), fill="#1f2937", width=4)

    rounded(d, (1120, 770, 1515, 1060), "#ecfdf3", "#58a889", radius=8)
    d.text((1156, 793), "Reread and select", font=font(24, True), fill=navy)
    d.text((1154, 858), "z_hat = R(x_hat)", font=font(22, True), fill=green)
    d.text((1154, 908), "arg max E(h_t, z_t, z_hat)", font=font(21), fill=grey)
    d.text((1154, 958), "write-read cycle loss", font=font(21), fill=grey)
    d.text((1154, 998), "chosen pixels become next input", font=font(19), fill=grey)

    rounded(d, (1640, 770, 2120, 1060), "#fef2f2", "#d65a50", radius=8)
    d.text((1675, 793), "Autonomous visual trajectory", font=font(24, True), fill=navy)
    d.text((1680, 860), "x_1 -> x_2 -> ... -> x_T", font=font(23, True), fill=red)
    d.text((1680, 910), "The model observes its own marks", font=font(20), fill=grey)
    d.text((1680, 952), "V6: ink stable, text unreadable", font=font(19, True), fill=red)
    d.text((1680, 992), "Next: context margin + sampled identity", font=font(17), fill=navy)

    arrow_any(d, (530, 915), (645, 915), green)
    arrow_any(d, (1000, 915), (1115, 915), green)
    arrow_any(d, (1515, 915), (1635, 915), green)
    arrow_any(d, (1875, 770), (1245, 695), red, width=3)
    d.text((1450, 684), "visual feedback", font=font(18, True), fill=red)

    # Measured receipt and acceptance gate.
    rounded(d, (80, 1120, 2120, 1265), "#ffffff", "#cbd5e1", radius=8)
    d.text((108, 1143), "Measured one-RTX-4090 receipt", font=font(23, True), fill=navy)
    d.text((108, 1192), "11.69M parameters  |  2.576 GiB train peak  |  21.5 generated cells/s", font=font(20), fill=grey)
    d.text((875, 1143), "Vision", font=font(22, True), fill=green)
    d.text((875, 1192), "98.18% oracle top-1", font=font(21), fill=grey)
    d.text((1235, 1143), "Language", font=font(22, True), fill=red)
    d.text((1235, 1182), "full 1.20% < last 1.69%", font=font(18), fill=grey)
    d.text((1235, 1213), "< unigram 1.86% << bigram 13.58%", font=font(17), fill=grey)
    d.text((1770, 1143), "VERDICT", font=font(22, True), fill=red)
    d.text((1770, 1192), "V6 REJECTED", font=font(22, True), fill=red)

    img.save(FIG / "retinal_flow_paradigm.png", quality=95)


def predictive_visual_field_paradigm() -> None:
    """Render the post-V7 image-native state-and-rendering separation."""
    img = Image.new("RGB", (2400, 1440), "#f8fafc")
    d = ImageDraw.Draw(img)
    navy = "#102a43"
    blue = "#0b6fa4"
    green = "#16835b"
    amber = "#b45f06"
    red = "#b42318"
    grey = "#475467"

    d.text((80, 42), "Predictive Visual Field", font=font(54, True), fill=navy)
    d.text(
        (82, 112),
        "Language evolves as a distribution over continuous retinal states; a separate visual actuator writes each state as ink.",
        font=font(27),
        fill=grey,
    )
    d.rounded_rectangle((80, 172, 2320, 240), radius=8, fill="#e8f4f8", outline="#8ec9d6", width=2)
    centered_text(
        d,
        (95, 174, 2305, 238),
        "STRICT STUDENT: writing pixels -> continuous visual dynamics -> writing pixels   |   no tokens, OCR, Unicode IDs, codebook, or character lookup",
        font(22, True),
        navy,
    )

    d.text((80, 282), "1  SEE AND REMEMBER", font=font(28, True), fill=blue)
    rounded(d, (80, 335, 440, 645), "#ffffff", "#8ebbd2", radius=8)
    d.text((112, 360), "Visible writing", font=font(25, True), fill=navy)
    for index, glyph in enumerate(["天", "地", "玄", "黃", "宇", "宙"]):
        row, col = divmod(index, 3)
        x = 112 + col * 99
        y = 430 + row * 92
        d.rectangle((x, y, x + 76, y + 72), fill="#fffdf8", outline="#94a3b8", width=2)
        centered_text(d, (x, y - 3, x + 76, y + 72), glyph, cjk_font(45), "#111827")
    d.text((110, 610), "x_t: arbitrary image fixation", font=font(18), fill=grey)

    rounded(d, (545, 335, 875, 645), "#ecfdf3", "#58a889", radius=8)
    d.text((578, 360), "Retinal manifold", font=font(25, True), fill=navy)
    for r in range(5):
        for c in range(5):
            shade = 55 + 19 * ((r * 2 + c) % 4)
            d.ellipse((605 + c * 43, 430 + r * 34, 625 + c * 43, 450 + r * 34), fill=(shade, 145, 172))
    d.text((594, 610), "z_t = R(x_t), continuous", font=font(18), fill=grey)

    rounded(d, (980, 335, 1395, 645), "#fff7ed", "#d99a52", radius=8)
    d.text((1015, 360), "Causal visual field", font=font(25, True), fill=navy)
    for i, label in enumerate(("fast", "line", "page")):
        y = 430 + i * 58
        d.rounded_rectangle((1035, y, 1338, y + 42), radius=6, fill="#ffffff", outline="#d97706", width=2)
        d.text((1060, y + 7), f"{label} predictive state", font=font(19, i == 2), fill=amber)
    d.text((1018, 610), "h_t = C(z_1, ..., z_t)", font=font(18), fill=grey)

    arrow_any(d, (440, 490), (540, 490), blue)
    arrow_any(d, (875, 490), (975, 490), blue)

    d.text((1510, 282), "2  IMAGINE THE NEXT VISUAL STATE", font=font(28, True), fill=green)
    rounded(d, (1510, 335, 2320, 645), "#f0fdf4", "#58a889", radius=8)
    d.text((1548, 360), "Continuous retinal flow", font=font(25, True), fill=navy)
    d.text((1548, 412), "q_tau = (1-tau) z_(t+1) + tau epsilon", font=font(21), fill=grey)
    for i in range(7):
        x = 1580 + i * 94
        y = 525 - int(52 * math.sin(i / 6 * math.pi))
        radius = 17 + i
        color = "#8ecae6" if i < 3 else "#39a278"
        d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#ffffff", width=2)
        if i:
            arrow_any(d, (x - 74, y + (10 if i < 4 else -4)), (x - radius - 5, y), green, width=3)
    d.text(
        (1550, 577),
        "noise -> sampled image-derived state z_hat_(t+1)",
        font=font(20, True),
        fill=green,
    )
    d.text((1550, 611), "no nearest-glyph lookup", font=font(18, True), fill=green)
    arrow_any(d, (1395, 490), (1505, 490), green)

    d.text((80, 710), "3  WRITE THE PLAN, THEN SEE IT AGAIN", font=font(28, True), fill=green)
    rounded(d, (80, 765, 610, 1085), "#ffffff", "#8ebbd2", radius=8)
    d.text((115, 790), "Visual actuator", font=font(25, True), fill=navy)
    d.text((115, 850), "pixel flow conditioned on:", font=font(21), fill=grey)
    d.text((115, 895), "h_t + z_hat_(t+1) + local style", font=font(22, True), fill=green)
    d.text((115, 950), "The writer draws an intended visual state;", font=font(19), fill=grey)
    d.text((115, 986), "it no longer discovers language and strokes", font=font(19), fill=grey)
    d.text((115, 1022), "inside the same denoising operation.", font=font(19), fill=grey)

    rounded(d, (735, 765, 1115, 1085), "#fff7ed", "#d99a52", radius=8)
    d.text((770, 790), "Continuous ink", font=font(25, True), fill=navy)
    for i in range(4):
        x = 782 + (i % 2) * 145
        y = 866 + (i // 2) * 92
        d.rectangle((x, y, x + 100, y + 74), fill="#ffffff", outline="#94a3b8", width=2)
        d.line((x + 18, y + 54, x + 46, y + 17, x + 78, y + 55), fill="#1f2937", width=5)
        d.line((x + 28, y + 36, x + 76, y + 36), fill="#1f2937", width=4 + i)

    rounded(d, (1240, 765, 1660, 1085), "#ecfdf3", "#58a889", radius=8)
    d.text((1274, 790), "Reread and verify", font=font(25, True), fill=navy)
    d.text((1275, 858), "z_read = R(x_hat)", font=font(22, True), fill=green)
    d.text((1275, 910), "cycle: z_read ~= z_hat", font=font(21), fill=grey)
    d.text((1275, 962), "pixels become the next input", font=font(20), fill=grey)
    d.text((1275, 1014), "unknown and ancient forms stay images", font=font(18), fill=grey)

    rounded(d, (1785, 765, 2320, 1085), "#fef2f2", "#d65a50", radius=8)
    d.text((1820, 790), "What V7 falsified", font=font(25, True), fill=red)
    d.text((1820, 850), "One pixel flow was asked to learn", font=font(20), fill=grey)
    d.text((1820, 888), "both linguistic identity and rendering.", font=font(20), fill=grey)
    d.text((1820, 942), "2.31% top-1 > 1.86% unigram", font=font(19, True), fill=green)
    d.text((1820, 980), "but normalized context gain = -0.215", font=font(19, True), fill=red)
    d.text((1820, 1018), "and autonomous writing is unreadable", font=font(19, True), fill=red)

    arrow_any(d, (610, 920), (730, 920), green)
    arrow_any(d, (1115, 920), (1235, 920), green)
    arrow_any(d, (1660, 920), (1780, 920), red)
    arrow_any(d, (1450, 765), (1190, 690), green, width=3)
    d.text((1215, 668), "visual feedback", font=font(18, True), fill=green)

    rounded(d, (80, 1150, 2320, 1370), "#ffffff", "#cbd5e1", radius=8)
    d.text((112, 1176), "PVF staged proof sequence", font=font(25, True), fill=navy)
    gates = [
        ("A", "visual-state core", "V16 passes last-only + unigram"),
        ("B", "causal language", "6.26% < 13.14% bigram"),
        ("C", "rendering actuator", "reread must match plan"),
        ("D", "closed visual loop", "32 readable cells"),
    ]
    for i, (tag, title, evidence) in enumerate(gates):
        x1 = 110 + i * 545
        d.ellipse((x1, 1245, x1 + 54, 1299), fill=navy)
        centered_text(d, (x1, 1245, x1 + 54, 1299), tag, font(22, True), "#ffffff")
        d.text((x1 + 70, 1238), title, font=font(20, True), fill=navy)
        d.text((x1 + 70, 1278), evidence, font=font(17), fill=grey)
        if i < len(gates) - 1:
            arrow_any(d, (x1 + 465, 1272), (x1 + 530, 1272), "#94a3b8", width=3)

    img.save(FIG / "predictive_visual_field_paradigm.png", quality=95)


def predictive_visual_field_v15_result() -> None:
    """Render the measured V15 architecture and frozen state-language result."""
    img = Image.new("RGB", (2400, 1500), "#f8fafc")
    d = ImageDraw.Draw(img)
    navy = "#102a43"
    blue = "#0b6fa4"
    green = "#16835b"
    amber = "#b45f06"
    red = "#b42318"
    grey = "#475467"
    light = "#e2e8f0"

    d.text((80, 45), "PVF V15: Language Signal Learned From Writing Images", font=font(51, True), fill=navy)
    d.text(
        (82, 116),
        "A 10.47M-parameter visual student learns causal next-state structure on one RTX 4090; pixels remain the only learned input boundary.",
        font=font(25),
        fill=grey,
    )
    d.rounded_rectangle((80, 173, 2320, 242), radius=8, fill="#e8f4f8", outline="#8ec9d6", width=2)
    centered_text(
        d,
        (95, 175, 2305, 240),
        "NO TOKENS  |  NO OCR  |  NO UNICODE IDs  |  NO CHARACTER LABELS  |  NO CODEBOOK  |  NO CANDIDATE CLASSIFIER",
        font(22, True),
        navy,
    )

    # Image-native causal path.
    d.text((80, 280), "A   IMAGE-NATIVE CAUSAL PATH", font=font(27, True), fill=blue)
    d.rounded_rectangle((80, 330, 565, 665), radius=8, fill="#ffffff", outline="#8ebbd2", width=2)
    d.text((112, 355), "Writing-image history", font=font(24, True), fill=navy)
    glyphs = ["學", "而", "時", "習", "之"]
    for index, glyph in enumerate(glyphs):
        x = 112 + index * 83
        d.rectangle((x, 430, x + 68, 502), fill="#fffdf8", outline="#94a3b8", width=2)
        centered_text(d, (x, 423, x + 68, 502), glyph, cjk_font(44), "#111827")
        centered_text(d, (x, 515, x + 68, 548), f"x{index + 1}", font(15), grey)
    d.text((112, 585), "observations: 32 x 32 continuous pixels", font=font(18), fill=grey)
    d.text((112, 620), "sequence length: 48 visual fixations", font=font(18), fill=grey)

    d.rounded_rectangle((670, 330, 995, 665), radius=8, fill="#ecfdf3", outline="#58a889", width=2)
    d.text((704, 355), "Frozen retina", font=font(24, True), fill=navy)
    center = (833, 482)
    d.ellipse((744, 393, 922, 571), outline=green, width=4)
    d.arc((760, 424, 906, 541), 10, 170, fill="#5ab28d", width=3)
    d.arc((760, 424, 906, 541), 190, 350, fill="#5ab28d", width=3)
    d.line((833, 395, 833, 568), fill="#5ab28d", width=3)
    for angle in (0.25, 1.05, 1.9, 2.8, 3.8, 4.7, 5.55):
        px = int(center[0] + 75 * math.cos(angle))
        py = int(center[1] + 75 * math.sin(angle))
        d.ellipse((px - 8, py - 8, px + 8, py + 8), fill=green)
    centered_text(d, (704, 585, 962, 630), "z_t in unit visual sphere", font(18, True), green)

    d.rounded_rectangle((1100, 330, 1450, 665), radius=8, fill="#fff7ed", outline="#d99a52", width=2)
    d.text((1133, 355), "Causal visual field", font=font(24, True), fill=navy)
    for index, label in enumerate(("retinal state", "history memory", "causal condition")):
        y = 420 + index * 64
        d.rounded_rectangle((1150, y, 1400, y + 43), radius=6, fill="#ffffff", outline="#d97706", width=2)
        centered_text(d, (1160, y, 1390, y + 43), label, font(18, index == 2), amber)
    centered_text(d, (1130, 605, 1420, 640), "h_t = GRU(z_1, ..., z_t)", font(17), grey)

    arrow_any(d, (565, 495), (665, 495), blue)
    arrow_any(d, (995, 495), (1095, 495), blue)

    # Two complementary continuous heads.
    d.rounded_rectangle((1570, 285, 2320, 475), radius=8, fill="#edf9f2", outline="#58a889", width=2)
    d.text((1605, 310), "Deterministic visual proposal", font=font(24, True), fill=navy)
    d.text((1605, 362), "mu_t = normalize P([h_t, z_t])", font=font(20), fill=green)
    d.text((1605, 406), "low-variance next-language mode", font=font(19, True), fill=green)

    d.rounded_rectangle((1570, 505, 2320, 695), radius=8, fill="#eef5ff", outline="#7aa7cf", width=2)
    d.text((1605, 530), "Hyperspherical visual state flow", font=font(24, True), fill=navy)
    d.text((1605, 582), "noise -> geodesic tangent field -> visual states", font=font(19), fill=blue)
    d.text((1605, 626), "multimodal uncertainty and form variation", font=font(19, True), fill=blue)

    arrow_any(d, (1450, 456), (1565, 382), green)
    arrow_any(d, (1450, 535), (1565, 598), blue)

    # Frozen benchmark bars.
    d.text((80, 742), "B   FROZEN 512-FORM NEXT-IMAGE EVALUATION", font=font(27, True), fill=blue)
    d.text((82, 790), "Evaluator labels score image-derived states only; they never enter the student.", font=font(20), fill=grey)
    chart_left = 350
    chart_right = 1440
    chart_top = 880
    scale_max = 14.0
    for tick in (0, 2, 4, 6, 8, 10, 12, 14):
        x = chart_left + int((chart_right - chart_left) * tick / scale_max)
        d.line((x, chart_top - 8, x, 1300), fill=light, width=2)
        centered_text(d, (x - 35, chart_top - 48, x + 35, chart_top - 12), f"{tick}%", font(15), grey)

    metrics = [
        ("random dynamics", 0.168, "#94a3b8"),
        ("unigram", 1.734, amber),
        ("last image only", 4.418, "#4f7cac"),
        ("PVF V15 proposal", 5.872, green),
        ("symbolic bigram", 13.143, red),
    ]
    for index, (label, value, color) in enumerate(metrics):
        y = chart_top + index * 83
        d.text((82, y + 10), label, font=font(19, label == "PVF V15 proposal"), fill=navy)
        width = max(4, int((chart_right - chart_left) * value / scale_max))
        d.rounded_rectangle((chart_left, y, chart_left + width, y + 52), radius=5, fill=color)
        d.text((chart_left + width + 14, y + 10), f"{value:.3f}%", font=font(18, True), fill=color)

    # Receipt and acceptance on the right.
    d.rounded_rectangle((1540, 790, 2320, 1050), radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((1575, 820), "Single-GPU receipt", font=font(24, True), fill=navy)
    receipt = [
        "10.47M parameters / 9.14M trainable",
        "1.181 GiB peak allocated CUDA memory",
        "about 100-106 sequences per second",
        "one RTX 4090, BF16",
    ]
    for index, line in enumerate(receipt):
        d.ellipse((1580, 883 + index * 38, 1592, 895 + index * 38), fill=blue)
        d.text((1610, 874 + index * 38), line, font=font(18), fill=grey)

    d.rounded_rectangle((1540, 1080, 2320, 1325), radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((1575, 1110), "What the evidence says", font=font(24, True), fill=navy)
    statements = [
        ("PASS", "full history > last image > unigram", green),
        ("PASS", "normalized context gain +0.0707", green),
        ("PASS", "state flow 3.41% > 2.68% last-only", green),
        ("OPEN", "5.87% remains below 13.14% bigram", red),
        ("OPEN", "state-conditioned pixel actuator", red),
    ]
    for index, (tag, line, color) in enumerate(statements):
        y = 1165 + index * 31
        d.text((1578, y), tag, font=font(16, True), fill=color)
        d.text((1652, y), line, font=font(17), fill=grey)

    d.rounded_rectangle((80, 1370, 2320, 1440), radius=8, fill="#fff7ed", outline="#d99a52", width=2)
    centered_text(
        d,
        (100, 1372, 2300, 1438),
        "Result: image-only causal language learning is feasible on one consumer GPU. General language and readable image output remain falsifiable next gates.",
        font(21, True),
        navy,
    )
    img.save(FIG / "predictive_visual_field_v15_result.png", quality=95)


def predictive_visual_field_v16_result() -> None:
    """Render the measured V16 memory architecture and frozen result."""
    img = Image.new("RGB", (2400, 1500), "#f8fafc")
    d = ImageDraw.Draw(img)
    navy = "#102a43"
    blue = "#0b6fa4"
    green = "#16835b"
    amber = "#b45f06"
    red = "#b42318"
    grey = "#475467"
    light = "#e2e8f0"

    d.text((80, 45), "PVF V16: Causal Visual Memory on One RTX 4090", font=font(50, True), fill=navy)
    d.text(
        (82, 116),
        "A 16.47M-parameter student reads writing pixels, forms continuous visual states, and predicts the next state without a symbolic vocabulary.",
        font=font(24),
        fill=grey,
    )
    d.rounded_rectangle((80, 173, 2320, 242), radius=8, fill="#e8f4f8", outline="#8ec9d6", width=2)
    centered_text(
        d,
        (95, 175, 2305, 240),
        "NO TOKENS  |  NO OCR  |  NO UNICODE IDs  |  NO CHARACTER LABELS  |  NO CODEBOOK  |  NO CLASSIFIER",
        font(22, True),
        navy,
    )

    d.text((80, 280), "A   MULTISCALE IMAGE-NATIVE MEMORY", font=font(27, True), fill=blue)
    d.rounded_rectangle((80, 330, 500, 670), radius=8, fill="#ffffff", outline="#8ebbd2", width=2)
    d.text((112, 355), "Writing-image history", font=font(23, True), fill=navy)
    glyphs = ["學", "而", "時", "習", "之"]
    for index, glyph in enumerate(glyphs):
        x = 112 + index * 70
        d.rectangle((x, 426, x + 58, 492), fill="#fffdf8", outline="#94a3b8", width=2)
        centered_text(d, (x, 420, x + 58, 492), glyph, cjk_font(39), "#111827")
    d.text((112, 535), "32 x 32 continuous pixels", font=font(18), fill=grey)
    d.text((112, 575), "48 causal visual fixations", font=font(18), fill=grey)
    d.text((112, 615), "eight independent fonts", font=font(18), fill=grey)

    d.rounded_rectangle((610, 330, 920, 670), radius=8, fill="#ecfdf3", outline="#58a889", width=2)
    d.text((650, 355), "Frozen retina", font=font(23, True), fill=navy)
    center = (765, 485)
    d.ellipse((680, 400, 850, 570), outline=green, width=4)
    for angle in (0.2, 0.95, 1.7, 2.55, 3.4, 4.25, 5.2):
        px = int(center[0] + 70 * math.cos(angle))
        py = int(center[1] + 70 * math.sin(angle))
        d.ellipse((px - 8, py - 8, px + 8, py + 8), fill=green)
    centered_text(d, (645, 590, 885, 630), "continuous z_t", font(18, True), green)

    d.rounded_rectangle((1030, 300, 1690, 700), radius=8, fill="#fff7ed", outline="#d99a52", width=2)
    d.text((1065, 325), "Residual causal visual memory", font=font(24, True), fill=navy)
    d.rounded_rectangle((1080, 390, 1290, 445), radius=6, fill="#ffffff", outline="#d97706", width=2)
    centered_text(d, (1090, 390, 1280, 445), "GRU base", font(19, True), amber)
    block_labels = ["local d=1", "local d=2", "local d=4"]
    for index, label in enumerate(block_labels):
        y = 480 + index * 61
        d.rounded_rectangle((1080, y, 1325, y + 46), radius=6, fill="#ffffff", outline="#d97706", width=2)
        centered_text(d, (1090, y, 1315, y + 46), label, font(17, True), amber)
        d.rounded_rectangle((1370, y, 1625, y + 46), radius=6, fill="#ffffff", outline="#0b6fa4", width=2)
        centered_text(d, (1380, y, 1615, y + 46), "global causal attention", font(15), blue)
        arrow_any(d, (1327, y + 23), (1367, y + 23), grey, width=2)
    d.text((1340, 400), "near-zero gated correction", font=font(17), fill=grey)
    d.text((1340, 435), "sigmoid(g) = 0.0180", font=font(17, True), fill=amber)
    arrow_any(d, (500, 500), (605, 500), blue)
    arrow_any(d, (920, 500), (1025, 500), blue)

    d.rounded_rectangle((1800, 330, 2320, 670), radius=8, fill="#eef5ff", outline="#7aa7cf", width=2)
    d.text((1835, 355), "Continuous predictions", font=font(23, True), fill=navy)
    d.rounded_rectangle((1845, 425, 2275, 495), radius=6, fill="#ffffff", outline="#58a889", width=2)
    centered_text(d, (1860, 425, 2260, 495), "deterministic visual proposal", font(18, True), green)
    d.rounded_rectangle((1845, 535, 2275, 605), radius=6, fill="#ffffff", outline="#7aa7cf", width=2)
    centered_text(d, (1860, 535, 2260, 605), "hyperspherical state flow", font(18, True), blue)
    arrow_any(d, (1690, 500), (1795, 500), blue)

    d.text((80, 755), "B   ONE-SHOT FROZEN 512-FORM EVALUATION", font=font(27, True), fill=blue)
    d.text((82, 802), "1,788 held-out contexts; evaluator labels score continuous image states only.", font=font(20), fill=grey)
    chart_left = 380
    chart_right = 1480
    chart_top = 885
    scale_max = 14.0
    for tick in (0, 2, 4, 6, 8, 10, 12, 14):
        x = chart_left + int((chart_right - chart_left) * tick / scale_max)
        d.line((x, chart_top - 8, x, 1340), fill=light, width=2)
        centered_text(d, (x - 35, chart_top - 48, x + 35, chart_top - 12), f"{tick}%", font(15), grey)

    metrics = [
        ("chance", 0.195, "#94a3b8"),
        ("unigram", 1.734, amber),
        ("last image only", 3.971, "#4f7cac"),
        ("PVF V15 full history", 5.872, "#6aa58a"),
        ("PVF V16 full history", 6.264, green),
        ("symbolic bigram", 13.143, red),
    ]
    for index, (label, value, color) in enumerate(metrics):
        y = chart_top + index * 70
        d.text((82, y + 8), label, font=font(18, label == "PVF V16 full history"), fill=navy)
        width = max(4, int((chart_right - chart_left) * value / scale_max))
        d.rounded_rectangle((chart_left, y, chart_left + width, y + 45), radius=5, fill=color)
        d.text((chart_left + width + 13, y + 8), f"{value:.3f}%", font=font(17, True), fill=color)

    d.rounded_rectangle((1570, 805, 2320, 1080), radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((1605, 835), "Measured causal evidence", font=font(24, True), fill=navy)
    evidence = [
        ("112 / 1,788", "correct; V15 was 105", green),
        ("+2.293 pp", "full history over last image", green),
        ("+0.0773", "normalized context log-probability", green),
        ("3.691%", "sampled state flow; last 3.244%", blue),
        ("13.143%", "symbolic bigram remains stronger", red),
    ]
    for index, (value, label, color) in enumerate(evidence):
        y = 895 + index * 37
        d.text((1608, y), value, font=font(17, True), fill=color)
        d.text((1745, y), label, font=font(17), fill=grey)

    d.rounded_rectangle((1570, 1110, 2320, 1340), radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((1605, 1140), "Single-GPU receipt", font=font(24, True), fill=navy)
    receipt = [
        "16.47M total / 15.14M trainable parameters",
        "6.00M parameters in causal visual memory",
        "1.479 GiB peak allocated CUDA memory",
        "92-114 sequences/s, BF16, one RTX 4090",
        "0 classifier / 0 pixel-actuator parameters",
    ]
    for index, line in enumerate(receipt):
        d.ellipse((1608, 1195 + index * 29, 1619, 1206 + index * 29), fill=blue)
        d.text((1638, 1187 + index * 29), line, font=font(16), fill=grey)

    d.rounded_rectangle((80, 1380, 2320, 1450), radius=8, fill="#fff7ed", outline="#d99a52", width=2)
    centered_text(
        d,
        (100, 1382, 2300, 1448),
        "Result: visual language signal scales cheaply, but seven extra correct contexts do not solve language; bigram and readable-pixel gates remain open.",
        font(20, True),
        navy,
    )
    img.save(FIG / "predictive_visual_field_v16_result.png", quality=95)


def anchor_identity_v7_result() -> None:
    """Render a matched V6/V7 autonomous comparison with measured receipts."""
    img = Image.new("RGB", (2200, 1120), "#f8fafc")
    d = ImageDraw.Draw(img)
    navy = "#102a43"
    red = "#b42318"
    green = "#16835b"
    grey = "#475467"
    d.text((70, 42), "V7 Anchor Identity: Better Signal, Still Not Language", font=font(48, True), fill=navy)
    d.text((72, 110), "Same prompt, seed, model size, sampler, and 32-cell horizon", font=font(25), fill=grey)

    root = ROOT.parent.parent
    rows = [
        (
            "V6 closed loop",
            root / "artifacts/retinal_flow_chinese_mvp_v6_closed_loop/autonomous_32_step_5200/complete_page.png",
            "top-1 1.20%   normalized context -0.907   generated signal +0.0077",
            "ink ratio 1.168   sparse 18.75%   unreadable",
        ),
        (
            "V7 anchor + sampled identity",
            root / "artifacts/retinal_flow_chinese_mvp_v7_anchor_identity/autonomous_32_step_5800/complete_page.png",
            "top-1 2.31%   normalized context -0.215   generated signal +0.0303",
            "ink ratio 1.050   sparse 15.63%   unreadable",
        ),
    ]
    for index, (label, path, metric, rollout) in enumerate(rows):
        y = 205 + index * 405
        d.text((78, y), label, font=font(28, True), fill=navy)
        d.rounded_rectangle((70, y + 50, 2130, y + 270), radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
        if path.exists():
            page = Image.open(path).convert("L").convert("RGB")
            page.thumbnail((1980, 190), Image.Resampling.NEAREST)
            px = 110 + (1980 - page.width) // 2
            py = y + 65 + (190 - page.height) // 2
            img.paste(page, (px, py))
        else:
            centered_text(d, (100, y + 70, 2100, y + 250), "local autonomous artifact unavailable", font(24), grey)
        d.text((82, y + 292), metric, font=font(22), fill=green if index else grey)
        d.text((82, y + 333), rollout, font=font(22, True), fill=red)

    d.rounded_rectangle((70, 1010, 2130, 1080), radius=7, fill="#fef2f2", outline="#d65a50", width=2)
    centered_text(
        d,
        (85, 1012, 2115, 1078),
        "Verdict: V7 closes most of the calibrated context deficit, but dense pseudo-glyphs are not readable Chinese. Separate visual-state prediction from rendering.",
        font(21, True),
        red,
    )
    img.save(FIG / "anchor_identity_v7_result.png", quality=95)


def curriculum() -> None:
    img = Image.new("RGB", (1800, 1050), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text(
        (70, 45),
        "Single-4090 ILM: Minimal Working Path",
        font=font(46, True),
        fill="#101828",
    )
    d.text(
        (72, 112),
        "One visual substrate from ordinary prompts and book pages to generated answer pages",
        font=font(25),
        fill="#475467",
    )

    phases = [
        ("0", "Compose", "typed prompt + page\n-> one prompt canvas", "#e0f2fe"),
        ("1", "Read", "pixels -> global state\n+ 4 x 4 field", "#dcfce7"),
        ("2", "Predict", "history -> next\nvisual language state", "#fef3c7"),
        ("3", "Write", "state + field ->\nink motor plan", "#f3e8ff"),
        ("4", "Reread", "generated pixels ->\nsame retina", "#ffe4e6"),
        ("5", "Answer", "answer page image\n+ optional text", "#e0e7ff"),
    ]
    x0 = 80
    y = 190
    w = 250
    h = 240
    gap = 30
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
        (
            "ONE 4090",
            "Measured: V16 state core 1.479 GiB peak; V18 writer 0.778 GiB peak; V19 trains only a small residual.",
        ),
        (
            "FIXED GATES",
            "Beat last-only + unigram + bigram; pass dense writing; preserve 32 readable generated regions.",
        ),
        (
            "IMAGE NATIVE",
            "No runtime tokens, Unicode IDs, OCR transcript, glyph lookup, or external language model.",
        ),
    ]
    y2 = 555
    for i, (tag, text) in enumerate(lanes):
        top = y2 + i * 120
        rounded(d, (120, top, 1680, top + 88), "#f8fafc", "#cbd5e1", radius=12)
        d.text((155, top + 26), tag, font=font(25, True), fill="#0f172a")
        d.text((385, top + 29), text, font=font(21), fill="#334155")

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
    d.text((70, 45), "Example Output: Evolution of Chinese Character Zhong", font=font(42, True), fill="#101828")
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


def yan_paths() -> dict[str, Path]:
    data_root = Path("/home/lachlan/ProjectsLFS/incoder/data/historic/glyphs/言")
    return {
        "oracle": data_root / "oracle" / "J04903.svg",
        "bronze": data_root / "bronze" / "B02975.svg",
        "seal": data_root / "seal" / "S01648.svg",
    }


def draw_visual_code_grid(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], colors: list[str]) -> None:
    x1, y1, x2, y2 = box
    cols = 10
    rows = 7
    gap = 5
    cw = (x2 - x1 - gap * (cols - 1)) // cols
    ch = (y2 - y1 - gap * (rows - 1)) // rows
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) % len(colors)
            xx = x1 + c * (cw + gap)
            yy = y1 + r * (ch + gap)
            draw.rounded_rectangle((xx, yy, xx + cw, yy + ch), radius=5, fill=colors[idx], outline="#ffffff")


def visual_training_pipeline() -> None:
    img = Image.new("RGB", (2400, 1600), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "Training ILM-V: Language as Visual Corpus", font=font(54, True), fill="#0f172a")
    d.text(
        (74, 116),
        "Every source is converted into page/glyph images. The model learns to predict and generate visual latents, not BPE or word tokens.",
        font=font(27),
        fill="#475467",
    )

    cols = [
        (70, 185, 560, 1175, "#eff6ff", "#2563eb", "1  Visual Corpus"),
        (620, 185, 1110, 1175, "#f0fdf4", "#16a34a", "2  Canvas Builder"),
        (1170, 185, 1710, 1175, "#111827", "#38bdf8", "3  ILM-V Training Core"),
        (1770, 185, 2330, 1175, "#fff7ed", "#f97316", "4  Image Targets"),
    ]
    for x1, y1, x2, y2, fill, outline, title in cols:
        d.rounded_rectangle((x1, y1, x2, y2), radius=28, fill=fill, outline=outline, width=3)
        d.text((x1 + 28, y1 + 24), title, font=font(29, True), fill="#f8fafc" if fill == "#111827" else "#0f172a")

    # Column 1: data as pages/glyphs.
    mini_page(
        img,
        (105, 260, 525, 485),
        "Rendered corpus page",
        ["The river speaks in lines.", "漢字之源在形", "子曰：學而時習之。"],
        tint="#fffaf0",
    )
    mini_page(
        img,
        (105, 520, 525, 745),
        "Scanned book / manuscript",
        ["ink, margin, damage", "古書一頁，字有殘闕。", "layout is evidence"],
        tint="#f7efe1",
    )
    d.rounded_rectangle((105, 780, 525, 1088), radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((128, 805), "Local ziyuan glyph data", font=font(23, True), fill="#0f172a")
    paths = yan_paths()
    light_glyph(img, (130, 855, 250, 1035), "Oracle", paths["oracle"], sublabel="J04903")
    light_glyph(img, (268, 855, 388, 1035), "Bronze", paths["bronze"], sublabel="B02975")
    light_glyph(img, (406, 855, 500, 1035), "Seal", paths["seal"], sublabel="S01648")
    d.text((130, 1050), "unencoded historical signs\nstay as image patches", font=font(18), fill="#475467")

    # Column 2: rendering and normalization.
    d.rounded_rectangle((675, 275, 1055, 395), radius=16, fill="#ffffff", outline="#bbf7d0", width=2)
    centered_text(d, (685, 290, 1045, 380), "Text sources are rendered\nonto prompt/answer pages", font(24, True), "#14532d")
    d.rounded_rectangle((675, 455, 1055, 595), radius=16, fill="#ffffff", outline="#bbf7d0", width=2)
    centered_text(d, (685, 470, 1045, 575), "Image sources are cropped,\ndeskewed, tiled,\nand normalized", font(23, True), "#14532d")
    d.rounded_rectangle((675, 665, 1055, 885), radius=16, fill="#dcfce7", outline="#16a34a", width=2)
    d.text((710, 690), "Shared visual canvas", font=font(26, True), fill="#14532d")
    draw_visual_code_grid(d, (710, 748, 1020, 850), ["#166534", "#22c55e", "#86efac", "#fef3c7", "#0f766e"])
    d.text((710, 928), "Metadata stays outside\nthe language path:\nstage, crop box, source, mask region.", font=font(20), fill="#166534")
    arrow_any(d, (560, 520), (620, 520), "#64748b", 5)
    arrow_any(d, (1110, 520), (1170, 520), "#64748b", 5)
    arrow_any(d, (1710, 520), (1770, 520), "#64748b", 5)

    # Column 3: model core.
    d.rounded_rectangle((1225, 285, 1655, 1035), radius=28, fill="#0b1220", outline="#67e8f9", width=3)
    d.text((1265, 330), "Visual latent stack", font=font(29, True), fill="#e0f2fe")
    d.text((1265, 392), "VAE / VQ-VAE\n2D memory transformer\nmasked image generator\nlatent diffusion decoder", font=font(27), fill="#cbd5e1")
    d.rounded_rectangle((1265, 610, 1615, 795), radius=20, fill="#1e293b", outline="#38bdf8", width=2)
    draw_visual_code_grid(d, (1295, 642, 1585, 762), ["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6"])
    centered_text(d, (1268, 815, 1612, 890), "Visual units only:\nstrokes + patches\nlayout latents", font(24, True), "#fde68a")
    centered_text(d, (1268, 920, 1612, 990), "NO BPE\nNO hidden Unicode decoder", font(24, True), "#fecaca")

    # Column 4: image targets.
    mini_page(
        img,
        (1815, 265, 2285, 565),
        "Target answer page",
        ["言 YAN: speech / language", "English explanation paragraph", "中文說明與古文例句", "forms shown as glyph panels"],
        tint="#fffaf0",
    )
    d.rounded_rectangle((1815, 625, 2285, 1005), radius=18, fill="#ffffff", outline="#fed7aa", width=2)
    d.text((1840, 655), "Glyph-evolution target", font=font(27, True), fill="#7c2d12")
    light_glyph(img, (1845, 720, 1975, 945), "Oracle", paths["oracle"], sublabel="image")
    light_glyph(img, (1995, 720, 2125, 945), "Bronze", paths["bronze"], sublabel="image")
    light_glyph(img, (2145, 720, 2260, 945), "Modern", None, char="言", sublabel="U+8A00")
    d.text((1840, 1045), "Target is a PNG/page image first;\ntext labels are optional supervision.", font=font(20), fill="#9a3412")

    # Bottom loss band.
    d.rounded_rectangle((70, 1240, 2330, 1515), radius=30, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((105, 1268), "Loss mixture for one or two RTX 3090 GPUs", font=font(34, True), fill="#0f172a")
    losses = [
        ("Masked visual LM", "hide page patches\npredict visual latents", "#dbeafe"),
        ("Denoising", "blur/drop strokes\nrecover readable page", "#dcfce7"),
        ("Image instruction", "prompt image ->\nanswer image", "#fef3c7"),
        ("Glyph evolution", "modern glyph <->\nancient forms", "#fee2e2"),
        ("Auxiliary OCR", "readability metric only\nnot the main path", "#ede9fe"),
    ]
    lx = 105
    for title, body, fill in losses:
        d.rounded_rectangle((lx, 1335, lx + 405, 1475), radius=18, fill=fill, outline="#94a3b8", width=2)
        centered_text(d, (lx + 15, 1348, lx + 390, 1398), title, font(24, True), "#0f172a")
        centered_text(d, (lx + 20, 1402, lx + 385, 1462), body, font(21), "#334155")
        lx += 440

    img.save(FIG / "visual_language_training_pipeline.png", quality=95)


def draw_answer_page(base: Image.Image, box: Tuple[int, int, int, int]) -> None:
    d = ImageDraw.Draw(base)
    x1, y1, x2, y2 = box
    d.rounded_rectangle(box, radius=18, fill="#fffaf0", outline="#d6c7a1", width=3)
    d.rectangle((x1 + 34, y1 + 36, x2 - 34, y1 + 104), fill="#efe2bd")
    d.text((x1 + 56, y1 + 52), "言 YAN: speech, language, saying", font=cjk_font(31), fill="#3f2b12")
    line_text(
        d,
        (x1 + 58, y1 + 138),
        "English: YAN is the written idea of speech.\nIt links a mouth, sound, statement, and record.",
        font(24),
        "#2c2418",
        spacing=10,
    )
    line_text(
        d,
        (x1 + 58, y1 + 250),
        "中文：言，本義為言語、說話、辭令。\n古文：言者，心聲見於簡冊也。",
        cjk_font(26),
        "#2c2418",
        spacing=10,
    )
    d.line((x1 + 58, y1 + 365, x2 - 58, y1 + 365), fill="#d8c697", width=3)
    paths = yan_paths()
    gx = x1 + 58
    gy = y1 + 405
    card_w = 128
    gap = 18
    light_glyph(base, (gx, gy, gx + card_w, gy + 220), "Oracle", paths["oracle"], sublabel="J04903")
    light_glyph(base, (gx + (card_w + gap), gy, gx + (card_w + gap) + card_w, gy + 220), "Bronze", paths["bronze"], sublabel="B02975")
    light_glyph(base, (gx + 2 * (card_w + gap), gy, gx + 2 * (card_w + gap) + card_w, gy + 220), "Seal", paths["seal"], sublabel="S01648")
    light_glyph(base, (gx + 3 * (card_w + gap), gy, gx + 3 * (card_w + gap) + card_w, gy + 220), "Modern", None, char="言", sublabel="U+8A00")
    line_text(
        d,
        (x1 + 58, gy + 250),
        "Ancient forms are visual glyph regions.\nThey do not need a font table or codec.",
        font(20),
        "#5f4b24",
        spacing=8,
    )
    line_text(
        d,
        (x1 + 58, y2 - 92),
        "Output artifact: rendered page image first.\nA text layer may be extracted later.",
        font(20, True),
        "#7c2d12",
        spacing=8,
    )


def visual_inference_pipeline() -> None:
    img = Image.new("RGB", (2400, 1500), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((70, 45), "Inference ILM-V: Chat-Like UI, Image-First Output", font=font(54, True), fill="#0f172a")
    d.text(
        (74, 116),
        "Typed text and uploaded images are both turned into visual prompt canvases. The model returns a rendered page image first.",
        font=font(27),
        fill="#475467",
    )

    # Inputs.
    d.rounded_rectangle((80, 220, 600, 1280), radius=28, fill="#eff6ff", outline="#2563eb", width=3)
    d.text((115, 250), "Input interface", font=font(31, True), fill="#0f172a")
    mini_page(
        img,
        (125, 320, 555, 585),
        "Typed prompt rendered as image",
        ["Tell the evolution of 言.", "Use English + 中文 + 古文.", "Show forms not in fonts."],
        tint="#fffaf0",
    )
    mini_page(
        img,
        (125, 660, 555, 925),
        "Uploaded image prompt",
        ["book page crop / glyph photo", "oracle or cuneiform-like signs", "layout and damage preserved"],
        tint="#f7efe1",
    )
    d.rounded_rectangle((125, 990, 555, 1190), radius=18, fill="#dbeafe", outline="#60a5fa", width=2)
    centered_text(d, (150, 1015, 530, 1088), "No tokenization step\nis required before reading.", font(25, True), "#1e3a8a")
    centered_text(d, (150, 1105, 530, 1168), "Text input is simply\nanother visual page.", font(23), "#1e40af")

    # Canvas router.
    d.rounded_rectangle((710, 325, 1170, 1115), radius=28, fill="#f0fdf4", outline="#16a34a", width=3)
    d.text((745, 360), "Prompt canvas router", font=font(31, True), fill="#14532d")
    d.rounded_rectangle((760, 450, 1120, 640), radius=18, fill="#ffffff", outline="#86efac", width=2)
    centered_text(d, (790, 475, 1090, 545), "typed text ->\npage rasterizer", font(27, True), "#14532d")
    centered_text(d, (790, 555, 1090, 615), "font is an input renderer,\nnot the model language", font(21), "#166534")
    d.rounded_rectangle((760, 710, 1120, 900), radius=18, fill="#ffffff", outline="#86efac", width=2)
    centered_text(d, (790, 735, 1090, 810), "image upload ->\nvisual crop + tiling", font(27, True), "#14532d")
    centered_text(d, (790, 820, 1090, 875), "keeps unknown glyphs\nas pixels", font(21), "#166534")
    d.rounded_rectangle((780, 965, 1100, 1050), radius=16, fill="#dcfce7", outline="#16a34a", width=2)
    centered_text(d, (790, 970, 1090, 1045), "unified prompt image", font(25, True), "#14532d")
    arrow_any(d, (600, 450), (710, 515), "#64748b", 5)
    arrow_any(d, (600, 790), (710, 785), "#64748b", 5)

    # Model.
    d.rounded_rectangle((1250, 365, 1495, 1075), radius=34, fill="#111827", outline="#38bdf8", width=3)
    d.text((1284, 410), "ILM-V", font=font(45, True), fill="#e0f2fe")
    d.text((1284, 485), "visual encoder\n2D memory\nmasked image\ngenerator\nlatent decoder", font=font(24), fill="#cbd5e1")
    draw_visual_code_grid(d, (1285, 720, 1460, 850), ["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6"])
    centered_text(d, (1275, 895, 1470, 985), "internal state:\nvisual latent page", font(23, True), "#fde68a")
    arrow_any(d, (1170, 735), (1250, 735), "#64748b", 5)

    # Output page.
    d.rounded_rectangle((1570, 210, 2325, 1225), radius=30, fill="#fff7ed", outline="#f97316", width=3)
    d.text((1605, 245), "Image-first answer", font=font(32, True), fill="#7c2d12")
    draw_answer_page(img, (1615, 315, 2280, 1130))
    arrow_any(d, (1495, 735), (1570, 735), "#64748b", 5)

    # Optional conversion layer.
    d.rounded_rectangle((710, 1255, 2325, 1435), radius=26, fill="#ffffff", outline="#cbd5e1", width=2)
    d.text((745, 1282), "Optional post-processing after image generation", font=font(30, True), fill="#0f172a")
    line_text(
        d,
        (745, 1336),
        "OCR/transcoder can emit text only for computer-codec regions: English, Chinese, U+8A00.\nOracle, bronze, cuneiform-like, or unknown glyphs remain image assets with coordinates and provenance.",
        font(23),
        "#334155",
        spacing=8,
    )
    for x in range(1930, 1930 + 270, 28):
        d.line((x, 1225, x + 15, 1255), fill="#94a3b8", width=3)
    d.polygon([(2200, 1255), (2180, 1250), (2191, 1232)], fill="#94a3b8")

    img.save(FIG / "visual_language_inference_pipeline.png", quality=95)


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
    retinal_flow_paradigm()
    predictive_visual_field_paradigm()
    predictive_visual_field_v15_result()
    predictive_visual_field_v16_result()
    anchor_identity_v7_result()
    curriculum()
    zhong_evolution()
    yan_cover_hero()
    visual_training_pipeline()
    visual_inference_pipeline()
    aginti_loop()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
