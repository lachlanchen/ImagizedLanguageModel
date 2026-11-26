"""Place intact model rasters on the paper figure with a shared physical cell size."""

from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from scripts.export_raw_ilm import ASSETS, PAPER, PUB
from scripts.prepare_paired_ink import read, dump
import fitz


def main():
    files = read(PUB / "raw_rasters.json")["files"]
    pdf = PAPER / "figures/Figure_193_raw_ilm.pdf"
    w, h = 530, 326
    canvas = Canvas(str(pdf), pagesize=(w, h), pageCompression=1)
    items = []
    dark = "#172927"
    muted = "#415A58"

    def label(x, y, text, size=9, bold=False, color=dark):
        items.append(
            dict(type="text", x=x, y=y, text=text, size=size, bold=bold, color=color)
        )
        canvas.setFillColor(HexColor(color))
        canvas.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        canvas.drawString(x, h - y - size, text)

    def img(name, x, y, cell=14):
        f = files[name]
        width = f["width"] * cell / 32
        height = f["height"] * cell / 32
        canvas.drawImage(str(ASSETS / name), x, h - y - height, width, height)
        items.append(dict(type="image", x=x, y=y, w=width, h=height, name=name))

    label(3, 1, "a    The same writer, with the people exchanged", 11, True)
    label(3, 21, "English input", 8, True, muted)
    img("role-55-input.png", 68, 20)
    label(3, 54, "Chinese output", 8, True, muted)
    img("role-55-output.png", 68, 52)
    label(
        3,
        79,
        "b    A held scientific sentence, translated in both directions",
        11,
        True,
    )
    label(3, 100, "English input", 8, True, muted)
    img("natural-zh-input.png", 68, 99)
    label(3, 148, "Chinese output", 8, True, muted)
    img("natural-zh-output.png", 68, 146)
    label(3, 172, "Chinese input", 8, True, muted)
    img("natural-en-input.png", 68, 170)
    label(3, 195, "English output", 8, True, muted)
    img("natural-en-output.png", 68, 193)
    label(
        3,
        247,
        "c    A separate pixel writer receiving the correct visual history",
        11,
        True,
    )
    for role, y, title in (
        ("source", 267, "Target glyph"),
        ("ranking", 298, "Emitted pixels"),
    ):
        label(3, y + 4, title, 8, True, muted)
        for i in range(6):
            img(f"pixel-{i+1}-{role}.png", 84 + 68 * i, y, 24)
    canvas.showPage()
    canvas.save()
    with fitz.open(pdf) as d:
        d[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
            pdf.with_suffix(".png")
        )
    dump(PUB / "scene.json", dict(width=w, height=h, items=items))


if __name__ == "__main__":
    main()
