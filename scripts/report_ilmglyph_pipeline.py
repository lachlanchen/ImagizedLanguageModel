#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

from ilm.datasets.alpaca_glyph_dataset import tokenize_en, tokenize_zh


def write_report(out_dir: Path, lang: str, text: str, grid: int, cell: int) -> None:
    # Gather artifacts produced by demo script
    glyphs = out_dir / "01_input_glyphs.png"
    hstripe = out_dir / "02_code_hstripes.png"
    frame = out_dir / "03_code_frame.png"
    frame_mask = out_dir / "03b_code_frame_masked.png"
    frame_pred = out_dir / "04_code_frame_pred.png"
    text_txt = out_dir / "05_text.txt"
    # Build HTML
    title = f"ILM Pipeline Report ({lang})"
    input_tokens = tokenize_en(text) if lang == "en" else tokenize_zh(text)
    limit_tokens = grid * grid
    doc = []
    doc.append("<html><head><meta charset='utf-8'><title>%s</title>" % html.escape(title))
    doc.append("<style> body{font-family:sans-serif; max-width: 980px; margin: 2rem auto;} img{max-width:100%;} code{background:#f5f5f5;padding:2px 4px;border-radius:3px;} .row{display:flex;gap:16px;flex-wrap:wrap} .col{flex:1 1 45%} h2{margin-top:2rem;} </style>")
    doc.append("</head><body>")
    doc.append(f"<h1>{html.escape(title)}</h1>")
    doc.append("<h2>Input</h2>")
    doc.append(f"<p><b>Text</b> ({lang}): <code>{html.escape(text)}</code></p>")
    doc.append(f"<p><b>Tokens</b> ({len(input_tokens)}): <code>{html.escape(str(input_tokens))}</code></p>")
    doc.append("<div class='row'>")
    if glyphs.exists():
        doc.append("<div class='col'><h3>Glyphs (128×128 each)</h3>")
        doc.append(f"<img src='{glyphs.name}' alt='glyphs'></div>")
    if hstripe.exists():
        doc.append("<div class='col'><h3>Code stripes (channels over time)</h3>")
        doc.append(f"<img src='{hstripe.name}' alt='code stripes'></div>")
    doc.append("</div>")
    doc.append("<h2>Sentence as Image (Code Frame)</h2>")
    doc.append(f"<p>Grid <b>{grid}×{grid}</b> (max tokens <b>{limit_tokens}</b>), cell size <b>{cell}px</b> → frame <b>{grid*cell}×{grid*cell}</b> pixels.</p>")
    doc.append("<p>Default codebook capacity: <b>C=3</b> channels × <b>K=32</b> codes → <b>32<sup>3</sup>=32,768</b> unique hard codes. Adjust via training flags if needed.</p>")
    doc.append("<div class='row'>")
    if frame.exists():
        doc.append("<div class='col'><h3>Frame (original)</h3>")
        doc.append(f"<img src='{frame.name}' alt='code frame'></div>")
    if frame_mask.exists():
        doc.append("<div class='col'><h3>Masked Frame (red cells masked)</h3>")
        doc.append(f"<img src='{frame_mask.name}' alt='masked frame'></div>")
    doc.append("</div>")
    doc.append("<div class='row'>")
    if frame_pred.exists():
        doc.append("<div class='col'><h3>Predicted Frame (mask+infill baseline)</h3>")
        doc.append(f"<img src='{frame_pred.name}' alt='predicted frame'></div>")
    doc.append("</div>")
    doc.append("<h2>Decoded Text</h2>")
    if text_txt.exists():
        txt = text_txt.read_text(encoding="utf-8")
        doc.append("<pre>%s</pre>" % html.escape(txt))
    else:
        doc.append("<p>Decoded text file not found.</p>")
    doc.append("<h2>Design Notes</h2>")
    doc.append("<ul>")
    doc.append("<li>Glyph size per token: <b>128×128</b> (for visual inspection).</li>")
    doc.append("<li>Frame size per sentence: default <b>16×16</b> tokens with <b>8px</b> cells → <b>128×128</b> frame.</li>")
    doc.append("<li>For long paragraphs, <b>32×32</b> tokens with <b>4px</b> cells stays 128×128 but reduces per-token signal; recommended only when needed.</li>")
    doc.append("<li>The predicted frame here uses a nearest-neighbor mask+infill baseline in code space (a diffusion-style visualization). A full 2D UNet diffusion can replace this for iterative denoising.</li>")
    doc.append("</ul>")
    doc.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(doc), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an HTML report of the ILM pipeline for a single input")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--lang", required=True, choices=["en", "zh"], help="Language of input text")
    ap.add_argument("--text", required=True, help="Input text")
    ap.add_argument("--out", default="artifacts/ilm_report", help="Output directory")
    ap.add_argument("--glyph-db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--cell", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Run the demo to generate artifacts
    import subprocess, sys
    cmd = [
        sys.executable,
        "scripts/demo_ilmglyph_pipeline.py",
        "--ckpt", args.ckpt,
        "--lang", args.lang,
        "--text", args.text,
        "--out", str(out_dir),
        "--glyph-db", args.glyph_db,
        "--grid", str(args.grid),
        "--cell", str(args.cell),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Write HTML report
    write_report(out_dir, args.lang, args.text, args.grid, args.cell)
    print(f"Report ready: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
