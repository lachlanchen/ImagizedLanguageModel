#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import os
from pathlib import Path
from typing import Dict, List, Tuple

from ilm.db.glyph_db import GlyphDB


def _parse_lang_token(s: str) -> Tuple[str, str]:
    # Expect format like "en::word" or "zh::字"
    if "::" in s:
        lang, tok = s.split("::", 1)
        return lang, tok
    # Fallback: try to infer
    if all(ord(c) < 128 for c in s):
        return "en", s
    return "zh", s


def _safe_name(lang: str, tok: str) -> str:
    # Build a safe filename for glyph copy
    base = f"{lang}_{tok}"
    safe = []
    for ch in base:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    s = "".join(safe)
    if len(s) > 80:
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        s = s[:64] + "_" + h
    return s + ".png"


def collect_neighbors(nn_tsv: Path, top_en: int = 20, topk_per_en: int = 3) -> List[Tuple[str, List[Tuple[str, float]]]]:
    # Read TSV: en_token, zh_neighbor, sim
    rows: Dict[str, List[Tuple[str, float]]] = {}
    with open(nn_tsv, "r", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr, None)
        for en_tok, zh_tok, sim in rdr:
            try:
                score = float(sim)
            except Exception:
                score = 0.0
            rows.setdefault(en_tok, []).append((zh_tok, score))
    # Sort neighbors and pick topK per english token
    items: List[Tuple[str, List[Tuple[str, float]]]] = []
    for en_tok, nbrs in rows.items():
        nbrs_sorted = sorted(nbrs, key=lambda x: -x[1])[:topk_per_en]
        items.append((en_tok, nbrs_sorted))
    # Sort english tokens by max neighbor score descending
    items.sort(key=lambda it: - (it[1][0][1] if it[1] else 0.0))
    return items[:top_en]


def write_html(
    out_dir: Path,
    pca_png: Path,
    pca_c0_png: Path,
    examples: List[Tuple[str, List[Tuple[str, float]]]],
    glyph_db: GlyphDB,
) -> None:
    title = "ILM Embedding Visualization"
    glyphs_dir = out_dir / "glyphs"
    glyphs_dir.mkdir(parents=True, exist_ok=True)

    def ensure_copy(lang: str, tok: str) -> str:
        src = Path(glyph_db.ensure_glyph(lang, tok, 128))
        dst = glyphs_dir / _safe_name(lang, tok)
        if not dst.exists():
            try:
                os.link(src, dst)
            except Exception:
                import shutil
                shutil.copy2(src, dst)
        return dst.name

    # Build neighbor table rows
    rows_html: List[str] = []
    for en_tok, nbrs in examples:
        lang_en, tok_en = _parse_lang_token(en_tok)
        en_img = ensure_copy(lang_en, tok_en)
        nbr_cells = []
        for zh_tok, score in nbrs:
            lang_zh, tok_zh = _parse_lang_token(zh_tok)
            zh_img = ensure_copy(lang_zh, tok_zh)
            nbr_cells.append(
                f"<div class='nbr'><img src='glyphs/{zh_img}' alt='{html.escape(zh_tok)}'><div class='cap'>{html.escape(tok_zh)}<br><small>{score:.3f}</small></div></div>"
            )
        row = (
            "<tr>"
            f"<td class='en'><img src='glyphs/{en_img}' alt='{html.escape(en_tok)}'><div class='cap'>{html.escape(tok_en)}</div></td>"
            f"<td class='zh'>{''.join(nbr_cells)}</td>"
            "</tr>"
        )
        rows_html.append(row)

    doc: List[str] = []
    doc.append(f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>")
    doc.append(
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto}img{max-width:100%}.row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1 1 48%}table{width:100%;border-collapse:collapse}td{vertical-align:top;padding:8px;border-bottom:1px solid #eee}.en img{width:128px;height:128px;object-fit:contain}.zh .nbr{display:inline-block;margin-right:12px;text-align:center}.zh .nbr img{width:96px;height:96px;object-fit:contain}.cap{margin-top:4px;color:#333}</style>"
    )
    doc.append("</head><body>")
    doc.append(f"<h1>{html.escape(title)}</h1>")
    doc.append("<div class='row'>")
    if pca_png.exists():
        doc.append("<div class='col'><h3>PCA (tokens)</h3>")
        doc.append(f"<img src='{pca_png.name}' alt='pca tokens'></div>")
    if pca_c0_png.exists():
        doc.append("<div class='col'><h3>PCA colored by channel-0 index</h3>")
        doc.append(f"<img src='{pca_c0_png.name}' alt='pca c0'></div>")
    doc.append("</div>")
    doc.append("<h2>EN → ZH nearest neighbors</h2>")
    doc.append("<table>")
    doc.append("<tr><th>EN token</th><th>Top ZH neighbors (glyph + sim)</th></tr>")
    doc.extend(rows_html)
    doc.append("</table>")
    doc.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(doc), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an HTML viz report for ILM embeddings")
    ap.add_argument("--out", required=True, help="Output directory containing PCA and nn TSV (or to write them)")
    ap.add_argument("--glyph-db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--top-en", type=int, default=20)
    ap.add_argument("--topk-per-en", type=int, default=3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_png = out_dir / "pca_tokens.png"
    pca_c0_png = out_dir / "pca_tokens_c0.png"
    nn_tsv = out_dir / "nn_en_to_zh.tsv"
    if not nn_tsv.exists():
        raise SystemExit(f"Missing neighbor file: {nn_tsv}. Run viz_ilmglyph_reports.py first for this folder.")
    examples = collect_neighbors(nn_tsv, top_en=args.top_en, topk_per_en=args.topk_per_en)
    db = GlyphDB(args.glyph_db)
    write_html(out_dir, pca_png, pca_c0_png, examples, db)
    print(f"Report ready: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

