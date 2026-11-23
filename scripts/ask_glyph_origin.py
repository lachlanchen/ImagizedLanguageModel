"""Ask the local image writer about attested forms of a Chinese headword."""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from ilm.etymology.image_catalogue import ImageCatalogue, STAGES
from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.respond_pointing_glyphs import render_text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("headword")
    ap.add_argument("--stages", nargs=2, choices=STAGES)
    ap.add_argument("--model", type=Path, default=Path("writer"))
    ap.add_argument("--catalogue", type=Path, default=Path("catalogue"))
    ap.add_argument("--list-stages", action="store_true")
    q = ap.add_mutually_exclusive_group()
    q.add_argument("--question")
    q.add_argument("--question-image", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    ap.add_argument("--font-index", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    catalogue = ImageCatalogue(args.catalogue)
    try:
        available = catalogue.available(args.headword)
        if args.list_stages:
            print(json.dumps(available, ensure_ascii=False, indent=2))
            return
        rows = catalogue.card(args.headword, args.stages)
    except ValueError as err:
        ap.error(str(err))
    if args.output_dir is None:
        ap.error("--output-dir is required for an answer")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error("Choose a new or empty output directory")
    args.output_dir.mkdir(exist_ok=True, parents=True)
    stages = args.stages or [r["stage"] for r in rows if r["char"] == args.headword]
    question = (
        args.question
        or f"請介紹「{args.headword}」的字形，依{STAGES[stages[0]]}、{STAGES[stages[1]]}的順序作答。"
    )
    torch.set_num_threads(2)

    @lru_cache(maxsize=128)
    def draw(s):
        return render_text(s, args.font, args.font_index).numpy().tobytes()

    p = torch.from_numpy(
        np.frombuffer(catalogue.page(rows, draw), np.uint8).copy().reshape(-1, 128)
    )
    q = (
        packed_page(args.question_image)
        if args.question_image
        else render_text(question, args.font, args.font_index)
    )
    net = PreparedGlyphResponder(args.model, args.device)
    ((out, stopped),) = net.answer_many([(p, q)])
    for name, t in [("page", p), ("question", q), ("answer", out)]:
        write_strip(t.numpy(), args.output_dir / (name + ".png"))
    evidence = dict(
        headword=args.headword,
        requested_stages=stages,
        stopped=stopped,
        returned_cells=len(out),
        attestations=[
            r | {"source": "https://hanziyuan.net/#" + quote(r["char"])} for r in rows
        ],
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"{len(out)} glyphs, stopped={stopped}. {args.output_dir/'answer.png'}")


if __name__ == "__main__":
    main()
