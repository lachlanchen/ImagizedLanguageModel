"""Read a pictured page and write one stopped answer image per pictured question."""

import argparse
import json
from pathlib import Path
import torch
from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.respond_pointing_glyphs import render_text


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    pg = ap.add_mutually_exclusive_group(required=True)
    pg.add_argument("--page", type=Path)
    pg.add_argument("--page-text")
    qs = ap.add_mutually_exclusive_group(required=True)
    qs.add_argument("--questions", nargs="+", type=Path)
    qs.add_argument("--question-text", nargs="+")
    ap.add_argument(
        "--writer",
        type=Path,
        default=root / "artifacts/action_binding_v1/clause_aligned/page",
    )
    ap.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    ap.add_argument("--font-index", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    p = (
        packed_page(args.page)
        if args.page
        else render_text(args.page_text, args.font, args.font_index)
    )
    inputs = (
        [packed_page(path) for path in args.questions]
        if args.questions
        else [render_text(s, args.font, args.font_index) for s in args.question_text]
    )
    net = PreparedGlyphResponder(args.writer, args.device)
    # Limit peak memory for longer question lists. Preserve the supplied order.
    outputs = []
    for i in range(0, len(inputs), 8):
        outputs.extend(net.answer_many([(p, q) for q in inputs[i : i + 8]]))
    ink = torch.cat([im for im, _ in outputs])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    write_strip(ink.numpy(), args.output)
    print(
        json.dumps(
            dict(
                cells=[len(im) for im, _ in outputs],
                stops=[bool(e) for _, e in outputs],
            )
        )
    )


if __name__ == "__main__":
    main()
