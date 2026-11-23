"""Select an answer image using a pictured page and a pictured question."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.question_evidence_reader import QuestionEvidenceReader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    packed = root / "reader"
    ap.add_argument(
        "--reader",
        type=Path,
        default=packed
        if packed.is_dir()
        else root / "artifacts/bilingual_book_intervals_v1/aligned",
    )
    ap.add_argument(
        "--question-weights",
        type=Path,
        default=root / "models/pictured_question_alignment/weights.safetensors",
    )
    ap.add_argument(
        "--head",
        type=Path,
        default=root / "models/question_evidence_reader/head.safetensors",
    )
    ap.add_argument("--passage", type=Path, required=True)
    ap.add_argument("--question", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", choices=("excerpt","write"), default="excerpt")
    ap.add_argument("--writer",type=Path,default=root/"writer" if (root/"writer").is_dir() else root/"artifacts/bilingual_novel_v1/bilingual/page")
    args = ap.parse_args()
    torch.set_num_threads(2)
    engine = QuestionEvidenceReader(
        args.reader, args.question_weights, args.head, args.device
    )
    p, q = [
        packed_page(path).numpy().tobytes() for path in (args.passage, args.question)
    ]
    y = engine.answer(p, q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output=np.frombuffer(y, np.uint8).reshape(-1,128)
    stopped=None
    if args.mode=="write":
        from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
        writer=PreparedGlyphResponder(args.writer,args.device)
        question=torch.from_numpy(np.frombuffer(q,np.uint8).copy().reshape(-1,128))
        response,stopped=writer.answer_many([(torch.from_numpy(output.copy()),question)])[0]
        output=response.numpy()
    write_strip(output,args.output)
    print(json.dumps(dict(mode=args.mode,glyphs=len(output),stopped=stopped)))


if __name__ == "__main__":
    main()
