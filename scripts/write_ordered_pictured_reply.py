"""Generate a Chinese reply image from a pictured Chinese passage and two pictured questions."""

import argparse
import json
from pathlib import Path
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.ordered_evidence_writer import ranked_intervals, writer_inputs
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from ilm.visual_lm.question_evidence_reader import QuestionEvidenceReader
from scripts.evaluate_page_evidence_writer import tensor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument(
        "--questions", type=Path, nargs=2, required=True, metavar=("FIRST", "SECOND")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reader",
        type=Path,
        default=root / "artifacts/bilingual_book_intervals_v1/aligned",
    )
    parser.add_argument(
        "--question-weights",
        type=Path,
        default=root / "models/pictured_question_alignment/weights.safetensors",
    )
    parser.add_argument(
        "--head",
        type=Path,
        default=root / "models/question_evidence_reader/head.safetensors",
    )
    parser.add_argument(
        "--writer",
        type=Path,
        default=root / "artifacts/ordered_book_replies_v1/matched/page",
    )
    parser.add_argument(
        "--separator",
        type=Path,
        default=root
        / "references/evidence/ordered_book_replies_20261124/examples/separator.png",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    page, first, second, sep = [
        packed_page(p).numpy().tobytes()
        for p in (args.page, *args.questions, args.separator)
    ]
    reader = QuestionEvidenceReader(
        args.reader, args.question_weights, args.head, args.device
    )
    qs = [first, second]
    intervals = ranked_intervals(
        reader.reader, reader.questions, reader.head, [(page, q) for q in qs]
    )
    p, q = writer_inputs(page, qs, intervals, sep)
    writer = PreparedGlyphResponder(args.writer, args.device)
    im, end = writer.answer_many([(tensor(p), tensor(q))])[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(im.numpy(), args.output)
    print(json.dumps(dict(cells=len(im), stopped=end)))


if __name__ == "__main__":
    main()
