"""Generate a Chinese reply image from a pictured Chinese passage and two pictured questions."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder


def tensor(raw):
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1, 128))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument(
        "--questions", type=Path, nargs=2, required=True, metavar=("FIRST", "SECOND")
    )
    parser.add_argument("--mode", choices=("direct", "frame", "page"), default="direct")
    parser.add_argument(
        "--controls",
        type=Path,
        default=root / "references/evidence/anchored_book_clauses_20261124/examples",
        help="Directory containing the three physical control strips for frame mode",
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
        default=None,
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
    qs = [first, second]
    folders = dict(
        direct="artifacts/ordered_book_replies_v1/matched/page",
        frame="artifacts/anchored_book_clauses_v1/page",
        page="artifacts/role_change_writer_v1/aligned/page",
    )
    writer = PreparedGlyphResponder(
        args.writer or root / folders[args.mode], args.device
    )
    if args.mode == "page":
        im, end = writer.answer_many([(tensor(page), tensor(first + sep + second))])[0]
    else:
        from ilm.visual_lm.anchored_clause_writer import answer_pair
        from ilm.visual_lm.ordered_evidence_writer import (
            ranked_intervals,
            writer_inputs,
        )
        from ilm.visual_lm.question_evidence_reader import QuestionEvidenceReader

        reader = QuestionEvidenceReader(
            args.reader, args.question_weights, args.head, args.device
        )
        if args.mode == "frame":
            controls = [
                packed_page(args.controls / (name + ".png")).numpy().tobytes()
                for name in ("mark", "comma", "stop")
            ]
            ink, end = answer_pair(reader, writer, page, qs, *controls)
            im = tensor(ink)
        else:
            intervals = ranked_intervals(
                reader.reader, reader.questions, reader.head, [(page, q) for q in qs]
            )
            p, q = writer_inputs(page, qs, intervals, sep)
            im, end = writer.answer_many([(tensor(p), tensor(q))])[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(im.numpy(), args.output)
    print(json.dumps(dict(cells=len(im), accepted=end, mode=args.mode)))


if __name__ == "__main__":
    main()
