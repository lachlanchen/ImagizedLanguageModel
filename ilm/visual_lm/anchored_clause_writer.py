"""Write around a physical excerpt selected from the page, with explicit failure reporting."""

from dataclasses import dataclass

import numpy as np
import torch

from ilm.visual_lm.ordered_evidence_writer import ranked_intervals


def cells(raw):
    return torch.from_numpy(np.frombuffer(raw, np.uint8).copy().reshape(-1, 128))


def clause_inputs(page, question, fact, mark):
    if (
        not fact
        or len(mark) != 128
        or not all(s and len(s) % 128 == 0 for s in (page, question, fact))
    ):
        raise ValueError(
            "Expected aligned nonempty glyph strips and one insertion mark"
        )
    return page + mark + fact + mark, question


@dataclass(frozen=True)
class Clause:
    ink: bytes
    valid: bool
    written_cells: int


def fill_frame(frame, stopped, fact, mark):
    """Only an aligned, unique predicted insertion mark licenses expansion."""
    spots = [i for i in range(0, len(frame), 128) if frame.startswith(mark, i)]
    if not stopped or len(spots) != 1:
        return Clause(frame, False, 0)
    i = spots[0]
    return Clause(frame[:i] + fact + frame[i + 128 :], True, len(frame) // 128 - 1)


def join_clauses(first, second, comma, full_stop, limit=48):
    ink = first.ink + comma + second.ink + full_stop
    # Keep raw passage evidence and overlength outputs for inspection.
    return ink, first.valid and second.valid and len(ink) <= limit * 128


@torch.inference_mode()
def answer_clauses(writer, queries, mark):
    pairs = [
        clause_inputs(page, question, fact, mark) for page, question, fact in queries
    ]
    frames = writer.answer_many([(cells(p), cells(q)) for p, q in pairs], limit=24)
    return [
        fill_frame(im.numpy().tobytes(), stop, fact, mark)
        for (_, _, fact), (im, stop) in zip(queries, frames, strict=True)
    ]


@torch.inference_mode()
def answer_pair(reader, writer, page, questions, mark, comma, full_stop):
    if len(questions) != 2:
        raise ValueError("Expected two pictured questions")
    ranked = ranked_intervals(
        reader.reader, reader.questions, reader.head, [(page, q) for q in questions], 1
    )
    queries = []
    for q, intervals in zip(questions, ranked, strict=True):
        a, b = intervals[0]
        queries.append((page, q, page[128 * a : 128 * b]))
    clauses = answer_clauses(writer, queries, mark)
    return join_clauses(*clauses, comma, full_stop)
