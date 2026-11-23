"""Compare two existing visual contextual readers on the identical classical questions."""

import argparse
from time import monotonic
import torch

from ilm.visual_lm.book_participant_reader import InkBank, best_strip, group_spans
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_classical_argument_reader import populations
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_role_question_binding import scores
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_book_span_distillation import OpticalTeacher
from scripts.prepare_classical_argument_reader import WORK, PUBLIC, read, record, sha
from scripts.train_bilingual_novel_writer import WORK as NOVEL
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face


@torch.inference_mode()
def evaluate(kind):
    rows = [r for r in populations() if r["group"] in ("zuo", "shishuo")]
    face = Face(FACES[0])
    pairs = [(face(r["page"]), face(r["question"])) for r in rows]
    outs = []
    if kind == "optical":
        bank = InkBank()
        encoded = [(bank.encode(p), bank.encode(q)) for p, q in pairs]
        net = OpticalTeacher(bank)
        weights = net.source
        first = snapshot()
        t = monotonic()
        for i in range(0, len(rows), 16):
            batch = encoded[i : i + 16]
            for (p, _), z in zip(batch, net.score(batch), strict=True):
                best = best_strip(z.cpu().numpy(), group_spans(p))
                outs.append((b"".join(bank.cells[j] for j in best), True))
    else:
        folder = NOVEL / "bilingual/page"
        weights = read(folder.parent / "page_fit.json")["weights"]
        assert all(sha(folder / k) == v for k, v in weights.items())
        net = PreparedGlyphResponder(folder, "cuda")
        first = snapshot()
        t = monotonic()
        for i in range(0, len(rows), 16):
            for pixels, stopped in net.answer_many(
                [(tensor(p), tensor(q)) for p, q in pairs[i : i + 16]]
            ):
                outs.append((pixels.cpu().numpy().tobytes(), bool(stopped)))
    seconds = monotonic() - t
    energy = consumed(first, snapshot())
    detail = []
    for r, (y, end) in zip(rows, outs, strict=True):
        detail.append(
            dict(
                **{k: r[k] for k in ("key", "group", "world", "lang", "h")},
                reply=y.hex(),
                ended=end,
                correct=end and y == face(r["answer"]),
                rival=end and y == face(r["other"]),
            )
        )
    f = WORK / (kind + "_responses.json")
    record(f, detail)
    record(
        PUBLIC / (kind + "_scores.json"),
        dict(
            counts=scores(detail),
            response_sha=sha(f),
            weights=weights,
            seconds=seconds,
            energy=energy,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("optical", "bilingual_writer"))
    args = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        evaluate(args.arm)
