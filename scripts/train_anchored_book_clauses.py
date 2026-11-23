"""Learn clause frames from observed book QA pairs and test actual visual evidence insertion."""

from collections import Counter, defaultdict
import json
from time import monotonic
import torch
from safetensors.torch import load_file

from ilm.visual_lm.anchored_clause_writer import (
    answer_clauses,
    clause_inputs,
    join_clauses,
)
from ilm.visual_lm.ordered_evidence_writer import ranked_intervals
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.mine_stopped_book_replies import BOOK_PUBLIC, PARENT
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_ordered_book_replies import PUBLIC as LAST, sample
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_ordered_book_replies import cases, grades, LANGS
from scripts.train_question_evidence_head import bodies
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

WORK = ROOT / "artifacts/anchored_book_clauses_v1"
PUBLIC = ROOT / "references/evidence/anchored_book_clauses_20261124"


def training(face):
    rows = []
    for c in cases()["train"]:
        p = face(c["page"])
        for k, q in enumerate(c["qs"]):
            fact = q["refs"][0]
            clause = c["clauses"][k]
            assert (
                clause.count(fact) == 1 and "※" not in clause and "※" not in c["page"]
            )
            y = face(clause.replace(fact, "※"))
            for lang in LANGS:
                source, query = clause_inputs(p, face(q[lang]), face(fact), face("※"))
                rows.append(Row(lang, c["key"] + str(k), source, query, y, b""))
    return rows


def fit():
    WORK.mkdir(exist_ok=False)
    face = Face(FACES[0])
    weights = json.loads((BOOK_PUBLIC / "bilingual_fit.json").read_text())["weights"]
    rows = training(face)
    fit_rows(WORK, PARENT, weights, rows * 3, 241209)
    record(
        PUBLIC / "fit.json",
        dict(unique_rows=len(rows), **json.loads((WORK / "page_fit.json").read_text())),
    )


@torch.inference_mode()
def evaluate():
    start = monotonic()
    face = Face(FACES[0])
    groups = cases()
    additional = sample("valid", 160)[96:]
    if len(additional) != 64 or {c["page"] for c in additional} & {
        c["page"] for rows in groups.values() for c in rows
    }:
        raise ValueError("Additional pages overlap or are missing")
    groups = {"new": groups["new"], "book": groups["book"], "additional": additional}
    read, questions = bodies()
    head = QuestionEvidenceHead().cuda().eval()
    from scripts.reserve_mlqa_pictures import PUBLIC as R, WORK as H

    head_file = H / "matched.safetensors"
    assert sha(head_file) == json.loads((R / "matched_fit.json").read_text())["sha256"]
    head.load_state_dict(load_file(str(head_file)))
    writer = PreparedGlyphResponder(WORK / "page", "cuda")
    mark, comma, stop = map(face, ("※", "，", "。"))
    results = {}
    for group, cs in groups.items():
        results[group] = {}
        inputs = []
        for c in cs:
            p = face(c["page"])
            for lang in LANGS:
                for q in c["qs"]:
                    inputs.append((p, face(q[lang]), face(q["refs"][0])))
        for mode in ("predicted", "oracle", "bare"):
            clauses = []
            for j in range(0, len(inputs), 24):
                chunk = inputs[j : j + 24]
                if mode == "oracle":
                    use = chunk
                else:
                    ranks = ranked_intervals(
                        read, questions, head, [(p, q) for p, q, f in chunk], 1
                    )
                    use = [
                        (p, q, p[128 * ls[0][0] : 128 * ls[0][1]])
                        for (p, q, _), ls in zip(chunk, ranks, strict=True)
                    ]
                if mode == "bare":
                    from ilm.visual_lm.anchored_clause_writer import Clause

                    clauses.extend(Clause(f, True, 0) for p, q, f in use)
                else:
                    clauses.extend(answer_clauses(writer, use, mark))
            ys, ends = [], []
            counts = defaultdict(Counter)
            flags = defaultdict(list)
            diagnostics = []
            for i, c in enumerate(cs):
                facts = [face(q["refs"][0]) for q in c["qs"]]
                for li, lang in enumerate(LANGS):
                    pair = clauses[4 * i + 2 * li : 4 * i + 2 * li + 2]
                    for rev in (False, True):
                        ordered = pair[::-1] if rev else pair
                        y, end = join_clauses(*ordered, comma, stop)
                        ok = grades(y, end, facts, rev)
                        g = counts[lang]
                        g["requests"] += 1
                        g["valid_and_in_budget"] += int(end)
                        g["both_frames_valid"] += int(all(c.valid for c in pair))
                        g["both_frames_have_context"] += int(
                            all(c.written_cells > 0 for c in pair)
                        )
                        g["ordered"] += int(ok)
                        if "clauses" in c:
                            ref = c["clauses"][::-1] if rev else c["clauses"]
                            g["exact_reference"] += int(
                                end and y == face("，".join(ref) + "。")
                            )
                        flags[c["key"], lang].append(ok)
                        ys.append(y)
                        ends.append(end)
                        diagnostics.append(
                            dict(
                                key=c["key"],
                                lang=lang,
                                reverse=rev,
                                valid=end,
                                ordered=ok,
                                context_cells=[v.written_cells for v in ordered],
                            )
                        )
            for (_, lang), values in flags.items():
                counts[lang]["both_orders"] += int(all(values))
            path = WORK / f"{group}_{mode}.npz"
            store_streams(path, ys, ends)
            record(WORK / f"{group}_{mode}.json", diagnostics)
            results[group][mode] = dict(
                counts=counts,
                all_four=sum(
                    all(flags[c["key"], "en"] + flags[c["key"], "zh"]) for c in cs
                ),
                pages=len(cs),
                sha256=sha(path),
            )
    record(
        PUBLIC / "result.json",
        dict(
            scores=results,
            seconds=monotonic() - start,
            prior=sha(LAST / "result.json"),
            additional_ids=[c["key"] for c in additional],
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        torch.set_num_threads(2)
        fit()
        evaluate()
