"""Test source dependence by exchanging equal-width answer phrases on held trial pages."""

from collections import Counter
import json
import unicodedata

import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.train_causal_glyph_writer import sha, record
from scripts.train_span_choice import gpu_owner
from scripts.train_stroke_scale import ARMS, FANDOL, WORK, PUBLIC
from scripts.train_typeface_writer import HELD, FACES, WORK as PARENT, Face
from scripts.typeface_writer_cases import extra_sources, pictures


def same_script(text):
    if all(unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH") for c in text):
        return "han"
    if text.isascii() and text.isdecimal():
        return "numeral"
    if text.isascii() and text.isalpha():
        return "latin"
    return None


def exchange(first, second):
    p = first["p"]
    assert p == second["p"]
    if len(first["refs"]) != 1 or len(second["refs"]) != 1:
        return "alternative_answers", None
    a, b = first["refs"][0], second["refs"][0]
    if len(a) != len(b) or not 2 <= len(a) <= 8 or a == b:
        return "width", None
    if not same_script(a) or same_script(a) != same_script(b):
        return "script", None
    if any(w in q["q"] for w in (a, b) for q in (first, second)):
        return "question_mentions_answer", None
    # find with a one-character advance to detect self-overlapping occurrences.
    spans = []
    for w in (a, b):
        k = p.find(w)
        if k < 0 or p.find(w, k + 1) >= 0:
            return "nonunique_source", None
        spans.append(k)
    x, y = spans
    if abs(x - y) < len(a):
        return "overlap", None
    chars = list(p)
    chars[x : x + len(a)] = b
    chars[y : y + len(b)] = a
    edited = "".join(chars)
    return "included", [
        dict(v, p=edited, refs=[w]) for v, w in ((first, b), (second, a))
    ]


def sample():
    original, changed, reasons = [], [], Counter()
    available = json.loads((PUBLIC / "coverage.json").read_text())["cohorts"]
    keep = set.intersection(
        *(
            set(available["trial_" + f]["question_ids"])
            for f in ("regular", "fandol", "hana")
        )
    )
    rows, _ = extra_sources()
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        if a["id"] not in keep or b["id"] not in keep:
            reasons["font_coverage"] += 1
            continue
        reason, edited = exchange(a, b)
        reasons[reason] += 1
        if edited:
            original.extend((a, b))
            changed.extend(edited)
    return original, changed, reasons


@torch.inference_mode()
def run():
    original, changed, reasons = sample()
    folder = WORK / "exchanged"
    folder.mkdir()
    result = dict(
        exclusions=dict(reasons), boards=[v["board"] for v in original[::2]], scores={}
    )
    if not original:
        record(PUBLIC / "exchanged.json", result)
        return
    canonical = Face(FACES[0])
    model = PreparedGlyphResponder(PARENT / "mapped/page", "cuda")
    for arm in ARMS:
        path = WORK / (arm + ".safetensors")
        assert (
            sha(path)
            == json.loads((PUBLIC / (arm + "_fit.json")).read_text())["sha256"]
        )
        model.net.writer.attach_typeface_map(path)
        result["scores"][arm] = {}
        for family, f in (("regular", FACES[0]), ("hana", HELD), ("fandol", FANDOL)):
            face = Face(f)
            rows = pictures(original, face, canonical) + pictures(
                changed, face, canonical
            )
            assert len(rows) == 2 * len(original)
            pairs = [(tensor(c.p), tensor(c.q)) for c in rows]
            outs = []
            for i in range(0, len(rows), 16):
                outs.extend(
                    (x.numpy().tobytes(), s)
                    for x, s in model.answer_many(pairs[i : i + 16])
                )
            target = folder / (arm + "_" + family + ".npz")
            ys, ends = zip(*outs, strict=True)
            store_streams(target, ys, ends)
            ys, ends = load_streams(target)
            ok = [c.grade(y, s)[2] for c, y, s in zip(rows, ys, ends, strict=True)]
            n = len(original)
            result["scores"][arm][family] = dict(
                questions=n,
                boards=n // 2,
                original=sum(ok[:n]),
                exchanged=sum(ok[n:]),
                both_versions=sum(a and b for a, b in zip(ok[:n], ok[n:], strict=True)),
                all_four=sum(
                    all(ok[j] for j in (i, i + 1, n + i, n + i + 1))
                    for i in range(0, n, 2)
                ),
                output_changed=sum(
                    ys[i] != ys[i + n] or ends[i] != ends[i + n] for i in range(n)
                ),
                sha256=sha(target),
            )
    record(PUBLIC / "exchanged.json", result)


if __name__ == "__main__":
    with gpu_owner():
        run()
