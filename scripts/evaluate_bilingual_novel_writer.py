"""Score stopped image replies to two different questions in each language."""

from collections import defaultdict
import json
from time import monotonic

import numpy as np
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.audit_interval_teacher import compare, save_image
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.evaluate_print_routing import xquad
from scripts.evaluate_stroke_scale import inputs
from scripts.train_bilingual_novel_writer import ARMS, WORK, PUBLIC, opposite
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face, WORK as PARENT
from scripts.typeface_writer_cases import Styled


def corpus():
    groups = {"book_zh": [], "book_en": [], "book_erased": []}
    validation = WORK / "valid.jsonl"
    assert (
        sha(validation)
        == json.loads((PUBLIC / "corpus.json").read_text())["splits"]["valid"]["sha256"]
    )
    face = Face(FACES[0])
    for line in validation.read_text().splitlines():
        row = json.loads(line)
        docs = row["questions"]
        first, second = next(
            (a, b) for j, a in enumerate(docs) for b in docs[j + 1 :] if opposite(a, b)
        )
        page = face(row["page"])
        for r in (first, second):
            refs = tuple(map(face, r["refs"]))
            for lang in ("zh", "en", "erased"):
                question = bytes(128) if lang == "erased" else face(r[lang])
                groups["book_" + lang].append(
                    Styled(first["id"], page, question, refs, refs, r["id"])
                )
    for key, rows in xquad().items():
        if "regular" in key or "hana" in key:
            groups[key] = rows
    control = inputs()[0]
    for key in (
        "trial_regular",
        "trial_hana",
        "archive_familiar",
        "archive_rephrased",
        "archive_english",
        "yan",
        "open_absent_content_ink",
    ):
        groups[key] = control[key]
    return groups


def overlapping_images(a, b):
    """Can one stopped reply match both requests under the two-print scorer?"""
    for sa, ca in zip(a.source, a.canonical, strict=True):
        for sb, cb in zip(b.source, b.canonical, strict=True):
            if len(sa) == len(sb) and all(
                bool({x, y} & {u, v})
                for x, y, u, v in zip(
                    in_cells(sa), in_cells(ca), in_cells(sb), in_cells(cb), strict=True
                )
            ):
                return True
    return False


def ordered_glyph_f1(case, output):
    """Maximum subsequence F1 against any complete accepted reference."""
    cells = in_cells(output)
    best = 0.0
    for src, can in zip(case.source, case.canonical, strict=True):
        allowed = list(zip(in_cells(src), in_cells(can), strict=True))
        prior = [0] * (len(allowed) + 1)
        for cell in cells:
            current = [0]
            for j, pair in enumerate(allowed):
                current.append(
                    prior[j] + 1 if cell in pair else max(current[-1], prior[j + 1])
                )
            prior = current
        denominator = len(cells) + len(allowed)
        best = max(best, 2 * prior[-1] / denominator if denominator else 1.0)
    return best


def recount(groups, folder):
    result = {}
    passed = {}
    for name, cases in groups.items():
        path = folder / (name + ".npz")
        ys, ends = load_streams(path)
        flags = [c.grade(y, s)[2] for c, y, s in zip(cases, ys, ends, strict=True)]
        by_board = defaultdict(list)
        for c, ok in zip(cases, flags, strict=True):
            by_board[c.board].append(ok)
        passed[name] = np.array([all(v) for v in by_board.values()], bool)
        result[name] = dict(
            n=len(cases),
            exact=sum(flags),
            boards=len(by_board),
            complete=int(passed[name].sum()),
            stops=sum(ends),
            sha256=sha(path),
        )
        if name.startswith(("book_", "xquad_")):
            result[name]["mean_ordered_glyph_f1"] = sum(
                ordered_glyph_f1(c, y) for c, y in zip(cases, ys, strict=True)
            ) / len(cases)
            result[name]["exact_other_question"] = sum(
                cases[i ^ 1].grade(y, end)[2]
                for i, (y, end) in enumerate(zip(ys, ends, strict=True))
            )
    for prefix in ("book", "xquad_regular", "xquad_hana"):
        assert [
            (c.question_id, c.board, c.p, c.canonical) for c in groups[prefix + "_en"]
        ] == [
            (c.question_id, c.board, c.p, c.canonical) for c in groups[prefix + "_zh"]
        ]
        passed[prefix + "_four"] = passed[prefix + "_zh"] & passed[prefix + "_en"]
        result[prefix + "_four"] = dict(
            boards=len(passed[prefix + "_four"]),
            complete=int(passed[prefix + "_four"].sum()),
        )
    return result, passed


@torch.inference_mode()
def generate_archives(groups, base, folder):
    keys = sorted(
        {(c.p, c.q) for rows in groups.values() for c in rows},
        key=lambda t: (len(t[0]) + len(t[1]) * 2, t),
    )
    weights = json.loads((base.parent / "page_fit.json").read_text())["weights"]
    assert all(sha(base / k) == h for k, h in weights.items())
    receipt = folder / "archives.json"
    if receipt.exists():
        archived = json.loads(receipt.read_text())
        assert weights == archived["weights"] and all(
            sha(folder / k) == h for k, h in archived["files"].items()
        )
    else:
        folder.mkdir(exist_ok=True, parents=True)
        state = PreparedGlyphResponder(base, "cuda")
        started = monotonic()
        outputs = {}
        for i in range(0, len(keys), 16):
            batch = keys[i : i + 16]
            predictions = state.answer_many(
                [(tensor(p), tensor(q)) for p, q in batch]
            )
            outputs.update(
                (k, (im.numpy().tobytes(), s))
                for k, (im, s) in zip(batch, predictions, strict=True)
            )
        hashes = {}
        for name, cases in groups.items():
            ys, ends = zip(*(outputs[c.p, c.q] for c in cases), strict=True)
            dest = folder / (name + ".npz")
            store_streams(dest, ys, ends)
            hashes[dest.name] = sha(dest)
        record(
            receipt,
            dict(
                weights=weights,
                files=hashes,
                seconds=monotonic() - started,
                unique=len(keys),
            ),
        )
        del state, outputs


@torch.inference_mode()
def evaluate():
    groups = corpus()
    for prefix in ("book", "xquad_regular", "xquad_hana"):
        rows = groups[prefix + "_zh"]
        if any(
            overlapping_images(a, b) for a, b in zip(rows[::2], rows[1::2], strict=True)
        ):
            raise ValueError("Overlapping physical answer sets in " + prefix)
    scores = {}
    passes = {}
    for arm in ("parent", *ARMS):
        base = PARENT / "mapped/page" if arm == "parent" else WORK / arm / "page"
        folder = WORK / "evaluation" / arm
        generate_archives(groups, base, folder)
        scores[arm], passes[arm] = recount(groups, folder)
        print(
            arm,
            {
                k: scores[arm][k]["complete"]
                for k in (
                    "book_four",
                    "xquad_regular_four",
                    "trial_regular",
                    "archive_familiar",
                )
            },
            flush=True,
        )
    result = scores["bilingual"]
    criteria = {
        g: result[g]["complete"] > max(scores[a][g]["complete"] for a in ARMS[:2])
        for g in ("book_four", "xquad_regular_four")
    }
    criteria.update(
        beats_erasure=result["book_four"]["complete"]
        > result["book_erased"]["complete"],
        regular_retention=result["trial_regular"]["complete"] >= 14,
        historical_retention=result["archive_familiar"]["complete"] >= 99,
    )
    comparisons = {
        g: {
            a: compare(passes["bilingual"][g], passes[a][g])
            for a in ("parent", *ARMS[:2])
        }
        for g in ("book_four", "xquad_regular_four", "xquad_hana_four")
    }
    examples = {}
    target = PUBLIC / "examples"
    target.mkdir(exist_ok=True)
    for tag, mask in (
        ("book_success", passes["bilingual"]["book_four"]),
        ("book_miss", ~passes["bilingual"]["book_four"]),
    ):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        i = int(indices[0]) * 2
        examples[tag] = [groups["book_en"][k].question_id for k in (i, i + 1)]
        for lang in ("zh", "en"):
            cases = groups["book_" + lang]
            ys, ends = load_streams(
                WORK / "evaluation/bilingual" / ("book_" + lang + ".npz")
            )
            for k in range(2):
                c = cases[i + k]
                for name, raw in (
                    ("question", c.q),
                    ("answer", ys[i + k]),
                    ("reference", c.source[0]),
                ):
                    save_image(target / (f"{tag}_{lang}_{k}_{name}.png"), raw)
    record(
        PUBLIC / "result.json",
        dict(
            scores=scores,
            criteria=criteria,
            selected=all(criteria.values()),
            paired=comparisons,
            examples=examples,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
