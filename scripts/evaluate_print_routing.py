"""Run the frozen print router on paired Chinese passages and English questions."""

from collections import Counter
import json
import os
from time import monotonic

import torch

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.audit_interval_teacher import save_image
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.evaluate_stroke_scale import inputs
from scripts.interval_teacher_corpus import fingerprints
from scripts.span_reservation import reservation
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_stroke_scale import FANDOL, WORK, PUBLIC
from scripts.train_typeface_writer import FACES, HELD, ROOT, WORK as PARENT, Face
from scripts.typeface_writer_cases import pictures

DIGESTS = {
    "en": "e4c57d1c9143aaa1c5d265ba5987a65f4e69528d2a98f29d6e75019b10344f29",
    "zh": "691d0b3359bc6b8faa8de931dfdfe21d50a65861ae348e32a0d1a0190b0c8835",
}


def paragraphs(path):
    for article in json.loads(path.read_text())["data"]:
        yield from article["paragraphs"]


def xquad():
    data = ROOT / "data/raw"
    files = {k: data / "xquad" / ("xquad." + k + ".json") for k in DIGESTS}
    for k, path in files.items():
        assert sha(path) == DIGESTS[k]
    english = {
        q["id"]: q["question"] for v in paragraphs(files["en"]) for q in v["qas"]
    }
    squad = {
        q["id"] for v in paragraphs(data / "squad_v1.1/dev-v1.1.json") for q in v["qas"]
    }
    blocked = set()
    for split in ("train", "dev", "trial"):
        for p in paragraphs(data / "cmrc2018" / ("cmrc2018_" + split + ".json")):
            blocked.update(fingerprints(p["context"][:416]))
    guard = reservation()
    rows = []
    reasons = Counter()
    for i, v in enumerate(paragraphs(files["zh"])):
        p = v["context"][:416]
        if guard.overlaps(v["context"]) or fingerprints(p) & blocked:
            reasons["overlap"] += 1
            continue
        candidates = []
        for q in v["qas"]:
            en = english[q["id"]]
            if len(p) + 2 * max(len(q["question"]), len(en)) + 52 > 1024:
                continue
            refs = list(
                dict.fromkeys(
                    a["text"]
                    for a in q["answers"]
                    if 0 < len(a["text"]) <= 48 and a["text"] in p
                )
            )
            if refs:
                candidates.append(
                    dict(
                        p=p, q=q["question"], en=en, refs=refs, id=q["id"], board=str(i)
                    )
                )
        pair = next(
            (
                (a, b)
                for j, a in enumerate(candidates)
                for b in candidates[j + 1 :]
                if a["q"] != b["q"]
                and a["en"] != b["en"]
                and all(x not in y and y not in x for x in a["refs"] for y in b["refs"])
            ),
            None,
        )
        if pair:
            rows.extend(pair)
        reasons["included" if pair else "insufficient_questions"] += 1
    canonical = Face(FACES[0])
    cohorts = {}
    for family, desc in (("regular", FACES[0]), ("hana", HELD), ("fandol", FANDOL)):
        font = Face(desc)
        zh = pictures(rows, font, canonical)
        en = pictures([dict(v, q=v["en"]) for v in rows], font, canonical)
        eligible = {c.question_id for c in zh} & {c.question_id for c in en}
        for lang, cases in (("en", en), ("zh", zh)):
            cohorts["xquad_" + family + "_" + lang] = [
                c for c in cases if c.question_id in eligible
            ]
    record(
        PUBLIC / "xquad_sources.json",
        dict(
            source_sha256=DIGESTS,
            paragraph_decisions=dict(reasons),
            original_squad_dev_ids=sum(v["id"] in squad for v in rows),
            selected_questions=len(rows),
            ids={k: [c.question_id for c in rows] for k, rows in cohorts.items()},
        ),
    )
    return cohorts


def bind():
    folder = WORK / "routed"
    dest = folder / "page"
    dest.mkdir(parents=True)
    base = PARENT / "mapped/page"
    weights = json.loads((base.parent / "page_fit.json").read_text())["weights"]
    for filename, digest in weights.items():
        assert sha(base / filename) == digest
        os.link(base / filename, dest / filename)
    auxiliary = WORK / "stroke_scale.safetensors"
    weights["augmented_typeface.safetensors"] = json.loads(
        (PUBLIC / "stroke_scale_fit.json").read_text()
    )["sha256"]
    assert sha(auxiliary) == weights["augmented_typeface.safetensors"]
    os.link(auxiliary, dest / "augmented_typeface.safetensors")
    record(folder / "page_fit.json", dict(weights=weights))
    return dest


@torch.inference_mode()
def evaluate():
    dest = bind()
    extra = xquad()
    groups = inputs()[0] | extra
    scores = {}
    for arm, model_path, cohorts in (
        ("translation", PARENT / "mapped/page", extra),
        ("routed", dest, groups),
    ):
        model = PreparedGlyphResponder(model_path, "cuda")
        folder = WORK / ("check_" + arm)
        folder.mkdir()
        keys = sorted(
            {(c.p, c.q) for rows in cohorts.values() for c in rows},
            key=lambda t: (len(t[0]) + len(t[1]) * 2, t),
        )
        outputs = {}
        routes = {}
        started = monotonic()
        for i in range(0, len(keys), 16):
            batch = keys[i : i + 16]
            pairs = [(tensor(p), tensor(q)) for p, q in batch]
            ans = model.answer_many(pairs)
            for k, (im, stop), (p, q) in zip(batch, ans, pairs, strict=True):
                outputs[k] = im.numpy().tobytes(), stop
                routes[k] = model.net.writer.uses_augmented_print(p, q, model.fixed)
        scores[arm] = {}
        for name, rows in cohorts.items():
            f = folder / (name + ".npz")
            ys, ends = zip(*(outputs[c.p, c.q] for c in rows), strict=True)
            store_streams(f, ys, ends)
            ys, ends = load_streams(f)
            right = [c.grade(y, s)[2] for c, y, s in zip(rows, ys, ends, strict=True)]
            boards = {}
            for c, ok in zip(rows, right, strict=True):
                boards.setdefault(c.board, []).append(ok)
            scores[arm][name] = dict(
                n=len(rows),
                exact=sum(right),
                boards=len(boards),
                complete=sum(map(all, boards.values())),
                augmented=sum(routes[c.p, c.q] for c in rows),
                sha256=sha(f),
            )
            if arm == "routed" and name == "xquad_hana_zh" and any(right):
                i = right.index(True)
                for key, raw in (
                    ("page", rows[i].p),
                    ("question", rows[i].q),
                    ("answer", ys[i]),
                    ("reference", rows[i].source[0]),
                ):
                    save_image(PUBLIC / "examples" / ("xquad_" + key + ".png"), raw)
        print(arm, len(keys), round(monotonic() - started, 1), flush=True)
    a = scores["routed"]
    original = json.loads((PUBLIC / "result.json").read_text())["scores"]["translation"]
    selected = (
        all(
            a[k]["complete"] > original[k]["complete"]
            for k in ("trial_hana", "trial_fandol")
        )
        and a["trial_regular"]["complete"] >= 14
        and a["archive_familiar"]["complete"] >= 99
        and a["xquad_hana_zh"]["complete"]
        > scores["translation"]["xquad_hana_zh"]["complete"]
        and a["xquad_regular_zh"]["complete"]
        >= scores["translation"]["xquad_regular_zh"]["complete"]
    )
    record(PUBLIC / "routed.json", dict(scores=scores, selected=selected))


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
