"""Measure both correspondence fits on original questions, erasures and historical cards."""

from collections import Counter
from dataclasses import replace
import json
from time import monotonic
import unicodedata

import numpy as np
import torch

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.audit_interval_teacher import compare, save_image
from scripts.evaluate_interval_teacher import trials
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.test_origin_catalogue_writer import assemble as catalogue_cases
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_stroke_scale import ARMS, FANDOL, PUBLIC, WORK
from scripts.train_typeface_writer import FACES, HELD, PREVIOUS, WORK as PARENT, Face
from scripts.typeface_writer_cases import Styled, extra_sources, pictures


def inputs():
    development = PREVIOUS / "pairs_dev.jsonl"
    assert (
        sha(development)
        == json.loads((PREVIOUS / "corpus.json").read_text())["fresh"]["sha256"]
    )
    sources = {
        "development": list(map(json.loads, development.read_text().splitlines())),
        "trial": extra_sources()[0],
    }
    fonts = {"hana": Face(HELD), "fandol": Face(FANDOL), "regular": Face(FACES[0])}
    groups, coverage, occurrences = {}, {}, {}
    for split, rows in sources.items():
        for family, font in fonts.items():
            key = split + "_" + family
            cases = pictures(rows, font, fonts["regular"])
            kept = {c.question_id for c in cases}
            omitted = [v["id"] for v in rows if v["id"] not in kept]
            missing = sorted(
                {
                    c
                    for v in rows
                    for t in (v["p"], v["q"], *v["refs"])
                    for c in t
                    if not c.isspace()
                    and (
                        ord(c) not in font.coverage
                        or ord(c) not in fonts["regular"].coverage
                    )
                }
            )
            coverage[key] = dict(
                question_ids=[v.question_id for v in cases],
                omitted_question_ids=omitted,
                omitted_pairs=len(omitted) // 2,
                unavailable_characters=missing,
            )
            if not cases:
                raise ValueError(f"No renderable paired passages in {key}")
            groups[key] = cases
            if family != "regular":
                groups[key + "_erased"] = [replace(c, q=bytes(128)) for c in cases]
            occurrences[key] = (
                font,
                Counter(
                    c
                    for v in rows
                    if v["id"] in kept
                    for t in (v["p"], v["q"])
                    for c in t
                ),
            )
    for group, rows in catalogue_cases().items():
        groups["archive_" + group] = [
            Styled(str(i // 2), p, q, (y,), (y,), str(i))
            for i, (p, q, y, *_) in enumerate(rows)
        ]
    control = trials()
    for group in ("yan", "open_absent_content_ink"):
        groups[group] = [
            Styled(c.board, c.p, c.q, tuple(c.refs), tuple(c.refs), str(i))
            for i, c in enumerate(control[group])
        ]
    record(
        PUBLIC / "coverage.json",
        dict(cohorts=coverage, held_font_sha256=sha(FANDOL[0])),
    )
    return groups, fonts["regular"], occurrences


def optical(model, canonical, font, occurrences):
    extra = sorted({font.cell(c) for c in occurrences} - model.fixed.keys())
    resolved = {}
    for pos in range(0, len(extra), 256):
        chunk = extra[pos : pos + 256]
        ids, accept = model.net.writer.typeface_map.identify(
            tensor(b"".join(chunk)).cuda()
        )
        resolved.update(
            (k, i if ok else None)
            for k, i, ok in zip(chunk, ids.tolist(), accept.tolist(), strict=True)
        )
    tally = Counter()
    per_char = {}
    for c, n in occurrences.items():
        ink = font.cell(c)
        target = model.fixed.get(canonical.cell(c))
        predicted = model.fixed.get(ink, resolved.get(ink))
        status = (
            "abstained"
            if predicted is None
            else "correct"
            if predicted == target
            else "wrong"
        )
        per_char[c] = status
        tally[status] += n
        tally["occurrences"] += n
        if target is None:
            tally["outside_output_inventory"] += n
        if unicodedata.category(c).startswith("P"):
            tally["punctuation_" + status] += n
        if np.unpackbits(np.frombuffer(ink, np.uint8)).sum() <= 48:
            tally["at_most_48_ink_pixels_" + status] += n
    return dict(tally), per_char


@torch.inference_mode()
def run():
    groups, canonical, occurrences = inputs()
    paths = json.loads((PARENT / "mapped/page_fit.json").read_text())["weights"]
    writer_path = PARENT / "mapped/page"
    assert all(sha(writer_path / k) == v for k, v in paths.items())
    model = PreparedGlyphResponder(writer_path, "cuda")
    order = sorted(
        {(c.p, c.q) for rows in groups.values() for c in rows},
        key=lambda pq: (len(pq[0]) + len(pq[1]) * 2, pq),
    )
    scores, individual, pixels, identities = {}, {}, {}, {}
    for arm in ARMS:
        folder = WORK / arm
        folder.mkdir()
        checkpoint = WORK / (arm + ".safetensors")
        assert (
            sha(checkpoint)
            == json.loads((PUBLIC / (arm + "_fit.json")).read_text())["sha256"]
        )
        model.net.writer.attach_typeface_map(checkpoint)
        scores[arm], identities[arm] = {}, {}
        raw = {}
        for key, (font, chars) in occurrences.items():
            counts, by_char = optical(model, canonical, font, chars)
            identities[arm][key] = counts
            raw[key] = by_char
        record(folder / "optical_by_char.json", raw)
        started = monotonic()
        outputs = {}
        for i in range(0, len(order), 16):
            part = order[i : i + 16]
            generated = model.answer_many([(tensor(p), tensor(q)) for p, q in part])
            outputs.update(
                (key, (im.numpy().tobytes(), stop))
                for key, (im, stop) in zip(part, generated, strict=True)
            )
        for key, rows in groups.items():
            f = folder / (key + ".npz")
            images, stopped = zip(*(outputs[c.p, c.q] for c in rows), strict=True)
            store_streams(f, images, stopped)
            ys, ends = load_streams(f)
            right = np.array(
                [c.grade(y, end)[2] for c, y, end in zip(rows, ys, ends, strict=True)]
            )
            individual[arm, key] = right
            pairs = {}
            for c, ok in zip(rows, right, strict=True):
                pairs.setdefault(c.board, []).append(bool(ok))
            scores[arm][key] = dict(
                n=len(rows),
                exact=int(right.sum()),
                boards=len(pairs),
                complete=sum(map(all, pairs.values())),
                stops=sum(ends),
                sha256=sha(f),
            )
            if key == "trial_hana":
                pixels[arm] = ys
        print(
            arm,
            "generated",
            len(order),
            "unique pairs in",
            round(monotonic() - started, 1),
            "s",
            flush=True,
        )
    record(PUBLIC / "optical.json", identities)
    shared = set(c.board for c in groups["trial_hana"]) & {
        c.board for c in groups["trial_fandol"]
    }
    paired = {}
    for key in ("trial_hana", "trial_fandol"):

        def complete(arm):
            flags = individual[arm, key]
            return flags[::2] & flags[1::2]

        common = np.array([c.board in shared for c in groups[key][::2]])
        paired[key] = dict(
            all=compare(complete(ARMS[1]), complete(ARMS[0])),
            shared=compare(complete(ARMS[1])[common], complete(ARMS[0])[common]),
        )
    study = scores["stroke_scale"]
    accepted = all(
        study[key]["complete"] > scores["translation"][key]["complete"]
        for key in paired
    )
    accepted = (
        accepted
        and study["trial_regular"]["complete"] >= 14
        and study["archive_familiar"]["complete"] >= 99
    )
    examples = {}
    (PUBLIC / "examples").mkdir(exist_ok=True)
    right = individual["stroke_scale", "trial_hana"]
    for tag, mask in (
        ("gain", right & ~individual["translation", "trial_hana"]),
        ("error", ~right),
        ("correct", right),
    ):
        ix = np.flatnonzero(mask)
        if not len(ix):
            continue
        i = int(ix[0])
        c = groups["trial_hana"][i]
        examples[tag] = dict(index=i, id=c.question_id)
        for name, raw in (
            ("page", c.p),
            ("question", c.q),
            ("reference", c.source[0]),
            ("control", pixels["translation"][i]),
            ("answer", pixels["stroke_scale"][i]),
        ):
            save_image(PUBLIC / "examples" / (tag + "_" + name + ".png"), raw)
    record(
        PUBLIC / "result.json",
        dict(
            scores=scores,
            paired=paired,
            selected=accepted,
            examples=examples,
            shared_passages=len(shared),
            writer_weights=paths,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
