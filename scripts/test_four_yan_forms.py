"""Read and verify four historical catalogue forms on a pictured source page."""

from collections import Counter
from hashlib import sha256
import json
import sqlite3
import numpy as np
import torch
from ilm.visual_lm.continuous_glyph_codec_data import render_historic_svg
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.figure_role_change_writer import png
from scripts.ordered_image_corpus import STAGES, PLATES
from scripts.prepare_role_change_actions import ROOT
from scripts.train_individual_clause_writer import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha, in_cells
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

EN = dict(oracle="oracle bone", bronze="bronze", seal="seal", liushutong="Liushutong")


def sources():
    root = ROOT.parent / "incoder"
    db = root / "data/historic/etymology.sqlite3"
    source = {}
    receipt = []
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
        for c in ("言", "征"):
            for stage in STAGES:
                rows = conn.execute(
                    "select label, local_path from glyphs join chars on chars.id=glyphs.char_id where chars.char=? and stage=? and source_site='hanziyuan' order by label, glyphs.id",
                    (c, stage),
                ).fetchall()
                label, relative = next((n, p) for n, p in rows if (root / p).is_file())
                path = root / relative
                tile = np.packbits(
                    render_historic_svg(path).numpy().reshape(-1) == 0
                ).tobytes()
                assert len(tile) == 128 and any(tile)
                source[c, stage] = tile
                receipt.append(
                    dict(
                        char=c,
                        stage=stage,
                        label=label,
                        svg_sha=sha(path),
                        raster_sha=sha256(tile).hexdigest(),
                    )
                )
    assert len(set(source.values())) == 8
    return source, receipt


def page(face, source, stages):
    return b"".join(
        face(c + "：")
        + b"".join(face(STAGES[s]) + source[c, s] for s in stages)
        + face("。")
        for c in ("征", "言")
    )


def question(face, stages, lang):
    if lang == "zh":
        text = (
            "請介紹「言」的字形，依"
            + "、".join(STAGES[s] for s in stages)
            + "的順序作答。"
        )
    else:
        text = (
            "Show the forms of 言 in this order: "
            + ", ".join(EN[s] for s in stages)
            + "."
        )
    return face(text)


@torch.inference_mode()
def run():
    face = Face(FACES[0])
    positive, receipt = sources()
    d = PUBLIC / "yan"
    d.mkdir(exist_ok=True)
    for c, s in positive:
        png(positive[c, s], d / f"{c}_{s}.png")
    net = PreparedGlyphResponder(WORK / "clause_aligned/page", "cuda")
    ids = list(STAGES)
    results = []
    png(b"".join(positive["言", s] for s in ids), d / "four_reference.png")
    for polarity in ("print", "legacy"):
        source = (
            positive
            if polarity == "print"
            else {k: bytes(b ^ 255 for b in v) for k, v in positive.items()}
        )
        for layout in ("forward", "reverse"):
            idx = ids if layout == "forward" else ids[::-1]
            for order in ("forward", "reverse"):
                wanted = ids if order == "forward" else ids[::-1]
                for lang in ("zh", "en"):
                    for route in ("whole", "two", "source_assisted"):
                        chunks = (
                            [wanted] if route == "whole" else [wanted[:2], wanted[2:]]
                        )
                        pairs = []
                        title = f"{polarity}_{layout}_{order}_{lang}_{route}"
                        for j, chunk in enumerate(chunks):
                            visible = (
                                [s for s in idx if s in chunk]
                                if route == "source_assisted"
                                else idx
                            )
                            p = page(face, source, visible)
                            q = question(face, chunk, lang)
                            pairs.append((tensor(p), tensor(q)))
                            if (
                                polarity == "print"
                                and layout == order == "forward"
                                and lang == "zh"
                            ):
                                png(p, d / f"{route}_page{j}.png")
                                png(q, d / f"{route}_question{j}.png")
                        ys = net.answer_many(pairs)
                        ink = b"".join(im.numpy().tobytes() for im, _ in ys)
                        ended = all(end for _, end in ys)
                        png(ink, d / (title + ".png"))
                        lookup = {v: k for k, v in source.items()}
                        actual = [lookup[v] for v in in_cells(ink) if v in lookup]
                        expected = [("言", s) for s in wanted]
                        results.append(
                            dict(
                                polarity=polarity,
                                layout=layout,
                                order=order,
                                lang=lang,
                                route=route,
                                ended=ended,
                                actual=actual,
                                all_four=bool(ended and actual == expected),
                                distinct_correct=len(set(expected) & set(actual)),
                                image=title + ".png",
                                sha256=sha(d / (title + ".png")),
                            )
                        )
    old = json.loads((PLATES / "yan/source.json").read_text())
    pixels = np.load(PLATES / "pixels.npy")
    for r in old["rows"]:
        path = (
            ROOT.parent
            / "incoder/data/historic/glyphs"
            / r["char"]
            / r["stage"]
            / (r["label"] + ".svg")
        )
        raster = render_historic_svg(path).numpy().reshape(-1)
        assert (
            sha(path) == r["svg"]
            and np.packbits(raster).tobytes() == pixels[r["tile"]].tobytes()
        )
    record(
        PUBLIC / "yan_four_forms.json",
        dict(
            previous_white_one_rasters_verified=4,
            prior_card_split=old["split"],
            prior_card=[
                {k: r[k] for k in ("char", "stage", "label")} for r in old["rows"]
            ],
            sources=receipt,
            counts={
                polarity + "_" + r: dict(
                    Counter(
                        x["all_four"]
                        for x in results
                        if x["route"] == r and x["polarity"] == polarity
                    )
                )
                for polarity in ("print", "legacy")
                for r in ("whole", "two", "source_assisted")
            },
            cases=results,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        torch.set_num_threads(2)
        run()
