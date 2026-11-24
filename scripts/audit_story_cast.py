"""Replay PNG inputs and rename the cast with previously unseen source bitmaps."""

from collections import Counter
import gzip
import hashlib
import io
import json
import numpy as np
import torch
from PIL import Image
from scripts.check_mention_transfer import datasets
from scripts.train_story_cast import partitions
from scripts.prepare_story_mentions import PUBLIC, MODEL, RUN, read, dump
from scripts.story_mention_options import symbols, collapse
from scripts.write_story_mentions import render_cells, tiles, display
from scripts.write_masked_mentions import MaskedReader, cast_cells, cast_options
from scripts.prepare_visual_sequence_corpus import Rasterizer


def png(image):
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        buf.seek(0)
        with Image.open(buf) as reloaded:
            return reloaded.copy()


def rename(cells, q, panel):
    indices, slots, spans, marked = collapse(
        symbols(cells), q, cast_options(panel, cells, q[0])
    )
    count = max(slots, default=-1) + 1
    rng = np.random.default_rng(11380)
    new = rng.integers(0, 256, (count, 128), dtype=np.uint8)
    if (
        set(map(bytes, new)) & set(map(bytes, cells))
        or len(set(map(bytes, new))) != count
    ):
        raise ValueError("Renamed marks collide")
    replaced = cells[indices].copy()
    for i, s in enumerate(slots):
        if s >= 0:
            replaced[i] = new[s]
    novel = cast_cells([s[None] for s in new])
    mapping = {
        cells[a:b].tobytes(): new[s].tobytes()
        for s, (a, b) in zip(slots, spans)
        if s >= 0
    }
    return render_cells(replaced, marked), novel, mapping


def audit():
    torch.set_num_threads(2)
    raster = Rasterizer()
    reader = MaskedReader(trained_cast=True)
    external, _ = datasets()
    parts, _ = partitions()
    receipt = {}
    allgroups = [(k + "-official", v) for k, v in external.items()] + [
        (k + "-fresh", v[1]) for k, v in parts.items()
    ]
    for lang in ("chinese", "english"):
        rs = [r for r in read(RUN / "test.json") if r["lang"] == lang]
        allgroups.append(
            (
                lang + "-test",
                [
                    dict(
                        text=r["text"],
                        query=r["query"],
                        cast=[(a, b) for a, b, e in r["names"]],
                    )
                    for r in rs
                ],
            )
        )
    for key, rows in allgroups:
        saved = json.loads(
            gzip.decompress(
                (PUBLIC / ("cast-" + key + "-decisions.json.gz")).read_bytes()
            )
        )["ink"]
        counts = Counter()
        digest = hashlib.sha256()
        for i, (r, expected) in enumerate(zip(rows, saved, strict=True)):
            cells = tiles(r["text"], raster)
            q = r["query"]
            names = list(dict.fromkeys(r["text"][a:b] for a, b in r["cast"]))
            page = png(render_cells(cells, q))
            panel = png(cast_cells([tiles(s, raster) for s in names]))
            result, ink = reader.answer(page, panel)
            expected = None if expected is None else tuple(expected)
            counts["pages"] += 1
            counts["interval_reproduced"] += expected == result
            truth = None if expected is None else cells[slice(*expected)].tobytes()
            got = None if ink is None else ink.tobytes()
            counts["ink_reproduced"] += truth == got
            digest.update(
                b"\0" if got is None else b"\1" + hashlib.sha256(got).digest()
            )
            if i % 4 == 0:
                renamed, cast, mapping = rename(cells, q, panel)
                _, changed = reader.answer(png(renamed), png(cast))
                counts["substitutions"] += 1
                counts["equivariant"] += (
                    None if changed is None else changed.tobytes()
                ) == (None if ink is None else mapping[got])
        receipt[key] = dict(counts, answer_sequence_sha256=digest.hexdigest())
        print(key, receipt[key], flush=True)
    dump(PUBLIC / "png-replay.json", receipt)


def authored():
    raster = Rasterizer()
    reader = MaskedReader(trained_cast=True)
    out = []
    for lang, text, names, refs in [
        (
            "Chinese",
            "清溪园和星河塔隔水相望。前者的门票是12元，后者收35元。",
            ["清溪园", "星河塔"],
            ["前者", "后者"],
        ),
        (
            "English",
            "Willow Park faces Star Tower. The former charges 12 yuan and the latter charges 35 yuan.",
            ["Willow Park", "Star Tower"],
            ["former", "latter"],
        ),
    ]:
        cells = tiles(text, raster)
        cast = cast_cells([tiles(n, raster) for n in names])
        for i, ref in enumerate(refs):
            start = text.index(ref)
            page = render_cells(cells, (start, start + len(ref)))
            loc, answer = reader.answer(png(page), png(cast))
            dest = MODEL / "examples" / f"{lang.lower()}-{i+1}"
            dest.mkdir(parents=True, exist_ok=True)
            page.save(dest / "question.png")
            cast.save(dest / "cast.png")
            if answer is not None:
                display(answer).save(dest / "answer.png")
            out.append(
                dict(
                    language=lang,
                    marked=ref,
                    wanted=names[i],
                    output=None if loc is None else text[slice(*loc)],
                    interval=loc,
                )
            )
    dump(PUBLIC / "former-latter.json", out)


if __name__ == "__main__":
    torch.set_num_threads(2)
    authored()
    audit()
