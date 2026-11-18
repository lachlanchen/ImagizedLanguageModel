"""Supply authenticated historical participant images to the trained event translators."""

import hashlib
import fcntl
import numpy as np
import torch

from scripts.prepare_source_plate_questions import CACHE, SPLITS
from scripts.evaluate_visual_event_roles import EVAL, INVALID, role_positions
from scripts.evaluate_paired_student import save_streams, predict
from scripts.prepare_visual_event_roles import OUT, ROOT, load, sha, write_new
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_pictured_revision import composer
from scripts.train_visual_event_roles import PARENT, raster

WORK = OUT / "historical"


def examples(states, render):
    report = load(SPLITS / "report.json")
    assert sha(CACHE) == report["raster_cache"]["cache_sha256"]
    pixels = torch.load(CACHE, weights_only=True, map_location="cpu", mmap=True)[
        "pixels"
    ]
    picked = {}
    labels = {}
    rank = {k: i for i, k in enumerate(["bronze", "oracle", "seal", "liushutong"])}
    for split in ("train", "development"):
        for entry in load(SPLITS / (split + ".json")):
            for c in states.intersection(
                {entry["char"], entry["fields"]["simplified"]}
            ):
                for g in entry["glyphs"]:
                    if g["stage"] not in rank:
                        continue
                    candidate = (rank[g["stage"]], g["label"], entry["entry_id"])
                    if c in labels and labels[c] <= candidate:
                        continue
                    cell = np.packbits(pixels[g["raster_index"]].numpy().reshape(1024))
                    assert (
                        hashlib.sha256(cell.tobytes()).hexdigest() == g["raster_sha256"]
                    )
                    if np.all(cell == 255):
                        continue
                    picked[c] = (
                        cell,
                        dict(g, character=c, entry=entry["entry_id"], split=split),
                    )
                    labels[c] = candidate
    from ilm.visual_lm.continuous_glyph_codec_data import render_historic_svg

    for c, (cached, record) in list(picked.items()):
        svg = ROOT.parent / "incoder" / record["source_local_path"]
        assert sha(svg) == record["sha256"]
        paper = np.packbits(render_historic_svg(svg).numpy().reshape(1024))
        assert np.array_equal(paper, cached)
        # The V34 cache encodes luminance (one for paper); this writer encodes ink.
        ink = np.bitwise_not(paper)
        picked[c] = ink, dict(record, ink_sha256=hashlib.sha256(ink).hexdigest())
    return picked


def mix(text, names, render):
    base = raster(text, render).copy()
    for j, c in enumerate(text):
        if c in names:
            base[j] = names[c]
    return base


def summarize(outputs, sources, refs, meta, names):
    report = {}
    for k, seq in outputs.items():
        ordered = [
            role_positions(v, names[r["first"]].tobytes(), names[r["second"]].tobytes())
            for v, r in zip(seq, meta)
        ]
        exact = [np.array_equal(v, t) for v, t in zip(seq, refs)]
        changed = [not np.array_equal(v, s) for v, s in zip(seq, sources)]
        pairs = lambda flags: sum(a and b for a, b in zip(flags[::2], flags[1::2]))
        report[k] = dict(
            ordered_pairs=pairs(ordered),
            ordered_changed_pairs=pairs([a and b for a, b in zip(ordered, changed)]),
            exact_pairs=pairs(exact),
            exact=sum(exact),
            ordered_individual=sum(ordered),
        )
    return report


def run():
    torch.set_num_threads(2)
    WORK.mkdir()
    render = Rasterizer()
    rows = load(EVAL / "rows.json")
    count = len(load(OUT / "roles.json"))
    rows = rows[: 2 * count]
    states = {
        r[n] for r in rows if r["base"] not in INVALID for n in ("first", "second")
    }
    picked = examples(states, render)
    names = {c: x[0] for c, x in picked.items()}
    glyphs = {render.tile(c) for r in rows for c in r["source"] + r["target"]}
    assert not glyphs.intersection(x.tobytes() for x in names.values())
    sources = []
    refs = []
    meta = []
    for r in rows:
        a, b = r["first"], r["second"]
        if (
            r["base"] in INVALID
            or a not in names
            or b not in names
            or np.array_equal(names[a], names[b])
        ):
            continue
        mapping = {a: names[a], b: names[b]}
        sources.append(mix(r["source"], mapping, render))
        refs.append(mix(r["target"], mapping, render))
        meta.append(r)
    assert len(sources) > 1 and all(
        a["base"] == b["base"] for a, b in zip(meta[::2], meta[1::2])
    )
    cmp = composer()
    outputs = {
        "source": sources,
        "reference": refs,
        "composer": [cmp.generate(x) for x in sources],
    }
    for k in ("parent", "replay", "swapped"):
        path = PARENT if k == "parent" else OUT / (k + ".safetensors")
        outputs[k], _, _, _ = predict(path, sources, pool_routes=True)
    for k, v in outputs.items():
        save_streams(k, v, WORK)
    write_new(WORK / "rows.json", meta)
    write_new(WORK / "forms.json", {c: x[1] for c, x in picked.items()})
    write_new(
        WORK / "scores.json",
        dict(
            pairs=len(meta) // 2,
            missing=sorted(states - set(names)),
            cache=sha(CACHE),
            scores=summarize(outputs, sources, refs, meta, names),
            files={p.name: sha(p) for p in WORK.iterdir()},
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
