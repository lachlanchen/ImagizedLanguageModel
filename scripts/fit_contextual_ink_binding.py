"""Fit a physical neighborhood model, then test the writer on historic and modern ink."""

from collections import Counter
from pathlib import Path
import fcntl
import random
import time

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.contextual_ink_binding import ContextualInkBinding, InkPermutation
from scripts.evaluate_historical_event_roles import WORK, summarize
from scripts.evaluate_paired_student import read_streams, save_streams
from scripts.evaluate_visual_event_roles import EVAL
from scripts.prepare_visual_event_roles import OUT, ROOT, EVENT, load, sha, write_new
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_causal_glyph_translator import read_rows, tiles
from scripts.train_paired_student import restore
from scripts.train_visual_event_roles import raster

RESULT = OUT / "binding"


def fit(render):
    raw = tiles()
    rows = read_rows("train")
    used = {j for r in rows for field in ("x", "z") for j in r[field]}
    known = set(bytes(raw[j]) for j in used)
    training = load(OUT / "train.json")
    aligned = []
    for r in training:
        seq = raster(r["source"], render)
        src = r["source"]
        aligned.append((seq, [src.index(r["a"]), src.index(r["b"])]))
    model = ContextualInkBinding.fit(known, aligned)
    path = RESULT / "binding.msgpack.gz"
    model.save(path)
    return ContextualInkBinding.restore(path)


@torch.inference_mode()
def write(model, source, pairs):
    dev = next(model.parameters()).device
    pic = torch.from_numpy(np.unpackbits(source, axis=1)).bool()[None].to(dev)
    lens = torch.tensor([len(source)], device=dev)
    p = InkPermutation(pairs, dev) if pairs else None
    result, n, stop, _ = model.generate(pic, lens, pool_routes=True, ink_permutation=p)
    return np.packbits(result[0, : n.item()].cpu().numpy(), axis=1), bool(stop.item())


def run(reuse_binding=None):
    RESULT.mkdir()
    torch.set_num_threads(2)
    render = Rasterizer()
    began = time.monotonic()
    if reuse_binding is None:
        binder = fit(render)
    else:
        from shutil import copyfile

        copyfile(reuse_binding, RESULT / "binding.msgpack.gz")
        binder = ContextualInkBinding.restore(RESULT / "binding.msgpack.gz")
    fitted = time.monotonic() - began
    model = restore(OUT / "swapped.safetensors").cuda().eval()
    historic = load(WORK / "rows.json")
    forms = load(WORK / "forms.json")
    sources = read_streams("source", WORK)
    refs = read_streams("reference", WORK)
    names = {}
    for row, x in zip(historic, sources):
        for k in ("first", "second"):
            c = row[k]
            cell = x[row["source"].index(c)]
            assert sha_bytes(cell) == forms[c]["ink_sha256"]
            names[c] = cell
    results = {
        k: read_streams(k, WORK) for k in ("source", "reference", "composer", "swapped")
    }
    coverage = {}
    times = {}
    for arm in ("frequency", "context"):
        start = time.monotonic()
        selected = []
        outputs = []
        for x in sources:
            permutation = binder.choose(x, frequency_only=arm == "frequency")
            selected.append(len(permutation))
            outputs.append(write(model, x, permutation)[0])
        save_streams(arm, outputs, RESULT)
        results[arm] = outputs
        coverage[arm] = dict(Counter(selected))
        times[arm] = time.monotonic() - start
    ordinary = load(EVAL / "rows.json")
    printed = read_streams("source", EVAL)
    native = read_streams("swapped", EVAL)
    expected = read_streams("reference", EVAL)
    cut = 2 * len(load(OUT / "roles.json"))
    actual = []
    invoked = []
    check = []
    for i, x in enumerate(printed):
        perm = binder.choose(x)
        if perm:
            invoked.append(i)
            actual.append(write(model, x, perm)[0])
        else:
            actual.append(native[i])
            if i in (0, cut, cut + 11, cut + 23):
                assert np.array_equal(write(model, x, [])[0], native[i])
                check.append(i)
    save_streams("modern", actual, RESULT)
    edits = [
        Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, t)))
        for x, t in zip(actual[cut:], expected[cut:])
    ]
    modern = dict(
        invoked=invoked,
        broad_edits=sum(edits),
        broad_changed=sum(
            not np.array_equal(x, y) for x, y in zip(actual[cut:], native[cut:])
        ),
        printed_role_changed=sum(
            not np.array_equal(x, y) for x, y in zip(actual[:cut], native[:cut])
        ),
        no_binding_replays=check,
    )
    marker = next(iter(names.values()))
    false = []
    for i, row in enumerate(historic):
        x = raster(row["source"], render).copy()
        m = EVENT.search(row["source"])
        assert m is not None
        x[m.start(2)] = marker
        if any(a == marker.tobytes() for a, b in binder.choose(x)):
            false.append(i)
    sample = []
    chapters = set()
    verbs = set()
    for diverse in [True, False]:
        for j, r in enumerate(historic[::2]):
            if (
                j not in sample
                and r["chapter"] not in chapters
                and (not diverse or r["verb"] not in verbs)
            ):
                sample.append(j)
                chapters.add(r["chapter"])
                verbs.add(r["verb"])
            if len(sample) == 8:
                break
        if len(sample) == 8:
            break
    assert len(sample) == 8
    rng = random.Random(71932)
    ids = [2 * j + b for j in sample for b in (0, 1)]
    rng.shuffle(ids)
    lex = {}
    for r in read_rows("train"):
        for c in r["new"]:
            glyph = render.tile(c)
            if glyph not in lex:
                lex[glyph] = c
    for r in ordinary:
        for c in r["source"] + r["target"]:
            lex.setdefault(render.tile(c), c)
    for c, cell in names.items():
        lex[bytes(cell)] = "⟦" + c + "⟧"
    review = []
    mapping = []
    unknown = 0
    for i in ids:
        choices = {}
        for k, v in results.items():
            words = [lex.get(bytes(cell), "▧") for cell in v[i]]
            unknown += words.count("▧")
            choices[k] = "".join(words)
        values = list(dict.fromkeys(choices.values()))
        rng.shuffle(values)
        review.append(dict(source=choices["source"], choices=values))
        mapping.append(
            dict(row=i, options={k: values.index(v) for k, v in choices.items()})
        )
    write_new(RESULT / "review.json", review)
    write_new(RESULT / "mapping.json", mapping)
    write_new(
        RESULT / "summary.json",
        dict(
            setup_seconds=fitted,
            reused_binding=reuse_binding is not None,
            bank=len(binder.bank),
            contexts=len(binder.counts),
            fit_sha256=sha(RESULT / "binding.msgpack.gz"),
            writer_sha256=sha(OUT / "swapped.safetensors"),
            coverage=coverage,
            seconds=times,
            modern=modern,
            historical=summarize(results, sources, refs, historic, names),
            unfamiliar_predicate_bound=false,
            predicate_controls=len(historic),
            review_pairs=sample,
            unmapped_review_cells=unknown,
        ),
    )


def sha_bytes(cell):
    from hashlib import sha256

    return sha256(cell.tobytes()).hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-binding", type=Path)
    args = parser.parse_args()
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.reuse_binding)
