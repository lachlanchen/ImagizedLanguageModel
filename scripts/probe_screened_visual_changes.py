"""Perturb one matched glyph on reviewed training sources and score complete free image sequences."""

import fcntl
import hashlib
import re
import numpy as np
import torch
from scripts.evaluate_paired_student import read_streams
from scripts.evaluate_screened_visual_targets import (
    generate,
    raster,
    Rasterizer,
    save_streams,
)
from scripts.train_screened_visual_targets import (
    ROOT,
    OUT,
    DATA,
    EVIDENCE,
    PARENT,
    load,
    write_new,
    sha,
    forced_matches,
)

DIGITS = "二三四五六七八九"
# Numerals in names, listed offices, and idiomatic quantities need separate analysis.
NUMBER = re.compile("[" + DIGITS + "](?=[年月层里百千十篇])")


def shifted(cell):
    image = np.unpackbits(cell).reshape(32, 32)
    if image[:, -1].any() or not image.any():
        return None
    return np.packbits(np.pad(image[:, :-1], ((0, 0), (1, 0))))


def prepare():
    render = Rasterizer()
    with np.load(DATA / "training.npz") as z:
        bank = z["cells"]
        xs = z["x_offsets"]
        source = z["source"]
        trained = {
            hashlib.sha256(bank[source[a:b]].tobytes()).digest()
            for a, b in zip(xs[:-1], xs[1:])
        }
    supervised = load(EVIDENCE / "accepted.json")
    known = {bytes(v) for v in bank}
    known.update(render.tile(c) for r in supervised for c in r["target"])
    cases = []
    xout = []
    yout = []
    alts = []
    for i, r in enumerate(supervised):
        s, t = r["source"], r["target"]
        x, y = raster(s, render), raster(t, render)
        _, matches = forced_matches(list(map(bytes, x)), list(map(bytes, y)))
        lookup = dict(matches)

        def add(kind, si, tj, new, old_allowed):
            src = x.copy()
            src[si] = new
            if hashlib.sha256(src.tobytes()).digest() in trained:
                return False
            dst = y.copy()
            dst[tj] = new
            xout.append(src)
            yout.append(dst)
            alts.append(y if old_allowed else dst)
            cases.append(
                dict(
                    kind=kind,
                    training_position=i,
                    row=r["row"],
                    source_position=si,
                    target_position=tj,
                )
            )
            return True

        # Rotate deterministic eligibility order across rows, without inspecting any predictions.
        for si, tj in sorted(
            matches, key=lambda v: hashlib.sha256(f"shift/{i}/{v[0]}".encode()).digest()
        ):
            if not ("\u3400" <= s[si] <= "\u9fff"):
                continue
            cell = shifted(x[si])
            if (
                cell is not None
                and bytes(cell) not in known
                and add("shift", si, tj, cell, True)
            ):
                break
        for hit in NUMBER.finditer(s):
            si = hit.start()
            if si not in lookup or t[lookup[si] : lookup[si] + 2] != s[si : si + 2]:
                continue
            c = DIGITS[(DIGITS.index(s[si]) + 2) % len(DIGITS)]
            pic = np.frombuffer(render.tile(c), np.uint8)
            if add("number", si, lookup[si], pic, False):
                cases[-1].update(original=s[si], substitute=c)
                break
    assert len({(r["kind"], r["row"]) for r in cases}) == len(cases)
    return cases, xout, yout, alts


def run():
    cases, x, y, alts = prepare()
    registration = EVIDENCE / "change_probes.json"
    write_new(
        registration,
        dict(cases=cases, inputs=[hashlib.sha256(v.tobytes()).hexdigest() for v in x]),
    )
    folder = OUT / "changes"
    folder.mkdir()
    for k, v in (("source", x), ("target", y), ("canonical", alts)):
        save_streams(k, v, folder)
    orig = read_streams("screened_target", OUT / "training_replay")
    scores = {}
    for name, path in (
        ("parent", PARENT),
        ("book", OUT / "book.safetensors"),
        ("screened", OUT / "screened.safetensors"),
    ):
        generated, stops = generate(path, x)
        save_streams(name, generated, folder)
        prev = read_streams(name, OUT / "training_replay")
        checks = []
        for k, (r, v) in enumerate(zip(cases, generated)):
            i = r["training_position"]
            exact = bool(
                stops[k] and (np.array_equal(v, y[k]) or np.array_equal(v, alts[k]))
            )
            checks.append(
                dict(
                    probe=k,
                    base_exact=np.array_equal(prev[i], orig[i]),
                    complete=exact,
                    copied_changed_image=np.array_equal(v, y[k]) and stops[k],
                    unchanged_base_output=np.array_equal(v, prev[i]),
                    stopped=stops[k],
                )
            )
        scores[name] = checks
    write_new(
        EVIDENCE / "change_scores.json",
        dict(
            scores=scores,
            registration_sha=sha(registration),
            streams={
                k: sha(folder / (k + ".npz"))
                for k in ("source", "target", "canonical", "parent", "book", "screened")
            },
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        run()
