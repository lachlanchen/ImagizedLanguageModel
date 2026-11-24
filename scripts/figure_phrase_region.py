"""Compose the held gain and loss from the original emitted ink and release their inputs."""

import math
from hashlib import sha256
import numpy as np
from scripts.prepare_region_writer import MODEL as BASE, Ink
from scripts.train_phrase_region import MODEL
from scripts.prepare_phrase_confirmation import WORK, PUB
from scripts.prepare_open_exam import read, dump
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.figure_recurrent_evidence import Plate, MUTED, TEAL


def breaks(text):
    count = max(1, math.ceil(len(text) / 74))
    points = [0]
    for i in range(1, count):
        aim = round(len(text) * i / count)
        nearby = [
            k
            for k in range(max(points[-1] + 1, aim - 11), min(len(text), aim + 12))
            if text[k - 1].isspace()
        ]
        points.append(min(nearby, key=lambda k: abs(k - aim)) if nearby else aim)
    return [*points, len(text)]


def figure(
    work=WORK,
    public=PUB,
    model_dir=MODEL,
    candidate="phrase",
    prefix="confirmation",
    name="Figure_165_writing_with_visible_phrases",
    label="Variable run",
):
    rows = read(work / "confirmation.json")
    predictions = {
        a: unpack(work / (prefix + "-" + a + ".json.gz"))["original"]
        for a in ("single", candidate)
    }
    ink = Ink(np.load(BASE / "palette.npy"))

    def bits(s):
        return np.asarray(
            [np.frombuffer(ink.tiles[i], np.uint8) for i in ink.encode(s)]
        )

    def matches(i, a):
        v = predictions[a][i]
        return v["stop"] and any(
            bits(s).tobytes().hex() == v["ink_hex"] for s in rows[i]["answers"]
        )

    pool = [i for i, r in enumerate(rows) if r["task"] == "statements"]
    chosen = []
    for best, other in ((candidate, "single"), ("single", candidate)):
        options = [i for i in pool if matches(i, best) and not matches(i, other)]
        fallback = [
            i
            for i in pool
            if not matches(i, "single")
            and not matches(i, candidate)
            and i not in chosen
        ]
        chosen.append((options or fallback)[0])
    assets = public / "plates"
    assets.mkdir(parents=True, exist_ok=True)
    plate = Plate(name, assets)
    cases = []
    for j, i in enumerate(chosen):
        r = rows[i]
        folder = model_dir / "examples" / str(j + 1)
        folder.mkdir(parents=True, exist_ok=True)

        def raster(k, s):
            data = bits(s)
            draw(folder, k, data)
            return dict(cells=len(data), png=k + ".png")

        q = raster("question", r["q"])
        regions = [raster("source-" + str(k), b) for k, b in enumerate(r["blocks"])]
        dump(folder / "input.json", dict(question=q, regions=regions))
        ref = raster("reference", r["answers"][0])
        fields = [
            ("Question", "question", q["cells"]),
            ("Human reply", "reference", ref["cells"]),
        ]
        summary = dict(
            family=r["family"], question=r["q"], human=r["answers"][0], source_index=i
        )
        for name, arm in (("Single glyph", "single"), (label, candidate)):
            v = predictions[arm][i]
            data = bytes.fromhex(v["ink_hex"])
            assert sha256(data).hexdigest() == v["sha256"]
            draw(folder, arm, np.frombuffer(data, np.uint8).reshape(-1, 128))
            fields.append((name, arm, v["cells"]))
            summary[arm] = {
                k: v[k] for k in ("text", "sha256", "stop", "cells", "selected")
            }
            summary[arm]["bitmap_exact"] = matches(i, arm)
        top = 135 * j
        title = (
            "The run writer recovers the full sentence"
            if matches(i, candidate)
            else "The single-glyph writer preserves the full sentence"
            if matches(i, "single")
            else "Both writers diverge from the human answer"
        )
        plate.label(4, top + 1, "ab"[j], 12, True)
        plate.label(22, top + 1, title, 10, True)
        texts = [
            r["q"],
            r["answers"][0],
            predictions["single"][i]["text"],
            predictions[candidate][i]["text"],
        ]
        assert all(len(t) == n for t, (_, _, n) in zip(texts, fields))
        splits = [breaks(t) for t in texts]
        lines = [len(v) - 1 for v in splits]
        dy = min(19, (108 - 3 * 4) / sum(lines))
        y = top + 20
        for (name, f, n), nline, edges in zip(fields, lines, splits):
            plate.label(7, y + 1, name, 8, True, TEAL if f == candidate else MUTED)
            if n == 0:
                plate.label(87, y, "[empty]", 8)
            for z in range(nline) if n else ():
                plate.strip(
                    f"{j}-{f}-{z}",
                    folder / (f + ".png"),
                    n,
                    87,
                    y + dy * z,
                    448,
                    dy,
                    edges[z],
                    edges[z + 1],
                )
            y += dy * nline + 4
        cases.append(summary)
    plate.rule(132)
    dump(public / "examples.json", cases)
    dump(public / "plates.json", [plate.render()])


if __name__ == "__main__":
    figure()
