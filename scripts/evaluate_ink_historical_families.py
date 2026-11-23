"""Test autonomous two-form writing across held-out historical raster families."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.optical_role_reader import embeddings, read_many
from scripts.evaluate_grounded_ink_editor import first_region
from scripts.figure_role_change_writer import png
from scripts.prepare_grounded_ink_editor import ROOT, read, record, sha
from scripts.respond_grounded_ink import load
from scripts.train_optical_role_adapter import BASE
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

SRC = ROOT / "data/source_plates_v1"
PUBLIC = ROOT / "references/evidence/ink_historical_families_20261124"
WORK = ROOT / "artifacts/ink_historical_families_v1"


def prepare():
    report = read(SRC / "report.json")
    assert sha(SRC / "pixels.npy") == report["pixels_sha256"]
    tiles = np.load(SRC / "pixels.npy", mmap_mode="r")
    train = [r for group in read(SRC / "train_cards.json") for r in group]
    excluded = {
        k: {r[k] for r in train} for k in ("family", "raster", "label", "entry")
    }
    by = defaultdict(lambda: defaultdict(dict))
    for card in read(SRC / "development_cards.json"):
        for r in card:
            assert all(r[k] not in vs for k, vs in excluded.items())
            pix = tiles[r["tile"]].tobytes()
            assert hashlib.sha256(pix).hexdigest() == r["raster"]
            by[r["family"]][r["entry"], r["char"]].setdefault(r["stage"], r)
    out = []
    for family in sorted(by):
        for _, stages in sorted(by[family].items()):
            if not {"oracle", "bronze"} <= stages.keys():
                continue
            a, b = (stages[k] for k in ("oracle", "bronze"))
            if a["raster"] == b["raster"]:
                continue
            out.append([a, b])
            break
    assert len(out) >= 40 and len({e[0]["char"] for e in out}) == len(out)
    if len(out) % 2:
        out = out[1:] + [out[0]]  # Odd final entry uses the first as its distractor.
    record(PUBLIC / "families.json", out)
    record(
        PUBLIC / "input.json",
        dict(
            families=sha(PUBLIC / "families.json"),
            n=len(out),
            sources={
                n: sha(SRC / n)
                for n in ("pixels.npy", "train_cards.json", "development_cards.json")
            },
            editor=sha(ROOT / "models/grounded_ink_editor/editor.safetensors"),
            wording="列出兩種字形",
        ),
    )


def tasks():
    spec = read(PUBLIC / "input.json")
    assert (
        sha(PUBLIC / "families.json") == spec["families"]
        and sha(SRC / "pixels.npy") == spec["sources"]["pixels.npy"]
    )
    fs = read(PUBLIC / "families.json")
    pixels = np.load(SRC / "pixels.npy", mmap_mode="r")
    face = Face(FACES[0])
    for i, rows in enumerate(fs):
        other = fs[(i ^ 1) % len(fs)]

        def parts(rs):
            a, b = rs
            c = a["char"]
            return (
                face("「" + c + "」的甲骨文寫作")
                + pixels[a["tile"]].tobytes()
                + face("，金文寫作")
                + pixels[b["tile"]].tobytes()
                + face("。")
            )

        c = rows[0]["char"]
        targets = [pixels[v["tile"]].tobytes() for v in rows]
        for order in (0, 1):
            page = (
                parts(rows) + parts(other) if not order else parts(other) + parts(rows)
            )
            qs = [
                face("請寫出「" + c + "」的" + stage + "字形。")
                for stage in ("甲骨文", "金文")
            ]
            yield (c, order, page, qs, targets)


@torch.inference_mode()
def score(arm):
    device = "cuda"
    net, writer = load(
        BASE,
        device,
        "original" if arm in ("original", "generic", "supplied") else "adapted",
    )
    bank = InkBank()
    face = Face(FACES[0])
    rows = list(tasks())
    pairs = [
        (
            bank.encode(page),
            bank.encode(face("請寫出字形。") if arm == "generic" else q),
        )
        for _, _, page, qs, _ in rows
        for q in qs
    ]
    if arm == "supplied":
        out = [a for _, _, _, _, tar in rows for a in tar]
    else:
        emb = embeddings(net, bank)
        out = []
        for i in range(0, len(pairs), 24):
            for cells in read_many(net, emb, pairs[i : i + 24]):
                out.append(b"".join(bank.cells[k] for k in cells))
    details = []
    per = defaultdict(list)
    single = [0, 0]
    stopped = 0
    for i, (char, order, page, qs, targets) in enumerate(rows):
        got = out[2 * i : 2 * i + 2]
        regions = [first_region(page, x) for x in got]
        for k in (0, 1):
            single[k] += got[k] == targets[k]
        for rev in (0, 1):
            y, end = writer.emit(page, regions[:: 1 - 2 * rev], face("列出兩種字形"))
            a, b = targets[:: 1 - 2 * rev]
            wanted = a + face("與") + b + face("。")
            ok = end and y == wanted
            stopped += end
            per[char].append(ok)
            details.append(
                dict(
                    char=char,
                    order=order,
                    reverse=rev,
                    regions=regions,
                    correct_sources=[a == b for a, b in zip(got, targets)],
                    stopped=end,
                    correct=ok,
                    image=y.hex(),
                )
            )
    dest = PUBLIC / (arm + ".jsonl.gz")
    with (
        dest.open("wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
    ):
        for d in details:
            gz.write(
                (
                    json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
    assert all(len(v) == 4 for v in per.values())
    record(
        PUBLIC / (arm + ".json"),
        dict(
            replies=sha(dest),
            single=single,
            n_family=len(per),
            n_requests=len(details),
            correct=sum(sum(v) for v in per.values()),
            complete=sum(all(v) for v in per.values()),
            stopped=int(stopped),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "action", choices=("prepare", "adapted", "original", "generic", "supplied")
    )
    arg = ap.parse_args()
    if arg.action == "prepare":
        prepare()
    else:
        with gpu_owner():
            torch.set_num_threads(2)
            score(arg.action)
