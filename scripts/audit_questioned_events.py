"""Decode saved output strips and recount complete-page scores, including controls."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from scripts.prepare_questioned_events import PUB, read, dump
from scripts.train_typeface_writer import FACES, Face


def audit(folder):
    path = folder / "decisions.json"
    if path.exists():
        trace = read(path)
    else:
        with gzip.open(path.with_suffix(".json.gz"), "rt") as f:
            trace = json.load(f)
    result = read(folder / "result.json")
    pages = read(PUB / "held.json")
    face = Face(FACES[0])
    abcd = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    with np.load(folder / "ink.npz") as archive:
        bits = archive["pixels"]
        offset = archive["offsets"]
        strings = [
            "".join(abcd.get(cell.tobytes(), "�") for cell in bits[a:b])
            for a, b in zip(offset, offset[1:])
        ]
    grouped = defaultdict(list)
    for ix, mode, j, qi, arm, v, o, image, stop, conditional, gold, gold_stop in trace[
        "records"
    ]:
        page = pages[ix]
        wanted_v, wanted_o, _, terms = page["events"][j]
        v_ok = v == wanted_v
        o_ok = o == wanted_o
        written = bool(stop and strings[image] in terms and o_ok)
        metrics = dict(
            verb=v_ok,
            object=o_ok,
            binding=v_ok and o_ok,
            written=written,
            bound_written=v_ok and written,
            given_verb_object=conditional == wanted_o,
            given_object_written=bool(gold_stop and strings[gold] in terms),
        )
        grouped[(ix, mode, arm)].append((j, qi, metrics))
    masks = defaultdict(dict)
    for (ix, mode, arm), answers in grouped.items():
        page = pages[ix]
        count = len(page["events"])
        if (
            len(answers) != 2 * count
            or len({(j, q) for j, q, _ in answers}) != 2 * count
        ):
            raise ValueError("Missing or repeated question")
        flags = {k: all(r[k] for _, _, r in answers) for k in answers[0][2]}
        cat = [page["group"]]
        if page["group"] != "literature":
            cat += [
                "kyoto",
                "kyoto_adjacent"
                if all(o == v + 1 for v, o, _, _ in page["events"])
                else "kyoto_nonadjacent",
            ]
        for key in cat:
            masks[key + "_" + mode, arm][ix] = flags
    compared = {}
    for (name, arm), flags in masks.items():
        totals = {
            k: sum(f[k] for f in flags.values()) for k in next(iter(flags.values()))
        }
        assert dict(n=len(flags), **totals) == result["groups"][name][arm]
        if arm in ("aligned", "word_aligned"):
            compared[name + "_" + arm] = {}
            controls = (
                ("frozen", "exchanged", "adjacent")
                if arm == "aligned"
                else ("aligned", "word_exchanged")
            )
            for control in controls:
                other = masks[name, control]
                candidate = {i for i, f in flags.items() if f["bound_written"]}
                baseline = {i for i, f in other.items() if f["bound_written"]}
                compared[name + "_" + arm][control] = dict(
                    only_candidate=len(candidate - baseline),
                    only_control=len(baseline - candidate),
                    both=len(baseline & candidate),
                )
    report = dict(
        raw_decisions=len(trace["records"]),
        unique_strips=len(strings),
        groups_verified=len(masks),
        paired=compared,
    )
    dump(folder / "audit.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    audit(p.parse_args().folder)
