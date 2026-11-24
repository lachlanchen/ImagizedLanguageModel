"""Recount fixed-reference coverage and replay the article's physical answer strips on CPU."""

from hashlib import sha256
from collections import Counter
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.prepare_complete_fact_sets import PUB, MODEL
from scripts.prepare_multiple_ink_runs import (
    WORK as BEFORE,
    read,
    dump,
    multi,
    normalize,
)
from scripts.evaluate_complete_fact_sets import ARMS, setup, run_dir
from scripts.evaluate_multiple_ink_runs import grade
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.write_multiple_ink_runs import infer, load_reader, write, sentences
from scripts.write_visual_statements import editor
from scripts.write_visual_discourse import Writer


def main():
    torch.set_num_threads(3)
    rows, tiles = setup()
    lookup = {r["id"]: r for r in rows}
    stats = {}
    cases = {"poem": "186784", "music": "j5kbhyyg66xar1o5q4dr"}
    older = MODEL.parent / "multiple_ink_runs/examples/xixia"
    x = next(r for r in read(BEFORE / "confirm.json") if r["id"] == "186474")
    for arm in ARMS:
        net = load_reader("cpu", True, run_dir(arm))
        all_rows = unpack(PUB / (arm + "-answers.json.gz"))
        records = {
            r["id"]: r
            for r in all_rows
            if r["condition"] == "original" and r["mode"] == "count"
        }
        swapped = {
            r["id"]: r
            for r in all_rows
            if r["condition"] == "exchanged" and r["mode"] == "count"
        }
        summary = {}
        for l, plural in (("en", True), ("zh", True), ("zh", False)):
            fixed = []
            hist = Counter()
            both = 0
            dups = 0
            any_correct = 0
            for r in rows:
                if r["lang"] != l or multi(r) != plural:
                    continue
                v = records[r["id"]]
                g = grade(v["parts"], r["refs"][:1])
                fixed.append(g)
                hist[v["emitted"]] += 1
                any_correct += g["matched"] > 0
                dups += v["emitted"] - len(set(map(normalize, v["parts"])))
                both += v["exact"] and swapped[r["id"]]["exact"]
            summary[l + ("-multi" if plural else "-single")] = dict(
                n=len(fixed),
                parts_right=sum(g["matched"] for g in fixed),
                parts_written=sum(g["emitted"] for g in fixed),
                parts_expected=sum(g["expected"] for g in fixed),
                any_first_ref_part=int(any_correct),
                original_and_exchanged_exact=int(both),
                duplicate_outputs=dups,
                counts=dict(sorted(hist.items())),
            )
        stats[arm] = summary
        for name in ("music", "poem", "xixia"):
            dest = MODEL / "examples" / name
            dest.mkdir(parents=True, exist_ok=True)
            if name == "xixia":
                r = x
                q, p = (
                    read_image(older / "question.png", len(r["q"])),
                    read_image(older / "page.png", len(r["p"])),
                )
            else:
                r = lookup[cases[name]]
                q, p = tiles[r["qi"]], tiles[r["pi"]]
            if arm == ARMS[0]:
                draw(dest, "question", q)
                draw(dest, "page", p)
            q, p = (
                read_image(dest / "question.png", len(q)),
                read_image(dest / "page.png", len(p)),
            )
            bounds = infer(net, q, p, survival=True)
            parts = [p[a:b] for a, b in bounds]
            txt = [r["p"][a:b] for a, b in bounds]
            if name != "xixia":
                expected = records[r["id"]]
                assert expected["spans"] == bounds and expected["ink"] == [
                    sha256(v.tobytes()).hexdigest() for v in parts
                ]
            n = write(parts, dest / arm)
            receipt = dict(
                id=r["id"],
                question_cells=len(q),
                page_cells=len(p),
                answer_cells=n,
                spans=bounds,
                parts=txt,
                all_correct=grade(txt, r["refs"])["exact"],
                sha256=sha256((dest / arm / "answer.bin").read_bytes()).hexdigest(),
            )
            if name == "music" and arm == "contrast":
                pixels, stopped = sentences(
                    Writer(),
                    editor("cpu"),
                    [q],
                    [parts],
                    read_text(" and ", DEFAULT_FONT),
                )[0]
                expected = next(
                    v
                    for v in unpack(PUB / (arm + "-sentences.json.gz"))
                    if v["id"] == r["id"]
                )
                assert (
                    sha256(pixels.tobytes()).hexdigest() == expected["sha256"]
                    and stopped == expected["stop"]
                )
                draw(dest / arm, "sentence", pixels)
                receipt.update(
                    sentence_cells=len(pixels),
                    sentence=expected["text"],
                    stopped=stopped,
                )
            dump(dest / arm / "result.json", receipt)
    dump(PUB / "fixed-reference.json", stats)


if __name__ == "__main__":
    main()
