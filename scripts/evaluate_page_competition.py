"""Measure complete autonomous answers, full-page score gaps and transfer beyond lists."""

from collections import defaultdict
from hashlib import sha256
from statistics import median
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import intervals, best_path
from ilm.visual_lm.page_competition import best_reference
from ilm.visual_lm.portable_writer import read_text, DEFAULT_FONT
from scripts.audit_visible_answer_sets import placement
from scripts.evaluate_complete_fact_sets import paired_set, measure, history
from scripts.evaluate_multiple_ink_runs import corpus, evaluate_cases
from scripts.evaluate_visual_discourse import packed
from scripts.prepare_complete_fact_sets import MODEL as PARENT
from scripts.prepare_page_competition import WORK, MODEL, PUB, read, dump, sha
from scripts.prepare_multiple_ink_runs import normalize, multi
from scripts.prepare_recurrent_evidence import WORK as OLD, PUBLIC as OLD_PUB
from scripts.write_multiple_ink_runs import load_reader, forward, sentences, joined
from scripts.train_paired_evidence import gpu_owner
from scripts.write_visual_discourse import Writer
from scripts.write_visual_statements import editor

ARMS = ("parent", "control", "competition")


def release(arm):
    return PARENT / "contrast" if arm == "parent" else MODEL / arm


def setup():
    spec = read(PUB / "fresh.json")
    if (
        sha(WORK / "confirm.json") != spec["sha256"]
        or sha(WORK / "confirm-tiles.npy") != spec["tiles"]
    ):
        raise ValueError("Fresh pages changed")
    _, tiles = corpus()
    return read(WORK / "confirm.json"), np.concatenate(
        [tiles, np.load(WORK / "confirm-tiles.npy")]
    )


def physical_bank(net, tiles):
    return net.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda())


@torch.inference_mode()
def held_gap(arm, rows, tiles):
    net = load_reader("cuda", True, release(arm))
    bank = physical_bank(net, tiles)
    out = []
    for i in range(0, len(rows), 12):
        batch = rows[i : i + 12]
        roles, counts = forward(
            net, bank, [r["qi"] for r in batch], [r["pi"] for r in batch], True
        )
        for r, u, c in zip(batch, roles, counts, strict=True):
            opts = []
            cache = {}
            for ref in r["refs"]:
                words = list({normalize(w): w for w in reversed(ref)}.values())
                if placement(r["p"], words, cache) == "placed":
                    opts.append([[(i, i + len(w)) for i in cache[w]] for w in words])
            if opts:
                role = np.rint(u[: len(r["pi"])].astype(np.float64) * 1000).astype(
                    np.int64
                )
                count = np.rint(c.astype(np.float64) * 1000).astype(np.int64)
                _, gold = best_reference(role, count, opts)
                _, free = best_path(role, count)
                if free < gold:
                    raise ValueError("Contradictory full-page scores")
                out.append(
                    dict(
                        id=r["id"],
                        group=r["lang"] + ("-multi" if multi(r) else "-single"),
                        gap=(free - gold) / 1000,
                    )
                )
    dump(PUB / (arm + "-gaps.json"), out)


@torch.inference_mode()
def prose(arm):
    spec = read(OLD_PUB / "corpus.json")["files"]
    if any(sha(OLD / f) != spec[f] for f in ("held.json", "train.npz")):
        raise ValueError("Source-absent pages changed")
    rows = [r for r in read(OLD / "held.json") if r["subset"] == "absent"]
    with np.load(OLD / "train.npz") as f:
        tiles = f["tiles"]
    net = load_reader("cuda", True, release(arm))
    bank = physical_bank(net, tiles)
    writer, head = Writer(device="cuda"), editor("cuda")
    sep = read_text(" and ", DEFAULT_FONT)
    bar = read_text(" | ", DEFAULT_FONT)
    alpha = {bytes(b): c for b, c in zip(writer.palette, writer.transcript)}
    for r in rows:
        for ix, c in zip(r["q"] + r["s"], r["question"] + r["source"], strict=True):
            alpha.setdefault(bytes(tiles[ix]), c)
    out = []
    for i in range(0, len(rows), 8):
        batch = rows[i : i + 8]
        role, count = forward(
            net, bank, [r["q"] for r in batch], [r["s"] for r in batch], True
        )
        eligible = []
        for r, u, c in zip(batch, role, count, strict=True):
            spans = intervals(u[: len(r["s"])], c)
            parts = [tiles[r["s"][a:b]] for a, b in spans]
            actual = joined(parts, bar)
            text = " | ".join(r["source"][a:b] for a, b in spans)
            rec = dict(
                lang=r["lang"],
                id=r["id"],
                list=text,
                list_exact=any(normalize(text) == normalize(v) for v in r["refs"]),
                sha256=sha256(actual.tobytes()).hexdigest(),
                sentence_exact=False,
            )
            out.append(rec)
            if (
                r["lang"] == "en"
                and len(r["q"])
                + 2
                + sum(map(len, parts))
                + len(sep) * max(0, len(parts) - 1)
                <= 600
            ):
                eligible.append((r, parts, rec))
        if eligible:
            ans = sentences(
                writer,
                head,
                [tiles[r["q"]] for r, _, _ in eligible],
                [p for _, p, _ in eligible],
                sep,
            )
            for (pix, stop), (r, _, rec) in zip(ans, eligible, strict=True):
                txt = "".join(alpha[bytes(b)] for b in pix)
                rec.update(
                    sentence=txt,
                    stopped=stop,
                    sentence_exact=stop
                    and any(normalize(txt) == normalize(v) for v in r["refs"]),
                )
    packed(PUB / (arm + "-prose.json.gz"), out)
    return {
        l: dict(
            n=len(g),
            list_exact=sum(v["list_exact"] for v in g),
            sentence_exact=sum(v["sentence_exact"] for v in g),
            attempted_sentence=sum("sentence" in v for v in g),
            stopped=sum(v.get("stopped", False) for v in g),
        )
        for l in ("zh", "en")
        for g in [[v for v in out if v["lang"] == l]]
    }


def main():
    rows, tiles = setup()
    paired, pt = paired_set(tiles)
    summary = {}
    for arm in ARMS:
        evaluate_cases(rows, tiles, PUB, arm + "-", True, release(arm))
        with gpu_owner():
            held_gap(arm, rows, tiles)
            measure(arm, rows, tiles, paired, pt, release=release, output=PUB)
            summary[arm] = prose(arm)
    with gpu_owner():
        history(
            ARMS,
            release=release,
            output=PUB,
            example_root=MODEL,
            example_arm="competition",
        )
    dump(PUB / "prose-scores.json", summary)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
