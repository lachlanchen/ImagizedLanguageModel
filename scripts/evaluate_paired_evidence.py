"""Evaluate complete free writing, evidence edits and question changes on fresh articles."""

from collections import defaultdict
import hashlib
import re
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_paired_evidence import WORK, PUB, MODEL, read, dump
from scripts.train_paired_evidence import (
    Pairs,
    Corpus,
    load_model,
    gpu_owner,
    byte_features,
)
from scripts.write_paired_evidence import reader, generate
from scripts.write_visual_cursor import load_cursor, generate as cursor_generate
from scripts.evaluate_visual_discourse import cohorts, packed, transcript, draw
from scripts.evaluate_visual_cursor import normalize, overlap
from scripts.measure_writer_energy import consumed, snapshot


def setup(data, old):
    held = read(WORK / "held.json")
    tiles = list(map(bytes, data.a["tiles"]))
    chars = list(map(str, data.a["chars"]))
    by = {c: i for i, c in enumerate(chars) if i >= 2}
    all_shapes = set(tiles)
    novel = {}
    for d in "0123456789":
        bits = np.unpackbits(np.frombuffer(tiles[by[d]], np.uint8)).reshape(32, 32)
        g = np.packbits(np.rot90(bits)).tobytes()
        if g in all_shapes:
            a = np.rot90(bits).copy()
            a[1, 1:5] = 1
            a[2, 1] = 1
            g = np.packbits(a).tobytes()
        if g in all_shapes:
            raise ValueError("Synthetic mark already present")
        novel[d] = len(tiles)
        tiles.append(g)
        all_shapes.add(g)
    rows = []
    for p in held:
        for j, r in enumerate(p["rows"]):
            rec = dict(
                kind=p["kind"],
                doc=p["doc"],
                idx=j,
                q=r["q"],
                s=p["s"],
                refs=r["refs"],
                question=r["question"],
                absent=not any(
                    normalize(ref) in normalize(p["source"]) for ref in r["refs"]
                ),
            )
            rows.append(dict(rec, test="human"))
            rows.append(dict(rec, test="erased", q=[by[" "]]))
            # Intervene on a single explicit numeral, keeping all question pixels fixed.
            m = re.fullmatch(r"([1-9][0-9]{2,3})(年?)", r["refs"][0])
            if not m:
                continue
            number = m[1]
            replacement = str(int(number) + 1)
            matches = list(
                re.finditer(r"(?<![0-9])" + number + r"(?![0-9])", p["source"])
            )
            if (
                len(matches) != 1
                or number in r["query"]
                or replacement in p["source"]
                or len(replacement) != len(number)
            ):
                continue
            l = matches[0].start()
            s = p["s"].copy()
            s[l : l + len(number)] = [by[c] for c in replacement]
            rows.append(
                dict(
                    rec,
                    test="changed",
                    s=s,
                    refs=[replacement + m[2]],
                    original=r["refs"][0],
                    offset=l,
                )
            )
            s = p["s"].copy()
            ids = [novel[c] for c in number]
            s[l : l + len(number)] = ids
            rows.append(
                dict(
                    rec,
                    test="novel",
                    s=s,
                    wanted=ids + ([by["年"]] if m[2] else []),
                    offset=l,
                )
            )
    return rows, np.frombuffer(b"".join(tiles), np.uint8).reshape(-1, 128), chars, by


def score(arm, net, base, features, n, rows, by, single=False, cap=96):
    all_out = []
    for k in range(0, len(rows), 24):
        group = rows[k : k + 24]
        q = [r["q"] for r in group]
        s = [r["s"] for r in group]
        if arm == "cursor":
            if single:
                xs = s
            else:
                xs = []
                for r in group:
                    en = r["kind"] in ("coqa", "squad")
                    literal = lambda t: [by[c] for c in t]
                    xs.append(
                        literal("Q: " if en else "问：")
                        + r["q"]
                        + literal(" Passage: " if en else " 原文：")
                        + r["s"]
                        + literal(" A: " if en else " 答：")
                    )
                xs = [x[:768] for x in xs]
            a, b = cursor_generate(base, net, features, xs, n, cap, False)
            out = list(zip(a, b))
        else:
            out = generate(
                base,
                net,
                features,
                n,
                q,
                s,
                s if single else [a + b for a, b in zip(s, q)],
                cap,
            )
        all_out += out
    return all_out


@torch.inference_mode()
def main(arms):
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    data = Pairs()
    old = Corpus()
    rows, tiles, chars, by = setup(data, old)
    refs, blocks = cohorts(old)
    general = []
    for task, g in blocks.items():
        order = sorted(range(len(g)), key=lambda i: len(g[i]["x"]))
        if len(order) % 2:
            order.pop()
        general.extend(
            dict(r, q=r["x"][-160:], s=r["x"], kind=task)
            for i, r in enumerate(g)
            if i in order
        )
    win = [
        dict(r, q=r["x"][-160:], s=r["x"], kind="winograd")
        for r in refs
        if r["split"] == "winograd"
    ]
    with gpu_owner():
        for arm in arms:
            path = PUB / (arm + "-scores.json")
            if path.exists():
                continue
            byte = arm == "byte-evidence"
            batch = [r for r in rows if not byte or r["test"] != "novel"]
            base = load_model("byte-pair" if byte else "ink-pair", "cuda")
            net = (
                load_cursor("independent", "cuda")
                if arm == "cursor"
                else reader(arm, "cuda")
            )
            x = (
                byte_features(chars).cuda()
                if byte
                else torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
            )
            torch.cuda.synchronize()
            start = monotonic()
            meter = snapshot()
            detail = []
            outputs = {}
            for i, (r, (out, end)) in enumerate(
                zip(batch, score(arm, net, base, x, old.n, batch, by), strict=True)
            ):
                text = "".join(chars[c] if c < len(chars) else "�" for c in out)
                correct = end and (
                    out == r["wanted"]
                    if r["test"] == "novel"
                    else normalize(text) in [normalize(v) for v in r["refs"]]
                )
                detail.append(
                    dict(
                        kind=r["kind"],
                        doc=r["doc"],
                        idx=r["idx"],
                        test=r["test"],
                        question=r["question"],
                        refs=r["refs"],
                        absent=r["absent"],
                        output=text,
                        stopped=end,
                        correct=bool(correct),
                        glyphs=out,
                        sha256=hashlib.sha256(tiles[out].tobytes()).hexdigest(),
                        glyph_f1=max(overlap(text, v) for v in r["refs"]),
                    )
                )
                outputs[r["doc"], r["idx"], r["test"]] = bool(correct)
            result = {}
            for k in sorted({r["kind"] for r in batch}):
                res = {}
                for test in sorted({r["test"] for r in batch}):
                    g = [r for r in detail if r["kind"] == k and r["test"] == test]
                    docs = {r["doc"] for r in g}
                    res[test] = dict(
                        n=len(g),
                        correct=sum(r["correct"] for r in g),
                        stopped=sum(r["stopped"] for r in g),
                        mean_glyph_f1=float(np.mean([r["glyph_f1"] for r in g]))
                        if g
                        else None,
                        absent=sum(r["absent"] for r in g),
                        absent_correct=sum(r["correct"] and r["absent"] for r in g),
                        both_questions=sum(
                            all(outputs.get((d, i, test), False) for i in (0, 1))
                            for d in docs
                        ),
                        docs=len(docs),
                        original_and_changed=sum(
                            r["correct"] and outputs[r["doc"], r["idx"], "human"]
                            for r in g
                        ),
                    )
                result[k] = res
            packed(PUB / (arm + "-qa.json.gz"), detail)
            general_out = []
            for r, (out, end) in zip(
                general, score(arm, net, base, x, old.n, general, by, True), strict=True
            ):
                general_out.append(
                    dict(
                        task=r["kind"],
                        id=r["id"],
                        output=transcript(chars, out),
                        stopped=end,
                        first_eight=out[:8] == r["y"][:8],
                    )
                )
            packed(PUB / (arm + "-open.json.gz"), general_out)
            result["open"] = {
                k: dict(
                    n=len(g),
                    first_eight=sum(r["first_eight"] for r in g),
                    stopped=sum(r["stopped"] for r in g),
                    first=g[0]["output"],
                )
                for k in blocks
                if (g := [r for r in general_out if r["task"] == k])
            }
            wd = defaultdict(list)
            for r, (out, end) in zip(
                win, score(arm, net, base, x, old.n, win, by, True, 40), strict=True
            ):
                wd[r["pair"]].append(end and out == r["y"][:-1])
            result["winograd"] = dict(
                n=len(win),
                correct=sum(sum(g) for g in wd.values()),
                pairs=len(wd),
                both=sum(all(g) for g in wd.values()),
            )
            torch.cuda.synchronize()
            result["evaluation"] = dict(
                seconds=monotonic() - start, energy=consumed(meter, snapshot())
            )
            dump(path, result)
            print(
                arm,
                [
                    (
                        k,
                        result[k]["human"]["correct"],
                        result[k]["human"]["both_questions"],
                    )
                    for k in ("coqa", "squad", "cmrc", "drcd")
                ],
                flush=True,
            )
            if arm == "evidence":
                for k in ("coqa", "cmrc", "drcd", "squad"):
                    options = [
                        d
                        for d in detail
                        if d["kind"] == k and d["test"] == "human" and d["idx"] == 0
                    ]
                    for tag, complete in (("both", True), ("miss", False)):
                        hit = next(
                            (
                                d["doc"]
                                for d in options
                                if all(outputs[d["doc"], j, "human"] for j in range(2))
                                == complete
                            ),
                            None,
                        )
                        if not hit:
                            continue
                        for r, d in zip(batch, detail, strict=True):
                            if r["test"] != "human" or r["doc"] != hit:
                                continue
                            dest = MODEL / "examples" / k / tag / str(r["idx"])
                            dest.mkdir(exist_ok=True, parents=True)
                            for typ, seq in (
                                ("query", r["q"]),
                                ("passage", r["s"]),
                                ("answer", d["glyphs"]),
                            ):
                                draw(dest, typ, tiles[seq])
                            dump(
                                dest / "readout.json",
                                dict(
                                    **d,
                                    query_cells=len(r["q"]),
                                    passage_cells=len(r["s"]),
                                ),
                            )
    return


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or ["cursor", "answer", "evidence", "byte-evidence"])
