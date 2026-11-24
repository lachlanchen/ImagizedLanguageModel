"""Measure complete writing, reference cost, source changes and the established regressions."""

from collections import defaultdict
from hashlib import sha256
import re
import numpy as np
from rapidfuzz.distance import LCSseq
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_recurrent_evidence import WORK, PUBLIC, MODEL, read, dump
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_recurrent_evidence import Data
from scripts.train_paired_evidence import Corpus, gpu_owner, byte_features, load_model
from scripts.write_recurrent_evidence import load, generate
from scripts.write_paired_evidence import reader, generate as old_generate
from scripts.evaluate_paired_evidence import setup
from scripts.evaluate_visual_discourse import cohorts, packed, draw
from scripts.train_visual_cursor import base_batch, pad
from scripts.evaluate_visual_cursor import normalize

PUBLIC_PROMPTS = [
    (
        "museum-zh",
        "第一批和第二批各展出了多少件青铜器？请用一句话回答。",
        "馆方分两批展出新藏品。第一批展出了十七件青铜器。第二批展出了九件青铜器。",
        "第一批展出了十七件，第二批展出了九件。",
    ),
    (
        "museum-en",
        "How many bronzes were in each exhibition? Answer in one sentence.",
        "The museum held two exhibitions. The first exhibition contained seventeen bronzes. The second exhibition contained nine bronzes.",
        "The first exhibition had seventeen bronzes and the second had nine.",
    ),
    (
        "workshop-zh",
        "请说明两次开课的时间，先写晚班，再写早班。",
        "木刻课分两班。早班于八点开始。晚班于六点开始。",
        "晚班于六点开始，早班于八点开始。",
    ),
    (
        "workshop-en",
        "Give the evening class time, then the morning class time.",
        "The morning printmaking class starts at eight. The evening class starts at six.",
        "The evening class starts at six and the morning class starts at eight.",
    ),
]


def overlap(text, ref):
    a, b = normalize(text), normalize(ref)
    return 2 * LCSseq.similarity(a, b) / max(1, len(a) + len(b))


def lexical(r):
    choices = re.findall(r"[^。！？.!?]+[。！？.!?]?", r["source"]) or [r["source"]]
    q = set(
        normalize(r["question"])[i : i + 2]
        for i in range(len(normalize(r["question"])) - 1)
    )

    def rank(s):
        txt = normalize(s)
        keys = {txt[i : i + 2] for i in range(len(txt) - 1)}
        return len(q & keys) / (1 + len(keys)) ** 0.5

    return max(choices, key=rank).strip()


def scored(r, out, end):
    refs = r["refs"]
    text = normalize(out)
    grams = [text[i : i + 3] for i in range(max(0, len(text) - 2))]
    return dict(
        id=r["id"],
        lang=r["lang"],
        subset=r["subset"],
        output=out,
        stopped=end,
        exact=bool(end and any(text == normalize(a) for a in refs)),
        lcs=max(overlap(out, a) for a in refs),
        single_source=bool(text and text in normalize(r["source"])),
        repetition=1 - len(set(grams)) / len(grams) if grams else 0.0,
    )


def summary(rows):
    return dict(
        n=len(rows),
        exact=sum(r["exact"] for r in rows),
        stopped=sum(r["stopped"] for r in rows),
        contiguous=sum(r["single_source"] for r in rows),
        mean_lcs=float(np.mean([r["lcs"] for r in rows])),
        repeated_trigrams=float(np.mean([r["repetition"] for r in rows])),
    )


def changes(rows, by):
    edited = []
    for r in rows:
        for number in dict.fromkeys(
            re.findall(r"(?<![0-9])[1-9][0-9]{1,3}(?![0-9])", r["refs"][0])
        ):
            newer = str(int(number) + 1)
            matches = list(
                re.finditer(r"(?<![0-9])" + number + r"(?![0-9])", r["source"])
            )
            token = r"(?<![0-9])" + number + r"(?![0-9])"
            if len(matches) != 1 or not all(re.search(token, a) for a in r["refs"]):
                continue
            if (
                number in r["question"]
                or newer in r["source"]
                or len(number) != len(newer)
            ):
                continue
            s = r["s"].copy()
            idx = matches[0].start()
            s[idx : idx + len(number)] = [by[c] for c in newer]
            edited.append(dict(r, s=s, old=number, new=newer))
            break
    return edited


def inputs(data, old):
    qa, base_tiles, chars, by = setup(data, old)
    tiles = list(map(bytes, base_tiles))
    char = list(chars) + ["�"] * (len(tiles) - len(chars))
    key = {v: i for i, v in enumerate(tiles) if i >= 2}
    ras = Rasterizer()

    def ids(text):
        res = []
        for c in text:
            g = ras.tile(c)
            if g not in key:
                key[g] = len(tiles)
                tiles.append(g)
                char.append(c)
            res.append(key[g])
        return res

    held = read(WORK / "held.json")
    targets = []
    for r in held:
        try:
            y = ids(r["refs"][0][:96])
        except ValueError:
            y = []
        targets.append(y + ([1] if y and len(r["refs"][0]) < 96 else []))
    original = [
        dict(id=k, question=q, source=s, q=ids(q), s=ids(s), y=ids(y))
        for k, q, s, y in PUBLIC_PROMPTS
    ]
    return (
        qa,
        held,
        targets,
        original,
        np.asarray([np.frombuffer(g, np.uint8) for g in tiles]),
        char,
        by,
    )


@torch.inference_mode()
def evaluate(arm, old, data, prepared):
    qa, held, targets, original, tiles, chars, by = prepared
    baseline = arm == "previous"
    byte = arm == "byte"
    base = load_model("byte-pair" if byte else "ink-pair", "cuda")
    net = reader("evidence", "cuda") if baseline else load(arm, "cuda")
    x = (
        byte_features(chars).cuda()
        if byte
        else torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    )
    run = old_generate if baseline else generate

    def process(rows, cap, single=False):
        outs = []
        for j in range(0, len(rows), 14):
            g = rows[j : j + 14]
            qs = [r["q"] for r in g]
            ss = [r["s"] for r in g]
            ctx = ss if single else [a + b for a, b in zip(ss, qs, strict=True)]
            outs += run(base, net, x, old.n, qs, ss, ctx, cap)
        return outs

    def text(seq):
        return "".join(chars[i] for i in seq)

    out = process(held, 160)
    raw = [scored(r, text(ids), end) for r, (ids, end) in zip(held, out, strict=True)]
    results = {
        lang: {
            kind: summary([r for r in raw if r["lang"] == lang and r["subset"] == kind])
            for kind in ("absent", "additional")
        }
        for lang in ("en", "zh")
    }
    edits = changes(held, by)
    hits = 0
    edited_details = []
    for r, (ids, end) in zip(edits, process(edits, 160), strict=True):
        o = next(t for t in raw if t["id"] == r["id"])
        after = text(ids)
        word = lambda v, t: bool(re.search(r"(?<![0-9])" + v + r"(?![0-9])", t))
        ok = (
            o["stopped"]
            and end
            and word(r["old"], o["output"])
            and word(r["new"], after)
            and not word(r["old"], after)
        )
        hits += ok
        edited_details.append(dict(id=r["id"], changed=after, stopped=end, tracks=ok))
    results["edited_fact"] = dict(n=len(edits), tracks=hits)
    if not baseline:
        bank = base.encode(x)
        cost = []
        for idx in range(0, len(held), 16):
            ids = [i for i in range(idx, min(idx + 16, len(held))) if targets[i]]
            if not ids:
                continue
            for swapped in (False, True):
                qs = [held[i]["q"] for i in ids]
                ss = [
                    held[(i // 128) * 128 + (i + 1) % 128 if swapped else i]["s"]
                    for i in ids
                ]
                ys = [targets[i] for i in ids]
                bi, bp = [
                    torch.as_tensor(a, device="cuda")
                    for a in base_batch([s + q for q, s in zip(qs, ss)], ys)
                ]
                lp = base.log_prob(bank, bi, old.n, bp)
                val, _ = net.observed(
                    x,
                    *[torch.as_tensor(pad(v), device="cuda") for v in (qs, ss, ys)],
                    lp,
                    old.n,
                )
                cost.extend(
                    dict(
                        id=held[i]["id"],
                        lang=held[i]["lang"],
                        swapped=swapped,
                        n=len(targets[i]),
                        logp=z,
                    )
                    for i, z in zip(ids, val.tolist())
                )
        results["conditioned_bits"] = {
            l: {
                str(sw): -sum(
                    r["logp"] for r in cost if r["swapped"] == sw and r["lang"] == l
                )
                / sum(r["n"] for r in cost if r["swapped"] == sw and r["lang"] == l)
                / np.log(2)
                for sw in (False, True)
            }
            for l in ("en", "zh")
        }
        packed(PUBLIC / (arm + "-conditioned.json.gz"), cost)
    if not baseline:
        keys = defaultdict(list)
        qa = [r for r in qa if not byte or r["test"] != "novel"]
        truth = {}
        for r, (ids, end) in zip(qa, process(qa, 96), strict=True):
            ok = end and (
                ids == r["wanted"]
                if r["test"] == "novel"
                else normalize(text(ids)) in list(map(normalize, r["refs"]))
            )
            keys[r["test"], r["doc"]].append(ok)
            truth[r["doc"], r["idx"], r["test"]] = ok
        results["paired"] = dict(
            exact=sum(v for (d, i, k), v in truth.items() if k == "human"),
            both=sum(all(v) for (k, d), v in keys.items() if k == "human"),
            erased=sum(all(v) for (k, d), v in keys.items() if k == "erased"),
            changed=sum(
                v and truth[d, i, "human"]
                for (d, i, k), v in truth.items()
                if k == "changed"
            ),
            novel=None
            if byte
            else sum(
                v and truth[d, i, "human"]
                for (d, i, k), v in truth.items()
                if k == "novel"
            ),
        )
        refs, blocks = cohorts(old)
        win = [r for r in refs if r["split"] == "winograd"]
        w = defaultdict(list)

        def single(rs):
            return [dict(r, s=r["x"], q=r["x"][-160:]) for r in rs]

        for r, (ids, end) in zip(win, process(single(win), 40, True), strict=True):
            w[r["pair"]].append(end and ids == r["y"][:-1])
        results["winograd"] = dict(pairs=len(w), both=sum(all(g) for g in w.values()))
        op = []
        for task, rs in blocks.items():
            for r, (ids, end) in zip(rs, process(single(rs), 96, True), strict=True):
                op.append(
                    dict(
                        task=task,
                        id=r["id"],
                        output=text(ids),
                        stopped=end,
                        eight=ids[:8] == r["y"][:8],
                    )
                )
        results["open"] = {
            t: dict(
                n=len(g),
                eight=sum(r["eight"] for r in g),
                stopped=sum(r["stopped"] for r in g),
            )
            for t in blocks
            if (g := [r for r in op if r["task"] == t])
        }
        packed(PUBLIC / (arm + "-open.json.gz"), op)
    for r, (ids, end) in zip(original, process(original, 96), strict=True):
        dest = MODEL / "examples" / arm / r["id"]
        dest.mkdir(parents=True, exist_ok=True)
        for name, seq in (
            ("question", r["q"]),
            ("passage", r["s"]),
            ("answer", ids),
            ("reference", r["y"]),
        ):
            draw(dest, name, tiles[np.asarray(seq, dtype=int)])
        dump(
            dest / "readout.json",
            dict(
                question=r["question"],
                passage=r["source"],
                transcript=text(ids),
                stopped=end,
                question_cells=len(r["q"]),
                passage_cells=len(r["s"]),
                written=len(ids),
                sha256=sha256(tiles[np.asarray(ids, dtype=int)].tobytes()).hexdigest(),
            ),
        )
    # DuReader is a local research corpus: publish statistics and output hashes only.
    packed(
        WORK / (arm + "-private-answers.json.gz"),
        dict(answers=raw, edits=edited_details),
    )
    packed(
        PUBLIC / (arm + "-per-item.json.gz"),
        [
            dict(
                {k: v for k, v in r.items() if k != "output"},
                sha256=sha256(r["output"].encode()).hexdigest(),
            )
            for r in raw
        ],
    )
    dump(PUBLIC / (arm + "-scores.json"), results)
    print(arm, results, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    data = Data()
    old = Corpus()
    prepared = inputs(data, old)
    held = prepared[1]
    sents = [scored(r, lexical(r), True) for r in held]
    dump(
        PUBLIC / "sentence-control.json",
        {
            lang: {
                k: summary([r for r in sents if r["lang"] == lang and r["subset"] == k])
                for k in ("absent", "additional")
            }
            for lang in ("en", "zh")
        },
    )
    with gpu_owner():
        for arm in ("previous", "fixed", "dynamic", "byte"):
            if not (PUBLIC / (arm + "-scores.json")).exists():
                evaluate(arm, old, data, prepared)
