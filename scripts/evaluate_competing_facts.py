"""Measure unrestricted answer images and pairwise preferences on new page families."""

from collections import defaultdict
from pathlib import Path
from fontTools.ttLib import TTFont
from PIL import ImageFont
from scripts.prepare_visual_sequence_corpus import Rasterizer
from hashlib import sha256
import re
import numpy as np
import torch
from scripts.prepare_competing_facts import (
    WORK,
    PUB,
    PAIRED,
    KINDS,
    ROOT,
    read,
    dump,
    sha,
)
from scripts.prepare_paired_evidence import Encoder
from scripts.train_competing_facts import trained
from scripts.evaluate_visual_cursor import normalize
from scripts.evaluate_visual_discourse import packed
from scripts.train_paired_evidence import load_model, byte_features, gpu_owner
from scripts.train_visual_cursor import pad
from scripts.write_paired_evidence import reader, generate
from scripts.write_visual_statements import editor, realize
from ilm.visual_lm.book_prefix import observations


class CorpusFace(Rasterizer):
    """Keep every reserved page, including IPA missing from the CJK font."""

    def __init__(self):
        super().__init__()
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        self.alternate = Rasterizer()
        self.alternate.font = ImageFont.truetype(str(path), 26)
        with TTFont(path) as face:
            self.alternate.coverage = set(face.getBestCmap())

    def tile(self, c):
        if ord(c) in self.coverage or c.isspace():
            return super().tile(c)
        return self.alternate.tile(c)


def dataset(held_path=None, expected_hash=None, statements=True):
    spec = read(PUB / "corpus.json")
    held_path = WORK / "held.json" if held_path is None else held_path
    if sha(held_path) != (expected_hash or spec["files"]["held.json"]):
        raise ValueError("Evaluation split differs")
    enc = Encoder()
    enc.raster = CorpusFace()
    with np.load(PAIRED / "corpus.npz") as z:
        enc.tiles = list(map(bytes, z["tiles"]))
        enc.chars = list(map(str, z["chars"]))
    enc.shapes = {v: i for i, v in enumerate(enc.tiles) if i >= 2}
    enc.cache = {}
    rows = []
    for page in read(held_path):
        src = page["source"]
        s = enc.ids(src)
        kind = page["kind"]
        for i, r in enumerate(page["rows"]):
            q = enc.ids(r["query"])
            base = dict(
                key=page["doc"] + "/" + str(i),
                kind=kind,
                q=q,
                s=s,
                refs=r["refs"],
                question=r["query"],
                source=src,
                competitor=page["rows"][1 - i]["refs"],
                doc=page["doc"],
            )
            rows.append(dict(base, test="human"))
            rows.append(dict(base, q=enc.ids(" "), test="erased"))
            m = re.fullmatch(r"([1-9][0-9]{2,3})(年?)", r["refs"][0])
            if m:
                old = m[1]
                new = str(int(old) + 1)
                matches = list(re.finditer(r"(?<![0-9])" + old + r"(?![0-9])", src))
                if (
                    len(matches) == 1
                    and old not in r["query"]
                    and len(new) == len(old)
                    and new not in src
                ):
                    page2 = src[: matches[0].start()] + new + src[matches[0].end() :]
                    rows.append(
                        dict(
                            base,
                            test="edit",
                            s=enc.ids(page2),
                            source=page2,
                            refs=[new + m[2]],
                        )
                    )
    if not statements:
        return rows, enc
    for split, name in (("fresh", "fresh96"), ("held", "qa2d")):
        for r in read(ROOT / "artifacts/visual_statements_v1" / (split + ".json")):
            if not r.get("source"):
                continue
            refs = [r["fact"]] if "fact" in r else r["refs"]
            rows.append(
                dict(
                    key=r["id"],
                    kind="squad",
                    doc=r["id"],
                    test=name,
                    refs=refs,
                    s=enc.ids(r["source"]),
                    q=enc.ids(r["question"]),
                    question=r["question"],
                    source=r["source"],
                    reference=r.get("reference"),
                )
            )
    return rows, enc


def occurrences(s, t):
    if not 0 < len(t) <= 48:
        return []
    return [
        (i, len(t) - 1) for i in range(len(s) - len(t) + 1) if s[i : i + len(t)] == t
    ]


def positions(s, refs, enc):
    out = set()
    for ans in refs:
        try:
            t = enc.ids(ans)
        except ValueError:
            continue
        out.update(occurrences(s, t))
    return sorted(out)


@torch.inference_mode()
def assess(arm, rows, enc, *, supplied_reader=None, write_statements=True):
    # Precompute exact physical surface classes before materializing the glyph bank.
    human = [r for r in rows if r["test"] == "human"]
    for r in human:
        r["pos"] = positions(r["s"], r["refs"], enc)
        r["neg"] = positions(r["s"], r["competitor"], enc)
    tiles = np.array([np.frombuffer(t, np.uint8) for t in enc.tiles])
    byte = arm in ("bytes", "parent-bytes")
    optical = not byte
    features = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    x = byte_features(enc.chars).cuda() if byte else features
    n = len(np.load(ROOT / "models/visual_discourse/glyphs.npy"))
    base = load_model("byte-pair" if byte else "ink-pair", "cuda")
    net = (
        supplied_reader
        if supplied_reader is not None
        else (
            reader("byte-evidence" if byte else "evidence", "cuda")
            if arm.startswith("parent")
            else trained(arm, "cuda")
        )
    )
    sentence = editor("cuda") if optical and write_statements else None
    space = enc.ids(" ")[0]
    bank = net.ink(x)
    detail = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        q = [r["q"] for r in batch]
        s = [r["s"] for r in batch]
        facts = generate(base, net, x, n, q, s, [v + u for u, v in zip(q, s)], 96)
        write = [
            i
            for i, r in enumerate(batch)
            if optical
            and write_statements
            and (
                r["test"] in ("fresh96", "qa2d")
                or (r["test"] == "human" and r["kind"] in ("coqa", "squad"))
            )
        ]
        statements = (
            realize(
                base,
                sentence,
                features,
                n,
                [q[i] for i in write],
                [facts[i][0] for i in write],
                space,
            )
            if write
            else []
        )
        actual = dict(zip(write, statements, strict=True))
        spans = net.read(
            bank, *[torch.as_tensor(pad(a), device="cuda").long() for a in (q, s)]
        )[0]
        for i, (r, (fact, stop)) in enumerate(zip(batch, facts, strict=True)):

            def text(ids):
                return "".join(enc.chars[j] for j in ids)

            def physical(ids):
                return sha256(tiles[np.array(ids, dtype=int)].tobytes()).hexdigest()

            txt = text(fact)
            accepted = {normalize(a) for a in r["refs"]}
            row = dict(
                key=r["key"],
                doc=r["doc"],
                kind=r["kind"],
                test=r["test"],
                fact=txt,
                stopped=bool(stop),
                correct=bool(stop and normalize(txt) in accepted),
                physical=physical(fact),
                visible=any(a in normalize(r["source"]) for a in accepted if a),
            )
            if r["test"] == "human":
                row["rival_only"] = bool(
                    not row["correct"]
                    and stop
                    and normalize(txt) in set(map(normalize, r["competitor"]))
                )
                if r["pos"] and r["neg"] and set(r["pos"]).isdisjoint(r["neg"]):

                    def logmass(at):
                        a, b = zip(*at)
                        return int(
                            round(
                                float(spans[i, list(a), list(b)].logsumexp(0)) * 10000
                            )
                        )

                    row["target_log"] = logmass(r["pos"])
                    row["rival_log"] = logmass(r["neg"])
            if i in actual:
                out, end = actual[i]
                t = text(out)
                row.update(
                    answer=t,
                    answer_ink=physical(out),
                    answer_stop=bool(end),
                    contains=bool(
                        end and any(a in normalize(t) for a in accepted if a)
                    ),
                )
                if r.get("reference"):
                    row["exact_statement"] = bool(
                        end and normalize(t) == normalize(r["reference"])
                    )
            detail.append(row)
    return detail


def stats(rows):
    res = {}
    for k in (*KINDS, "all"):
        rs = [r for r in rows if r["kind"] == k or k == "all"]
        human = [r for r in rs if r["test"] == "human"]
        pairs = defaultdict(list)
        for r in human:
            pairs[r["doc"]].append(r)
        eligible = [g for g in pairs.values() if all("target_log" in r for r in g)]
        assert all(len(g) == 2 for g in pairs.values())
        plain = {r["key"]: r["correct"] for r in human}
        edits = [r for r in rs if r["test"] == "edit"]
        res[k] = dict(
            questions=len(human),
            correct=sum(r["correct"] for r in human),
            both=sum(all(r["correct"] for r in g) for g in pairs.values()),
            pages=len(pairs),
            rival=sum(r["rival_only"] for r in human),
            visible=sum(r["visible"] for r in human),
            visible_correct=sum(r["visible"] and r["correct"] for r in human),
            erased=sum(r["correct"] for r in rs if r["test"] == "erased"),
            edit_n=len(edits),
            edit_both=sum(r["correct"] and plain[r["key"]] for r in edits),
            eligible=len(eligible),
            both_prefer=sum(
                all(r["target_log"] > r["rival_log"] for r in g) for g in eligible
            ),
            bias_recoverable=sum(
                sum(r["target_log"] - r["rival_log"] for r in g) > 0 for g in eligible
            ),
        )
    for t in ("qa2d", "fresh96"):
        rs = [r for r in rows if r["test"] == t]
        res[t] = dict(n=len(rs), fact=sum(r["correct"] for r in rs))
        for k in ("contains", "exact_statement", "answer_stop"):
            if k in rs[0]:
                res[t][k] = sum(r[k] for r in rs)
    return res


def main():
    rows, enc = dataset()
    with gpu_owner():
        for arm in ("parent", "likelihood", "contrast", "parent-bytes", "bytes"):
            out = PUB / (arm + "-outputs.json.gz")
            receipt = PUB / (arm + "-scores.json")
            if receipt.exists():
                if sha(out) != read(receipt)["sha256"]:
                    raise ValueError("Published outputs changed")
                continue
            detail = assess(arm, rows, enc)
            packed(out, detail)
            s = stats(detail)
            dump(receipt, dict(sha256=sha(out), metrics=s))
            print(arm, s, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
