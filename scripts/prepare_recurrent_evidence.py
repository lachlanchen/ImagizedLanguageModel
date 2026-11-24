"""Prepare human explanatory answers and reserve complete page families before fitting."""

from collections import Counter
from hashlib import sha256
import json
import re
from zipfile import ZipFile
import numpy as np
from scripts.prepare_paired_evidence import (
    ROOT,
    WORK as PREVIOUS,
    clean,
    dump,
    read,
    sha,
    Encoder,
)
from scripts.evaluate_visual_cursor import normalize

WORK = ROOT / "artifacts/recurrent_evidence_v1"
PUBLIC = ROOT / "references/evidence/recurrent_evidence_v1"
MODEL = ROOT / "models/recurrent_evidence"
ARCHIVE = ROOT / "data/raw/dureader2/dureader_raw.zip"
DOLLY = ROOT / "data/general_instructions/dolly15k"
SEED = "recurrent-evidence-1"


def digest(s):
    return sha256(s.encode()).hexdigest()


def pieces(q, segments, english):
    def keys(t):
        if english:
            return set(re.findall(r"[a-z0-9]+", t.lower()))
        return {t[i : i + 2] for i in range(len(t) - 1)}

    cue = keys(q)
    best = None
    for text in segments:
        text = clean(text)
        for offset in range(0, min(len(text), 2400), 420):
            v = text[offset : offset + 600]
            if len(v) < 8:
                break
            score = len(keys(v) & cue) / (len(v) + 60) ** 0.22
            if best is None or score > best[0]:
                best = score, v
            if offset + 600 >= len(text):
                break
    return best[1] if best else ""


def raw(language, split):
    if language == "en":
        with (DOLLY / (split + ".jsonl")).open() as f:
            for line in f:
                r = json.loads(line)
                if not r["context"]:
                    continue
                q = clean(r["instruction"])[:160]
                yield (
                    r["id"],
                    q,
                    pieces(q, [r["context"]], True),
                    [clean(r["answer"])],
                    r["category"],
                )
    else:
        member = "train" if split == "train" else "dev"
        excluded = {
            r["question_id"]
            for r in read(
                ROOT / "data/composed_chinese_answers_v1/development/sources.json"
            )
        }
        with ZipFile(ARCHIVE) as z:
            for part in ("search", "zhidao"):
                with z.open(f"raw/{member}set/{part}.{member}.json") as f:
                    for i, line in enumerate(f):
                        if split == "train" and i >= 20000:
                            break
                        r = json.loads(line)
                        key = str(r["question_id"])
                        if key in excluded:
                            continue
                        q = clean(r["question"])[:160]
                        segments = [
                            " ".join(d["paragraphs"][:3]) for d in r["documents"][:5]
                        ]
                        yield (
                            key,
                            q,
                            pieces(q, segments, False),
                            list(
                                dict.fromkeys(
                                    clean(v) for v in r["answers"] if v.strip()
                                )
                            ),
                            r["question_type"],
                        )


def fragments(s, stride):
    text = normalize(s)
    # Use hashes of exactly the same normalized substrings on both sides.
    return {
        digest(text[i : i + 72])[:20] for i in range(0, len(text) - 71, stride)
    } or {digest(text)[:20]}


def prepared(receipt, work):
    if not receipt.is_file():
        return False
    files = read(receipt)["files"]
    for name, expected in files.items():
        path = work / name
        if path.exists() and sha(path) != expected:
            raise ValueError(f"Corpus digest mismatch: {path}")
    return all((work / name).is_file() for name in files)


def record_corpus(receipt, data):
    if receipt.exists():
        if read(receipt) != data:
            raise ValueError(
                "Rebuilt corpus differs from the published split: " + str(receipt)
            )
    else:
        dump(receipt, data)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    receipt = PUBLIC / "corpus.json"
    if prepared(receipt, WORK):
        return
    if (
        sha(ARCHIVE)
        != read(ROOT / "data/composed_chinese_answers_v1/preparation.json")[
            "archive_sha256"
        ]
    ):
        raise ValueError("Changed DuReader archive")
    enc = Encoder()
    with np.load(PREVIOUS / "corpus.npz") as old:
        enc.tiles = list(map(bytes, old["tiles"]))
        enc.chars = list(map(str, old["chars"]))
        enc.shapes = {v: i for i, v in enumerate(enc.tiles) if i >= 2}
        po, p = old["po"], old["p"]
        chars = enc.chars
        prior = set()
        for l, r in zip(po[:-1], po[1:], strict=True):
            prior.update(fragments("".join(chars[i] for i in p[l:r]), 48))
    for p in read(PREVIOUS / "held.json"):
        prior.update(fragments(p["source"], 48))
    held = []
    blocked = set()
    used_q = set()
    counts = Counter()
    for lang in ("zh", "en"):
        rows = sorted(
            raw(lang, "development"), key=lambda r: digest(SEED + lang + r[0])
        )
        used = set()
        for subset, count in (("absent", 96), ("additional", 32)):
            got = 0
            for key, q, p, refs, task in rows:
                if key in used or normalize(q) in used_q or not q or not p or not refs:
                    continue
                off = all(
                    normalize(v) not in normalize(p)
                    and normalize(v) not in normalize(q)
                    for v in refs
                )
                if subset == "absent" and not (off and 8 <= len(refs[0]) <= 144):
                    continue
                f = fragments(p, 1)
                if f & (prior | blocked):
                    continue
                try:
                    x, s = enc.ids(q), enc.ids(p)
                except ValueError:
                    continue
                held.append(
                    dict(
                        id=key,
                        lang=lang,
                        subset=subset,
                        absent=off,
                        question=q,
                        source=p,
                        refs=refs,
                        q=x,
                        s=s,
                        task=task,
                    )
                )
                used.add(key)
                blocked.update(fragments(p, 24))
                used_q.add(normalize(q))
                got += 1
                if got == count:
                    break
            if got != count:
                raise ValueError((lang, subset, got))
    dump(WORK / "held.json", held)
    samples = []
    qs = []
    ss = []
    ys = []
    seen = set()
    for lang in ("zh", "en"):
        for key, q, p, refs, task in raw(lang, "train"):
            if not q or not p or not refs or not refs[0]:
                continue
            h = normalize(q)
            if h in used_q or (h, normalize(p)) in seen or fragments(p, 1) & blocked:
                continue
            try:
                x, s = enc.ids(q), enc.ids(p)
                y = enc.ids(refs[0][:96])
            except ValueError:
                continue
            if len(refs[0]) < 96:
                y.append(1)
            samples.append((lang, key, task))
            qs.append(x)
            ss.append(s)
            ys.append(y)
            seen.add((h, normalize(p)))
            counts[lang] += 1
    fields = {}
    for name, items in (("q", qs), ("s", ss), ("y", ys)):
        fields[name] = np.concatenate(items).astype(np.uint16)
        fields[name + "o"] = np.r_[0, np.cumsum(list(map(len, items)))]
    if len(enc.tiles) > 65535:
        raise ValueError("Glyph index overflow")
    fields.update(
        tiles=np.asarray([np.frombuffer(g, np.uint8) for g in enc.tiles]),
        chars=np.asarray(enc.chars),
    )
    np.savez(WORK / "train.npz", **fields)
    dump(WORK / "rows.json", samples)
    record_corpus(
        receipt,
        dict(
            training=dict(counts),
            held=[dict(id=r["id"], lang=r["lang"], subset=r["subset"]) for r in held],
            source_archive=sha(ARCHIVE),
            dolly={s: sha(DOLLY / (s + ".jsonl")) for s in ("train", "development")},
            files={f: sha(WORK / f) for f in ("train.npz", "rows.json", "held.json")},
        ),
    )
    print(dict(counts), len(held), len(enc.tiles), flush=True)


if __name__ == "__main__":
    main()
