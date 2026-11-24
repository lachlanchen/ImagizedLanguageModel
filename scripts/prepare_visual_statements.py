"""Human QA2D statements, document exclusions and a separate fresh passage evaluation."""

from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from shutil import copyfileobj
import numpy as np
import pyarrow.parquet as pq
from scripts.prepare_recurrent_evidence import (
    ROOT,
    digest,
    fragments,
    dump,
    read,
    sha,
    prepared,
    record_corpus,
)
from scripts.prepare_paired_evidence import Encoder, clean, SOURCES as EARLIER
from scripts.evaluate_visual_cursor import normalize
from scripts.prepare_recurrent_evidence import WORK as CURRENT

WORK = ROOT / "artifacts/visual_statements_v1"
PUB = ROOT / "references/evidence/visual_statements_v1"
MODEL = ROOT / "models/visual_statements"
DIGESTS = {
    "train": "3fce9846e4ad2a409b548b39e5e333be34211452b066cd6bdaf7fd942e06fcad",
    "dev": "fd8a2843cd08410b96b49692aab1b817c6444aa46a0dee1a5e0d0511d923da85",
}


def detok(s):
    s = clean(s or "").replace("-LRB-", "(").replace("-RRB-", ")")
    s = re.sub(r"\s+([,.;:?!%)])", r"\1", s)
    s = re.sub(r"([(])\s+", r"\1", s)
    s = re.sub(r"\s+(['’](?:s|t|ve|re|m|d|ll)\b)", r"\1", s, flags=re.I)
    return s.replace(" n't", "n't")


def collect(split):
    p = ROOT / "data/raw/qa2d" / (split + "-00000-of-00001.parquet")
    if not p.exists():
        revision = "d38d3f42978e72c8c3ccc5dca0d3a2ac745f1fcf"
        url = f"https://huggingface.co/datasets/domenicrosati/QA2D/resolve/{revision}/data/{p.name}"
        p.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=p.parent, delete=False) as tmp:
            candidate = Path(tmp.name)
        try:
            with urlopen(url, timeout=60) as response, candidate.open("wb") as output:
                copyfileobj(response, output)
            if sha(candidate) != DIGESTS[split]:
                raise ValueError("QA2D download digest mismatch")
            candidate.replace(p)
        finally:
            candidate.unlink(missing_ok=True)
    if sha(p) != DIGESTS[split]:
        raise ValueError("QA2D digest mismatch")
    return pq.read_table(p).to_pylist()


def squad(split):
    p, h = EARLIER["squad"][split]
    file = ROOT / "data/raw" / p
    if sha(file) != h:
        raise ValueError("SQuAD digest mismatch")
    for a in read(file)["data"]:
        for p in a["paragraphs"]:
            for r in p["qas"]:
                yield (
                    r["id"],
                    clean(p["context"]),
                    clean(r["question"]),
                    [clean(t["text"]) for t in r["answers"]],
                )


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    if prepared(PUB / "corpus.json", WORK):
        return
    enc = Encoder()
    with np.load(CURRENT / "train.npz") as z:
        enc.tiles = list(map(bytes, z["tiles"]))
        enc.chars = list(map(str, z["chars"]))
        enc.shapes = {v: i for i, v in enumerate(enc.tiles) if i >= 2}
    pages = {k: p for k, p, q, a in squad(0)}
    held = []
    used_q = set()
    used_p = set()
    dev = sorted(
        collect("dev"),
        key=lambda r: digest("statement-split-1" + r["example_uid"] + r["question"]),
    )
    for subset, count in (("other", 32), ("squad", 96)):
        selected = 0
        for r in dev:
            k = r["example_uid"]
            q, ans, y = [detok(r[f]) for f in ("question", "answer", "turker_answer")]
            if (
                (subset == "squad") != (k in pages)
                or not q
                or not ans
                or not 6 <= len(y) <= 144
                or len(q) > 160
                or len(ans) > 144
            ):
                continue
            if normalize(y) in normalize(q) or normalize(y) in normalize(ans):
                continue
            p = pages.get(k, "")
            ph = digest(p) if p else ""
            if normalize(q) in used_q or (ph and ph in used_p):
                continue
            try:
                x, f, g = map(enc.ids, (q, ans, y))
                s = enc.ids(p[:600]) if p else []
            except ValueError:
                continue
            held.append(
                dict(
                    id=k,
                    subset=subset,
                    q=x,
                    f=f,
                    y=g,
                    s=s,
                    question=q,
                    source=p[:600],
                    fact=ans,
                    reference=y,
                )
            )
            used_q.add(normalize(q))
            used_p.add(ph)
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError((subset, selected))
    train = []
    seen = set()
    for r in collect("train"):
        k = r["example_uid"]
        q, ans, y = [detok(r[f]) for f in ("question", "answer", "turker_answer")]
        if not q or not ans or not y or len(q) > 160 or len(ans) > 144:
            continue
        if normalize(q) in used_q or (k in pages and digest(pages[k]) in used_p):
            continue
        tag = (q, ans, y)
        if tag in seen:
            continue
        try:
            x, f, g = map(enc.ids, (q, ans, y[:111]))
        except ValueError:
            continue
        train.append((x, f, g + ([1] if len(y) <= 111 else [])))
        seen.add(tag)
    # Exclude old page families and slots sharing a long scanned training passage fragment.
    seen_p = {digest(p) for p in pages.values()} | used_p
    for file in (
        ROOT / "artifacts/visual_cursor_v1/held.json",
        ROOT / "artifacts/paired_evidence_v1/held.json",
    ):
        seen_p.update(r["doc"] for r in read(file))
    blocked = set()
    for p in set(pages.values()):
        blocked.update(fragments(p[:600], 48))
    fresh = []
    for k, p, q, refs in sorted(
        squad(1), key=lambda r: digest("independent-statement-1" + r[0])
    ):
        ph = digest(p)
        if (
            ph in seen_p
            or normalize(q) in used_q
            or fragments(p[:600], 1) & blocked
            or len(q) > 160
        ):
            continue
        try:
            x, s = map(enc.ids, (q, p[:600]))
            f = enc.ids(refs[0])
        except ValueError:
            continue
        if not x or not f:
            continue
        fresh.append(dict(id=k, q=x, s=s, f=f, source=p[:600], question=q, refs=refs))
        seen_p.add(ph)
        blocked.update(fragments(p[:600], 24))
        if len(fresh) == 96:
            break
    if len(fresh) != 96:
        raise ValueError("Insufficient fresh passage families")
    arrays = {}
    for j, name in enumerate(("q", "f", "y")):
        arr = [r[j] for r in train]
        arrays[name] = np.concatenate(arr).astype(np.uint16)
        arrays[name + "o"] = np.r_[0, np.cumsum(list(map(len, arr)))]
    arrays.update(
        tiles=np.stack([np.frombuffer(v, np.uint8) for v in enc.tiles]),
        chars=np.asarray(enc.chars),
    )
    np.savez(WORK / "train.npz", **arrays)
    dump(WORK / "held.json", held)
    dump(WORK / "fresh.json", fresh)
    record_corpus(
        PUB / "corpus.json",
        dict(
            examples=len(train),
            held=[r["id"] for r in held],
            fresh=[r["id"] for r in fresh],
            sources=DIGESTS,
            files={f: sha(WORK / f) for f in ("held.json", "fresh.json", "train.npz")},
        ),
    )
    print(len(train), len(held), len(fresh), flush=True)


if __name__ == "__main__":
    main()
