"""Reserve new whole books and remove training windows containing held target surfaces."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import base64
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
import requests

from scripts import prepare_passage_selector as old

ROOT = old.ROOT
WORK = ROOT / "artifacts/causal_book_words_v1"
PUB = ROOT / "references/evidence/causal_book_words_v1"
MODEL = ROOT / "models/causal_book_words"
RULE = ROOT / "references/causal_book_words_protocol.md"
SEED = 371155
BASE = 1000003
U64 = (1 << 64) - 1


def signature(seq):
    value = 0
    for word in seq[-17:]:
        value = (value * BASE + int(word)) & U64
    return value


def windows(seq):
    """All contiguous 17-run signatures. Padding cannot collide except by hash collision."""
    out = np.zeros((len(seq), seq.shape[1] - 16), np.uint64)
    for j in range(17):
        out *= np.uint64(BASE)
        out += seq[:, j : j + out.shape[1]]
    return out


def pg(previous):
    stored = PUB / "data.json"
    if stored.exists():
        pool = old.base.read(stored)["books"]["gutenberg"]
    else:
        r = requests.get(
            "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o",
            params={
                "prefix": "test/",
                "maxResults": 1000,
                "fields": "items(name,size,generation,md5Hash)",
            },
            timeout=40,
        )
        r.raise_for_status()
        used = {b["name"] for b in previous["books"]["gutenberg"]}
        options = [
            b
            for b in r.json()["items"]
            if b["name"].endswith(".txt")
            and b["name"] not in used
            and 80000 <= int(b["size"]) <= 950000
        ]
        pool = sorted(
            options, key=lambda b: sha256((str(SEED) + b["name"]).encode()).digest()
        )[:16]

    def get(b):
        file = old.RAW / b["name"]
        old.download(old.GCS + b["name"] + "?generation=" + b["generation"], file)
        assert (
            base64.b64encode(hashlib.md5(file.read_bytes()).digest()).decode()
            == b["md5Hash"]
        )
        return dict(
            name=b["name"],
            generation=b["generation"],
            md5Hash=b["md5Hash"],
            sha=old.base.sha(file),
        )

    with ThreadPoolExecutor(4) as workers:
        return list(workers.map(get, pool))


def main():
    previous = old.base.read(old.PUB / "data.json")
    assert old.base.sha(old.SHARD) == old.SHARD_HASH
    assert all(old.base.sha(old.WORK / k) == v for k, v in previous["files"].items())
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    forms = old.base.read(old.WORK / "forms.json")
    handles = {w: i for i, w in enumerate(forms)}
    canonical = {}
    lower = []

    def word_id(w):
        if w not in handles:
            handles[w] = len(forms)
            forms.append(w)
        return handles[w]

    def num(w):
        key = w.lower()
        return canonical.setdefault(key, len(canonical) + 1)

    lower.extend(num(w) for w in forms)
    lower[0] = 0
    known = np.array(lower, np.uint64)
    blocked = set()
    for filename in ("lambada-validation.parquet", "lambada-test.jsonl"):
        path = old.RAW / filename
        texts = (
            pq.read_table(path, columns=["text"])["text"].to_pylist()
            if filename.endswith("parquet")
            else [json.loads(s)["text"] for s in path.read_text().splitlines()]
        )
        for t in texts:
            words = old.WORD.findall(old.normalize(t))
            if len(words) >= 17:
                blocked.add(signature([num(w) for w in words[-17:]]))
    for name in ("fiction", "gutenberg", "lambada"):
        with np.load(old.WORK / (name + ".npz")) as a:
            merged = np.column_stack((a["source"][:, -16:], a["target"]))
            h = windows(known[merged])
            blocked.update(map(int, h[merged[:, 16:] != 0]))
    with np.load(old.WORK / "train.npz") as a:
        source = a["source"]
        target = a["target"][:, 0]
    endpoint = set(
        map(int, windows(known[np.column_stack((source[:, -16:], target))]).ravel())
    )
    seen = blocked | endpoint
    files = {}
    books = {"fiction": [], "gutenberg": []}
    samples = {k: [] for k in books}
    used = {b["sha"] for items in previous["books"].values() for b in items}

    def sample(text, entry, split):
        found = list(old.WORD.finditer(old.normalize(text)))
        if not found:
            return
        normalized = found[0].string
        words = [m[0] for m in found]
        marks = np.array(
            [
                old.gap(
                    normalized[
                        m.end() : found[i + 1].start()
                        if i + 1 < len(found)
                        else len(normalized)
                    ]
                )
                for i, m in enumerate(found)
            ],
            np.uint8,
        )
        tokens = [num(w) for w in words]
        rng = np.random.default_rng(SEED ^ int(entry["sha"][:14], 16))
        group = len(books[split])
        added = []
        for task, limit in ((1, 48), (0, 96)):
            candidates = np.arange(96, len(words) - 2)
            if task:
                candidates = candidates[marks[candidates] == 1]
            n = 0
            for j in rng.permutation(candidates):
                if not all(len(w) <= 23 for w in words[j : j + 3]):
                    continue
                sigs = [signature(tokens[j + k - 16 : j + k + 1]) for k in range(3)]
                if any(h in seen for h in sigs):
                    continue
                seen.update(sigs)
                blocked.update(sigs)
                width = int(rng.integers(26, 97))
                src = np.zeros(96, np.int32)
                sep = np.zeros(96, np.uint8)
                src[-width:] = [word_id(w) for w in words[j - width : j]]
                sep[-width:] = marks[j - width : j]
                excerpt = normalized[found[j - width].start() : found[j].start()]
                added.append(
                    (
                        src,
                        sep,
                        [word_id(w) for w in words[j : j + 3]],
                        marks[j : j + 3] == 1,
                        task,
                        group,
                        excerpt,
                    )
                )
                n += 1
                if n == limit:
                    break
            if n != limit:
                raise ValueError("Short reserved book " + str(entry))
        samples[split].extend(added)
        books[split].append(entry)

    corpus = {}
    for idx, text, domain in old.fiction():
        h = sha256(text.encode()).hexdigest()
        if h not in used:
            corpus.setdefault(h, (idx, domain))
    chosen = {
        corpus[h][0]: dict(index=corpus[h][0], sha=h, domain=corpus[h][1])
        for h in sorted(
            corpus, key=lambda h: sha256((str(SEED) + h).encode()).digest()
        )[:48]
    }
    for idx, text, _ in old.fiction():
        if idx in chosen:
            sample(text, chosen[idx], "fiction")
    for b in pg(previous):
        assert b["sha"] not in used
        sample((old.RAW / b["name"]).read_text(encoding="utf-8-sig"), b, "gutenberg")
    for name, rows in samples.items():
        s, m, y, e, t, g, excerpt = zip(*rows)
        f = WORK / (name + ".npz")
        np.savez_compressed(
            f,
            source=np.stack(s),
            marks=np.stack(m),
            target=np.array(y, np.int32),
            ends=np.stack(e),
            end=np.array(e)[:, 0],
            task=np.array(t, np.uint8),
            group=np.array(g, np.int32),
        )
        old.base.dump(WORK / (name + "-sources.json"), excerpt)
        for p in (f, WORK / (name + "-sources.json")):
            files[p.name] = old.base.sha(p)
    old.base.dump(WORK / "forms.json", forms)
    files["forms.json"] = old.base.sha(WORK / "forms.json")
    held = np.array(sorted(blocked), np.uint64)
    keep = []
    interior = 0
    eligible = np.array([w.isascii() and w.isalpha() and len(w) <= 23 for w in forms])
    for j in range(0, len(source), 2048):
        x = source[j : j + 2048]
        ys = target[j : j + len(x)]
        full = np.column_stack((x, ys))
        sigs = windows(known[full])
        ids = np.searchsorted(held, sigs).clip(0, len(held) - 1)
        matches = (held[ids] == sigs) & (full[:, :-16] != 0)
        accepted = ~matches.any(1)
        keep.extend(accepted)
        interior += int(
            (eligible[x[:, 16:]] & (x[:, :-16] != 0) & accepted[:, None]).sum()
        )
    keep = np.array(keep)
    np.save(WORK / "keep.npy", keep)
    files["keep.npy"] = old.base.sha(WORK / "keep.npy")
    receipt = dict(
        books=books,
        files=files,
        protected=len(blocked),
        excluded=int((~keep).sum()),
        retained=int(keep.sum()),
        interior=interior,
        previous=old.base.sha(old.PUB / "data.json"),
        protocol=old.base.sha(RULE),
        seed=SEED,
    )
    if (PUB / "data.json").exists():
        assert receipt == old.base.read(PUB / "data.json")
    else:
        old.base.dump(PUB / "data.json", receipt)
    print(
        {k: receipt[k] for k in ("protected", "excluded", "retained", "interior")},
        flush=True,
    )


if __name__ == "__main__":
    main()
