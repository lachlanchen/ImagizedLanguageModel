"""Build aligned bilingual reading and writing sequences and hold out passage families."""

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
import requests

from scripts.check_winograd_images import schemas, extract
from scripts.prepare_story_mentions import source, documents
from scripts.prepare_dialogue_ink import ROOT, dump, sha
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.download_alpaca import EN_CANDIDATES, ZH_CANDIDATES

BASE = Path(os.environ.get("ILM_DISCOURSE_RUN", ROOT / "artifacts/visual_discourse_v1"))
PUBLIC = (
    BASE / "scores"
    if "ILM_DISCOURSE_RUN" in os.environ
    else ROOT / "references/evidence/visual_discourse_v1"
)
MODEL = (
    BASE / "model"
    if "ILM_DISCOURSE_RUN" in os.environ
    else ROOT / "models/visual_discourse"
)
REV = "01e74176c63542e6b0bcb004dcdea22d94fb67b5b0ab83a67066"
DIGESTS = {
    "train": "5d8c38ad12b9a6c88f79b6e00aaf0f40781f93d4f94816f6cff2b67625d67399",
    "validation": "f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1",
}
ALPACA = {
    "en": "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
    "zh": "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903",
}
WINDOW = 256


def read(path):
    return json.loads(path.read_text())


def fetch(path, digest, urls):
    if not path.exists():
        for url in urls:
            response = requests.get(url, timeout=60)
            if response.ok and hashlib.sha256(response.content).hexdigest() == digest:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(response.content)
                tmp.replace(path)
                break
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"Source did not match the pinned hash: {path}")
    return path


def clean(text):
    return " ".join(text.split())


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(sentence, names):
    text = sentence.casefold()
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(
            r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)", " entity ", text
        )
    return re.findall(r"\w+", text)


def shingles(words):
    return {" ".join(words[j : j + 8]) for j in range(len(words) - 7)} or {
        " ".join(words)
    }


def reference_row(label, seq, text, opts, correct):
    return dict(
        task="reference",
        id=f"{label}-{seq}",
        prompt=text + " → ",
        answer=opts[correct],
        other=opts[1 - correct],
        choices=opts,
        family=key(" ".join(normalize(text.split("_", 1)[0], opts))),
        split=label,
    )


def wino():
    by_split = defaultdict(list)
    excluded = Counter()
    for split, digest in DIGESTS.items():
        file = split + "-00000-of-00001.parquet"
        path = fetch(
            ROOT / "data/raw/winogrande" / REV / file,
            digest,
            [
                f"https://huggingface.co/datasets/allenai/winogrande/resolve/{REV}/winogrande_xl/{file}"
            ],
        )
        for i, r in enumerate(pq.read_table(path).to_pylist()):
            text = clean(r["sentence"])
            opts = [clean(r[f"option{k}"]) for k in (1, 2)]
            if (
                text.count("_") != 1
                or len(text) + max(map(len, opts)) + 4 > WINDOW
                or opts[0].casefold() == opts[1].casefold()
            ):
                excluded[split + "_format"] += 1
                continue
            by_split[split].append(
                reference_row(split, i, text, opts, int(r["answer"]) - 1)
            )
    # Reserve the exact original pair identities from the earlier image trial.
    old = read(ROOT / "references/evidence/story_mentions_v1/winograd-pairs.json")
    entries = [extract(s) for s in schemas()]
    for n, pair in enumerate(old["decisions"]):
        rs = []
        for ix in pair["xml_rows"]:
            t, q, spans, truth, _ = entries[ix - 1]
            text = t[: q[0]] + "_" + t[q[1] :]
            names = [t[a:b] for a, b in spans]
            rs.append(reference_row("winograd", ix, text, names, truth))
        if any(
            len(r["prompt"]) + max(len(r["answer"]), len(r["other"])) + 1 > WINDOW
            for r in rs
        ):
            excluded["winograd_long_pair"] += 1
            continue
        for r in rs:
            r["pair"] = n
        by_split["winograd"].extend(rs)
    out = by_split["validation"] + by_split["winograd"]
    official = {k for r in out for k in shingles(normalize(r["prompt"], r["choices"]))}
    reserved = []
    train = []
    for r in by_split["train"]:
        sh = shingles(normalize(r["prompt"], r["choices"]))
        if sh & official:
            excluded["official_overlap"] += 1
            continue
        if int(r["family"][:8], 16) % 16 == 0:
            r["split"] = "reserve"
            reserved.append(r)
        else:
            train.append(r)
    reserved_keys = {
        k for r in reserved for k in shingles(normalize(r["prompt"], r["choices"]))
    }
    seen = set()
    for r in train:
        k = (r["prompt"], r["answer"])
        if k in seen or shingles(normalize(r["prompt"], r["choices"])) & reserved_keys:
            excluded["reserve_overlap_or_duplicate"] += 1
            continue
        seen.add(k)
        out.append(r)
    out += reserved
    # Opposite answers under unchanged spelling and identical visible prefix.
    for split in ("train", "reserve", "validation"):
        group = defaultdict(list)
        for r in out:
            if r["split"] == split:
                group[(r["prompt"].split("_")[0], tuple(sorted(r["choices"])))].append(
                    r
                )
        for i, ((prefix, opts), rs) in enumerate(group.items()):
            arms = [[r for r in rs if r["answer"] == s] for s in opts]
            for j, (x, y) in enumerate(zip(*arms)):
                if x["prompt"] != y["prompt"]:
                    x["pair"] = y["pair"] = f"{i}.{j}"
    return out, excluded


def other_rows():
    for lang, full in (("en", "english"), ("zh", "chinese")):
        for split in ("train", "test"):
            for doc, text, _, _ in documents(source(full, split)):
                # Keep each work intact in its assigned split. Windows can follow any character.
                yield dict(
                    id=doc,
                    task="book-" + lang,
                    split=split,
                    prompt="",
                    answer=clean(text),
                )
        path = fetch(
            ROOT / "data/raw" / f"alpaca_{lang}.json",
            ALPACA[lang],
            EN_CANDIDATES if lang == "en" else ZH_CANDIDATES,
        )
        rows = read(path)
        prepared = []
        for i, r in enumerate(rows):
            ask = clean(r["instruction"] + " " + r["input"])
            answer = clean(r["output"])
            if not ask or not answer:
                continue
            prompt = (
                ("Q: " + ask + " A: ") if lang == "en" else ("问：" + ask + " 答：")
            )
            prepared.append(
                dict(
                    id=f"{lang}-{i}",
                    task="instruction-" + lang,
                    split="test"
                    if int(key(ask.casefold())[:8], 16) % 40 == 0
                    else "train",
                    prompt=prompt,
                    answer=answer,
                )
            )
        held = {r["answer"] for r in prepared if r["split"] == "test"}
        for r in prepared:
            if r["split"] == "test" or r["answer"] not in held:
                yield r


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    rows, excluded = wino()
    rows.extend(other_rows())
    raster = Rasterizer()
    chars = set("".join(r["prompt"] + r["answer"] + r.get("other", "") for r in rows))
    shapes = {}
    bad = set()
    for ch in sorted(chars):
        try:
            shapes[ch] = raster.tile(ch)
        except ValueError:
            bad.add(ch)
    good = []
    for r in rows:
        if any(c in bad for c in r["prompt"] + r["answer"] + r.get("other", "")):
            excluded["unsupported-" + r["task"]] += 1
            continue
        good.append(r)
    # Preserve both members of every accepted evaluation pair or count neither.
    counts = Counter((r["split"], r["pair"]) for r in good if "pair" in r)
    for r in good:
        if "pair" in r and counts[(r["split"], r["pair"])] != 2:
            r.pop("pair")
    trainchars = set(
        "".join(r["prompt"] + r["answer"] for r in good if r["split"] == "train")
    )
    ordered = sorted(trainchars) + sorted(set(shapes) - trainchars)
    # IDs are only a lossless cache index into actual bitmaps. Merge physical aliases.
    indices = {}
    char_id = {}
    rep = ["", ""]
    tiles = [bytes(128), bytes(128)]
    for ch in ordered:
        shape = shapes[ch]
        if shape not in indices:
            indices[shape] = len(tiles)
            tiles.append(shape)
            rep.append(ch)
        char_id[ch] = indices[shape]
    ntrain = max(char_id[ch] for ch in trainchars) + 1
    if len(tiles) >= 65535:
        raise ValueError("Too many distinct shapes for the cache")
    palette = np.frombuffer(b"".join(tiles), np.uint8).reshape(-1, 128)
    np.save(MODEL / "glyphs.npy", palette[:ntrain])
    corpus = []
    metadata = []
    offset = 0
    for r in good:
        seq = np.array(
            [char_id[ch] for ch in r["prompt"] + r["answer"]] + [1], np.uint16
        )
        record = {
            k: v
            for k, v in r.items()
            if k not in ("prompt", "answer", "other", "family", "choices")
        }
        record.update(start=offset, cut=len(r["prompt"]), end=offset + len(seq))
        if "other" in r:
            record["foil"] = [char_id[ch] for ch in r["other"]] + [1]
        if "choices" in r:
            record["choices"] = r["choices"]
        metadata.append(record)
        corpus.append(seq)
        offset += len(seq)
    np.savez(
        BASE / "corpus.npz",
        stream=np.concatenate(corpus),
        tiles=palette,
        characters=np.array(rep),
    )
    dump(BASE / "records.json", metadata)
    dump(
        MODEL / "glyphs.json",
        dict(sha256=sha(MODEL / "glyphs.npy"), transcript=rep[:ntrain]),
    )
    summary = Counter(r["task"] + "/" + r["split"] for r in metadata)
    pairs = Counter(r["split"] for r in metadata if "pair" in r)
    dump(
        PUBLIC / "corpus.json",
        dict(
            rows=summary,
            pairs={k: v // 2 for k, v in pairs.items()},
            excluded=excluded,
            source_sha256={**DIGESTS, **ALPACA},
            winogrande_revision=REV,
            train_shapes=ntrain - 2,
            all_shapes=len(tiles) - 2,
            stored_positions=offset,
            corpus=sha(BASE / "corpus.npz"),
            records=sha(BASE / "records.json"),
            palette=sha(MODEL / "glyphs.npy"),
        ),
    )
    print(summary, pairs, excluded, ntrain, offset, flush=True)


if __name__ == "__main__":
    main()
