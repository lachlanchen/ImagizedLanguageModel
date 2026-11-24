"""Prepare pictured Chinese exam questions with complete answer targets and no option input."""

import json
import re
from collections import Counter, defaultdict
from hashlib import sha1, sha256
import numpy as np
import requests

from scripts.prepare_c3_visual_reading import REVISION, FILES
from scripts.prepare_fact_binding import ROOT, read, sha
from scripts.prepare_fact_binding import dump as dump
from scripts.prepare_multiple_ink_runs import prepared_files, normalize, fragments
from scripts.evaluate_competing_facts import CorpusFace
from scripts.write_visual_discourse import DEFAULT

WORK = ROOT / "artifacts/open_exam_v1"
PUB = ROOT / "references/evidence/open_exam_v1"
MODEL = ROOT / "models/open_exam"
SOURCE = ROOT / "data/evaluation/c3_m_development_v1"
BLOBS = {
    "d-dev": "05f2622cce8aed0df95b8c457c2bcb84f4b5ed7e",
    "d-train": "6c3a8c25382266b8fd2d012fa66a0163a753f765",
    "m-dev": "32f7aa82d01a23e88d1e37eb21b4289fc008975a",
    "m-train": "2cdc3feacd93988c916099a23fd35a80a286f8e1",
}
CUE = re.compile("下列|以下|选项|说法|哪一项|哪个选项|哪项|选择|哪句|哪句话|哪一说法")


def pinned_source(name, blob):
    path = SOURCE / name
    if path.is_file():
        raw = path.read_bytes()
    else:
        response = requests.get(
            f"https://raw.githubusercontent.com/nlpdata/c3/{REVISION}/{name}",
            timeout=60,
        )
        response.raise_for_status()
        raw = response.content
    if sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != blob:
        raise ValueError(f"Pinned exam source differs: {name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return raw


def source(k):
    return json.loads(pinned_source("data/c3-" + k + ".json", BLOBS[k]))


def key(x):
    return sha256(("open-exam-v1 " + x).encode()).digest()


def build(stage, excluded_held=()):
    pinned_source("license.txt", FILES["license.txt"][1])
    previous = set(read(PUB / "reserved-documents.json"))
    covered = set()
    trained_pages = set()
    stats = Counter()
    groups = defaultdict(list)
    vals = list(map(bytes, np.load(DEFAULT / "glyphs.npy")))
    n = len(vals)
    lookup = {v: i for i, v in enumerate(vals) if i >= 2}
    face = CorpusFace()

    def encode(s):
        ids = []
        for c in s:
            try:
                b = face.tile(c)
            except ValueError:
                b = face.tile("�")
                stats["fallback_cells"] += 1
            if b not in lookup:
                lookup[b] = len(vals)
                vals.append(b)
            ids.append(lookup[b])
        return ids

    for split in ("train", "dev"):
        for genre in ("m", "d"):
            records = source(genre + "-" + split)
            for paras, questions, doc in sorted(
                records, key=lambda r: key(genre + r[2] + "".join(r[0]))
            ):
                text = " ".join(" ".join(p.split()) for p in paras).strip()
                if not text:
                    continue
                page = text[:600]
                doc_id = genre + "-" + doc
                digest = normalize(page)
                if split == "dev" and (
                    doc in previous
                    or digest in trained_pages
                    or fragments(page, 1) & covered
                ):
                    stats["blocked_dev_documents"] += 1
                    continue
                if split == "train":
                    covered.update(fragments(page, 32))
                    trained_pages.add(digest)
                seq = encode(page)
                rows = []
                for j, z in enumerate(questions):
                    q = " ".join(z["question"].split())[:160]
                    a = " ".join(z["answer"].split())
                    if not q or not a:
                        continue
                    opts = [" ".join(v.split()) for v in z["choice"]]
                    if a not in opts:
                        raise ValueError("No labelled choice")
                    qi = encode(q)
                    y = encode(a) + [1]
                    if split == "train" and (
                        len(y) > 65
                        or any(
                            i >= n and i not in seq and i not in (seq + qi)[-144:]
                            for i in y
                        )
                    ):
                        stats["excluded_training_targets"] += 1
                        continue
                    rows.append(
                        dict(
                            id=doc_id + "/" + str(j),
                            doc=doc_id,
                            genre=genre,
                            q=q,
                            p=page,
                            qi=qi,
                            pi=seq,
                            answer=a,
                            y=y,
                            opts=opts,
                            ys=[encode(v) + [1] for v in opts],
                            choice_cue=bool(CUE.search(q)),
                            absent=normalize(a) not in digest,
                        )
                    )
                if rows:
                    groups[split + "-" + genre].append(rows)
    train = sum((g for k in ("train-m", "train-d") for g in groups[k]), [])
    held = []
    seen = {normalize(r["p"]) for r in excluded_held}

    def windows(page):
        text = normalize(page)
        return {text[i : i + 32] for i in range(len(text) - 31)} or {text}

    prior_windows = {f for r in excluded_held for f in windows(r["p"])}
    prior_docs = {r["doc"] for r in excluded_held}
    for genre in ("m", "d"):
        remaining = []
        paired = 0
        for g in groups["dev-" + genre]:
            if (
                normalize(g[0]["p"]) in seen
                or g[0]["doc"] in prior_docs
                or (prior_windows and windows(g[0]["p"]) & prior_windows)
            ):
                continue
            other = next(
                (
                    z
                    for z in g[1:]
                    if normalize(z["q"]) != normalize(g[0]["q"])
                    and normalize(z["answer"]) != normalize(g[0]["answer"])
                ),
                None,
            )
            if other and paired < 64:
                for r in (g[0], other):
                    held.append(dict(r, cohort="paired"))
                seen.add(normalize(g[0]["p"]))
                paired += 1
            else:
                remaining.append(g)
        count = 0
        for g in remaining:
            if normalize(g[0]["p"]) in seen:
                continue
            held.append(dict(g[0], cohort="general"))
            seen.add(normalize(g[0]["p"]))
            count += 1
            if count == 64:
                break
        stats[genre + "_pairs"] = paired
        stats[genre + "_general"] = count
    # Sibling exchange on multi-question training pages. Unpaired rows receive a
    # permuted question of similar length within the same source genre.
    by = defaultdict(list)
    for j, r in enumerate(train):
        by[r["doc"]].append(j)
    singles = defaultdict(list)
    for members in by.values():
        if len(members) > 1:
            for i, j in zip(members, members[1:] + members[:1]):
                train[i]["swap"] = j
        else:
            singles[train[members[0]]["genre"]].append(members[0])
    for inds in singles.values():
        inds.sort(key=lambda i: (len(train[i]["qi"]), key(train[i]["id"])))
        for i, j in zip(inds, inds[1:] + inds[:1]):
            train[i]["swap"] = j
    assert sorted(r["swap"] for r in train) == list(range(len(train)))
    stats["unchanged_swapped_questions"] = sum(
        r["qi"] == train[r["swap"]]["qi"] for r in train
    )
    for name, rows in (("train", train), ("held", held)):
        with (stage / (name + ".json")).open("w") as f:
            json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    np.save(
        stage / "tiles.npy", np.frombuffer(b"".join(vals), np.uint8).reshape(-1, 128)
    )
    return dict(
        revision=REVISION,
        blobs=BLOBS,
        notice=sha(SOURCE / "license.txt"),
        canonical=n,
        train=len(train),
        held=len(held),
        stats=dict(stats),
        files={k: sha(stage / k) for k in ("train.json", "held.json", "tiles.npy")},
    )


if __name__ == "__main__":
    prepared_files(
        WORK,
        PUB / "corpus.json",
        {n: ("files", n) for n in ("train.json", "held.json", "tiles.npy")},
        build,
    )
