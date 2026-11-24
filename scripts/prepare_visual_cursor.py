"""Prepare human dialogue and reading pages with document-level official holdouts."""

from collections import Counter, defaultdict
import hashlib

import numpy as np
import pyarrow.parquet as pq

from scripts.prepare_dialogue_ink import ROOT, SOURCES
from scripts.prepare_visual_discourse import clean, read, dump, sha, BASE as OLD
from scripts.prepare_visual_sequence_corpus import Rasterizer

WORK = ROOT / "artifacts/visual_cursor_v1"
PUBLIC = ROOT / "references/evidence/visual_cursor_v1"
MODEL = ROOT / "models/visual_cursor"
LIMIT = 768
FILES = {
    "cmrc": [
        (
            "cmrc2018/cmrc2018_train.json",
            "5497aa2f81908e31d6b0e27d99b1f90ab63a8f58fa92fffe5d17cf62eba0c212",
        ),
        (
            "cmrc2018/cmrc2018_dev.json",
            "e9ff74231f05c230c6fa88b84441ee334d97234cbb610991cd94b82db00c7f1f",
        ),
    ],
    "drcd": [
        (
            "drcd/DRCD_training.json",
            "5e6268091ab98f0bb858f03c77fba85e23b31e49a91638e4f22d0d8ec703f79a",
        ),
        (
            "drcd/DRCD_dev.json",
            "e236df03861ba241bd7b29628cd48b3b8e339c60137e041a4358b5320ed61928",
        ),
    ],
    "coqa": [("coqa/" + s + ".parquet", v) for s, v in SOURCES.items()],
}


def hash_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def records(kind, path):
    if kind == "coqa":
        for doc in pq.read_table(path).to_pylist():
            history = ""
            for q, a in zip(
                doc["questions"], doc["answers"]["input_text"], strict=True
            ):
                yield clean(doc["story"]), clean(q), [clean(a)], history
                history = clean(q + " " + a)
    else:
        for doc in read(path)["data"]:
            for p in doc["paragraphs"]:
                for qa in p["qas"]:
                    answers = list(
                        dict.fromkeys(
                            clean(a["text"]) for a in qa["answers"] if a["text"].strip()
                        )
                    )
                    yield clean(p["context"]), clean(qa["question"]), answers, ""


def page(kind, q, history, passage):
    if kind == "coqa":
        intro = (
            "Q: "
            + q[:190]
            + (" Earlier: " + history[:100] if history else "")
            + " Passage: "
        )
        end = " A: "
    else:
        intro = "问：" + q[:190] + " 原文："
        end = " 答："
    return intro + passage[: LIMIT - len(intro) - len(end)] + end


def prepare():
    WORK.mkdir(exist_ok=True, parents=True)
    PUBLIC.mkdir(exist_ok=True, parents=True)
    with np.load(OLD / "corpus.npz") as old:
        tiles = list(map(bytes, old["tiles"]))
        chars = list(map(str, old["characters"]))
    shape = {s: i for i, s in enumerate(tiles) if i >= 2}
    mapping = {}
    raster = Rasterizer()

    def ids(s):
        seq = []
        for ch in s:
            if ch not in mapping:
                try:
                    tile = raster.tile(ch)
                except ValueError:
                    mapping[ch] = None
                else:
                    if tile not in shape:
                        shape[tile] = len(tiles)
                        tiles.append(tile)
                        chars.append(ch)
                    mapping[ch] = shape[tile]
            if mapping[ch] is None:
                raise ValueError("unsupported")
            seq.append(mapping[ch])
        return seq

    held = set()
    seen = set()
    pools = defaultdict(list)
    excluded = Counter()
    inputs = []
    targets = []
    offsets = [0]
    ends = [0]
    examples = []
    for phase in (1, 0):
        for kind, files in FILES.items():
            filename, digest = files[phase]
            path = ROOT / "data/raw" / filename
            if sha(path) != digest:
                raise ValueError(filename)
            docs = defaultdict(list)
            for passage, q, answers, history in records(kind, path):
                key = hash_text(passage)
                if phase:
                    held.add(key)
                if not phase and key in held:
                    excluded["passage_overlap"] += 1
                    continue
                if not answers or not answers[0] or not q:
                    continue
                identity = (key, q, history, answers[0])
                if identity in seen:
                    continue
                seen.add(identity)
                docs[key].append((q, history, passage, answers))
            for key in sorted(docs):
                selected = []
                for q, history, p, refs in docs[key]:
                    prompt = page(kind, q, history, p)
                    try:
                        x = ids(prompt)
                        y = ids(refs[0]) + [1]
                    except ValueError:
                        excluded[kind + "_unrenderable"] += 1
                        continue
                    if phase:
                        if selected and (
                            q == selected[0]["question"]
                            or set(refs) & set(selected[0]["references"])
                        ):
                            continue
                        selected.append(
                            dict(
                                doc=key,
                                kind=kind,
                                prompt=prompt,
                                question=q,
                                references=refs,
                                x=x,
                                y=y,
                                source_clipped=len(p) + len(q) + 12 > LIMIT,
                                verbatim_in_view=any(r in prompt for r in refs),
                            )
                        )
                        if len(selected) == 2:
                            break
                    else:
                        pools[kind].append(len(inputs))
                        inputs.append(np.asarray(x, np.uint16))
                        targets.append(np.asarray(y, np.uint16))
                        offsets.append(offsets[-1] + len(x))
                        ends.append(ends[-1] + len(y))
                if phase and len(selected) == 2:
                    examples.extend(selected)
                    if sum(r["kind"] == kind for r in examples) >= 128:
                        break
    np.savez(
        WORK / "pages.npz",
        x=np.concatenate(inputs),
        y=np.concatenate(targets),
        u=np.array(offsets),
        v=np.array(ends),
        tiles=np.frombuffer(b"".join(tiles), np.uint8).reshape(-1, 128),
        chars=np.array(chars),
        **{k: np.array(v) for k, v in pools.items()},
    )
    dump(WORK / "held.json", examples)
    dump(
        PUBLIC / "corpus.json",
        dict(
            training={k: len(v) for k, v in pools.items()},
            held=Counter(r["kind"] for r in examples),
            exclusions=excluded,
            source=FILES,
            inputs=int(offsets[-1]),
            targets=int(ends[-1]),
            stored_tiles=len(tiles),
            pages=sha(WORK / "pages.npz"),
            test=sha(WORK / "held.json"),
        ),
    )
    print(read(PUBLIC / "corpus.json"))


if __name__ == "__main__":
    prepare()
