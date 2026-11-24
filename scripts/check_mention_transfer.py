"""Evaluate frozen pictured pronoun readers on independent GAP and CLUE examples."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import zipfile
import numpy as np
import requests
import torch
from scripts.prepare_story_mentions import (
    ROOT,
    RUN,
    PUBLIC,
    CELLS,
    MAX_SPAN,
    dump,
    read,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_prose_fields import glyph_bank
from scripts.train_story_mentions import load
from scripts.train_masked_mentions import net_for
from scripts.train_semantic_page import gpu_owner
from scripts.story_mention_options import symbols, collapse, proposals, select
from scripts.write_story_mentions import location
from ilm.visual_lm.book_prefix import observations

SOURCES = {
    "gap-development.tsv": (
        "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/83135f26025cdd705c04f77e940d0c9201116b80/gap-development.tsv",
        "b9a01434fcf58d8c2f9bc762480c27e58ce466cf1ffe8b09cfecbc7a20d2d634",
    ),
    "gap-test.tsv": (
        "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/83135f26025cdd705c04f77e940d0c9201116b80/gap-test.tsv",
        "1c35e36d5b14f6313ec3f6cd67b275de282595dd59e59390e00cfff9897a6819",
    ),
    "cluewsc.zip": (
        "https://storage.googleapis.com/cluebenchmark/tasks/cluewsc2020_public.zip",
        "ebe5f357b19392d601be401b2fd4246983f534809a078eab7bda1de418fb3ffe",
    ),
}


def source(name):
    path = ROOT / "data/raw/story_mentions_external" / name
    url, expected = SOURCES[name]
    if path.exists():
        data = path.read_bytes()
    else:
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        data = res.content
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("External source digest mismatch")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def crop(key, text, marked, cast, truth):
    qa, q = marked
    qb = qa + len(q)
    if (
        not q
        or text[qa:qb] != q
        or any(text[i : i + len(s)] != s or not s for i, s in cast)
    ):
        return "offset"
    if any(len(s) > MAX_SPAN for i, s in cast):
        return "long_name"
    if any(i + len(s) > qa for i, s in cast):
        return "not_earlier"
    if len({s for i, s in cast}) != len(cast):
        return "duplicate"
    first = min(i for i, s in cast)
    if qb - first > CELLS:
        return "too_distant"
    end = min(len(text), qb + 32, first + CELLS)
    left = max(0, end - CELLS)
    return dict(
        id=key,
        text=text[left:end].rstrip(),
        query=[qa - left, qb - left],
        cast=[(i - left, i + len(s) - left) for i, s in cast],
        truth=truth,
        offset=left,
    )


def datasets(training=False):
    groups = {"gap": [], "clue": []}
    counts = {k: Counter() for k in groups}
    for r in csv.DictReader(
        io.StringIO(
            source("gap-development.tsv" if training else "gap-test.tsv").decode()
        ),
        delimiter="\t",
    ):
        names = [(int(r[c + "-offset"]), r[c]) for c in ("A", "B")]
        flags = [
            i for i, c in enumerate(("A", "B")) if r[c + "-coref"].upper() == "TRUE"
        ]
        record = (
            "multiple_labels"
            if len(flags) > 1
            else crop(
                r["ID"],
                r["Text"],
                (int(r["Pronoun-offset"]), r["Pronoun"]),
                names,
                flags[0] if flags else None,
            )
        )
        if isinstance(record, str):
            counts["gap"][record] += 1
        else:
            record["article"] = r["URL"]
            groups["gap"].append(record)
    with zipfile.ZipFile(io.BytesIO(source("cluewsc.zip"))) as z:
        for line in z.open("train.json" if training else "dev.json"):
            r = json.loads(line)
            t = r["target"]
            record = crop(
                str(r["idx"]),
                r["text"],
                (t["span2_index"], t["span2_text"]),
                [(t["span1_index"], t["span1_text"])],
                0 if r["label"] == "true" else None,
            )
            if isinstance(record, str):
                counts["clue"][record] += 1
            else:
                record["original"] = r["text"]
                groups["clue"].append(record)
    return groups, counts


@torch.inference_mode()
def evaluate(rows, device="cuda"):
    chars = sorted(set(" " + "".join(r["text"] for r in rows)))
    lookup = {s: i for i, s in enumerate(chars)}
    bank = glyph_bank(chars, Rasterizer())
    out = defaultdict(list)
    for mode in ("ink", "byte"):
        rawnet = load(mode, device)
        arms = [(mode, net_for(mode, device))]
        if mode == "ink":
            arms.append(("geometry", net_for("geometry", device)))
        obs = torch.as_tensor(observations(bank, mode), device=device)
        for first in range(0, len(rows), 32):
            batch = rows[first : first + 32]
            size = len(batch)
            codes = np.full((size, CELLS), lookup[" "], np.int32)
            for j, r in enumerate(batch):
                codes[j, : len(r["text"])] = [lookup[c] for c in r["text"]]

            def tensor(x):
                return torch.as_tensor(x, device=device)

            logits = (
                rawnet(
                    obs[tensor(codes).long()],
                    tensor([r["query"] for r in batch]),
                    tensor([len(r["text"]) for r in batch]),
                )[0]
                .cpu()
                .numpy()
            )
            for route in ("cast", "free"):
                src = np.zeros_like(codes)
                ids = np.full_like(codes, -1)
                meta = []
                bounds = []
                sizes = []
                for j, (r, line) in enumerate(zip(batch, logits)):
                    real = codes[j, : len(r["text"])]
                    qas = r["query"]
                    seq = symbols(bank["tiles"][real])
                    cand = r["cast"] if route == "cast" else proposals(line, qas[0])
                    index, slots, spans, q = collapse(seq, qas, cand)
                    n = len(index)
                    src[j, :n] = real[index]
                    ids[j, :n] = slots
                    sizes.append(n)
                    bounds.append(q)
                    meta.append((slots, spans, q))
                    if route == "free":
                        out[mode + "-original"].append(location(line.argmax(), CELLS))
                    if mode == "ink":
                        eligible = [
                            s for i, s in enumerate(spans[: q[0]]) if slots[i] >= 0
                        ]
                        out["near-" + route].append(eligible[-1] if eligible else None)
                x = obs[tensor(src).long()]
                for arm, net in arms:
                    scores = (
                        net(
                            torch.zeros_like(x) if arm == "geometry" else x,
                            tensor(ids),
                            tensor(bounds),
                            tensor(sizes),
                        )
                        .cpu()
                        .numpy()
                    )
                    out[arm + "-" + route].extend(
                        select(line, s, spans, q[0])
                        for line, (s, spans, q) in zip(scores, meta)
                    )
    return out


def score(rows, choices, binary):
    table = {}
    for arm, dec in choices.items():
        c = Counter()
        for r, p in zip(rows, dec, strict=True):
            text = r["text"]
            returned = None if p is None else text[slice(*p)]
            wanted = None if r["truth"] is None else text[slice(*r["cast"][r["truth"]])]
            if binary:
                positive = returned == text[slice(*r["cast"][0])]
                correct = positive == (wanted is not None)
            else:
                correct = returned == wanted
            c["n"] += 1
            c["correct"] += correct
            k = "positive" if wanted is not None else "negative"
            c[k] += 1
            c[k + "_correct"] += correct
        table[arm] = dict(c)
    return table


def main():
    groups, excluded = datasets()
    train = {r["text"] for r in read(RUN / "train.json")}
    with gpu_owner():
        for source_id, rows in groups.items():
            if any(r["text"] in train for r in rows):
                raise ValueError("External passage collision")
            choices = evaluate(rows)
            choices["always-null"] = [None] * len(rows)
            choices["always-first"] = [r["cast"][0] for r in rows]
            m = score(rows, choices, source_id == "clue")
            dump(
                PUBLIC / (source_id + "-scores.json"),
                dict(
                    exclusions=excluded[source_id], arms=m, training_page_collisions=0
                ),
            )
            index = [
                dict(
                    id=r["id"],
                    offset=r["offset"],
                    sha256=hashlib.sha256(r["text"].encode()).hexdigest(),
                )
                for r in rows
            ]
            (PUBLIC / (source_id + "-decisions.json.gz")).write_bytes(
                gzip.compress(
                    json.dumps(
                        dict(index=index, choices=choices), separators=(",", ":")
                    ).encode(),
                    mtime=0,
                )
            )
            print(
                source_id,
                len(rows),
                {k: v["correct"] for k, v in m.items()},
                flush=True,
            )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
