"""Build authentic one- and two-fact pictured questions from the pinned CrossWOZ splits."""

import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.prepare_chinese_card_reply import (
    RAW,
    ROOT,
    MODEL as PARENT,
    PUB as SOURCE,
    WIDTH,
    COPY,
    DOMAINS,
    FIELDS,
    extract,
    read,
    sha,
    dump,
    sources,
    value,
    Rasterizer,
)

WORK = ROOT / "artifacts/chinese_two_fact_v1"
PUB = ROOT / "references/evidence/chinese_two_fact_v1"
MODEL = ROOT / "models/chinese_two_fact"
# A separate run can train from scratch without replacing released evidence.
run_root = os.environ.get("ILM_TWO_FACT_RUN")
if run_root:
    run = Path(run_root).expanduser().resolve()
    WORK, PUB, MODEL = run / "data", run / "evidence", run / "model"

# SHA-256 of (group byte, 48 int64 sample indices) for the 2,400 registered
# batches. Both trainers check it without loading any earlier fitted network.
JOINED_SCHEDULE = "1456877b2626585a2707e458aa1d2332e8f62e9d11987ab9e09b6b6ab9d58196"
SECOND = "△"
MARKS = (*COPY, SECOND)


def double_rows(split, catalog):
    with zipfile.ZipFile(RAW / (split + ".json.zip")) as archive:
        dialogs = json.load(archive.open(split + ".json"))
    records = []
    for dialog, conversation in dialogs.items():
        user = None
        for turn, message in enumerate(conversation["messages"]):
            if message["role"] == "usr":
                user = message
                continue
            if user is None:
                continue
            requests = [(d, k) for a, d, k, _ in user["dialog_act"] if a == "Request"]
            informs = [
                (d, k, v) for a, d, k, v in message["dialog_act"] if a != "General"
            ]
            if len(requests) != 2 or len(informs) != 2 or len(set(requests)) != 2:
                continue
            if set(requests) != {(d, k) for d, k, _ in informs}:
                continue
            domain = requests[0][0]
            if any(
                d != domain or d not in catalog or k not in FIELDS for d, k in requests
            ):
                continue
            names = set(message["sys_state"][domain].get("selectedResults", []))
            names.update(
                v
                for a, d, k, v in user["dialog_act"]
                if (a, d, k) == ("Inform", domain, "名称")
            )
            if len(names) != 1:
                continue
            name = next(iter(names))
            entry = catalog[domain].get(name, {})
            card = {k: value(k, entry.get(k)) for k in FIELDS}
            card = {k: v for k, v in card.items() if v and len(k) + 1 + len(v) <= WIDTH}
            query, answer = user["content"], message["content"]
            if (
                not 0 < len(query) <= WIDTH
                or len(answer) > 120
                or len(name) + 3 > WIDTH
            ):
                continue
            if any(m in query + answer + name for m in MARKS):
                continue
            if any(
                card.get(k) != v
                or not v
                or v not in answer
                or list(card.values()).count(v) != 1
                for _, k, v in informs
            ):
                continue
            keys = sorted((k for _, k in requests), key=lambda k: answer.index(card[k]))
            a, b = (card[k] for k in keys)
            if a in b or b in a:
                continue
            mapping = {name: COPY[0], a: COPY[1], b: SECOND}
            if len(mapping) != 3:
                continue
            pattern = "|".join(
                re.escape(key) for key in sorted(mapping, key=len, reverse=True)
            )
            masked = re.sub(pattern, lambda m: mapping[m[0]], answer)
            if COPY[1] in masked and SECOND in masked:
                records.append(
                    dict(
                        dialog=dialog,
                        turn=turn,
                        q=query,
                        domain=domain,
                        name=name,
                        card=card,
                        keys=keys,
                        answer=answer,
                        masked=masked,
                    )
                )
    return records


def paired_training(singles):
    pools = defaultdict(list)
    for row in singles:
        pools[(row["domain"], row["name"])].append(row)
    rows = []
    for pool in pools.values():
        for left in pool:
            for right in pool:
                keys = [left["keys"][0], right["keys"][0]]
                if keys[0] == keys[1] or left["card"] != right["card"]:
                    continue
                query = left["q"] + right["q"]
                answer = left["answer"] + right["answer"]
                if len(query) > WIDTH or len(answer) > 120:
                    continue
                rows.append(
                    dict(
                        left,
                        q=query,
                        keys=keys,
                        answer=answer,
                        masked=left["masked"]
                        + right["masked"].replace(COPY[1], SECOND),
                    )
                )
    return rows


def main():
    for dest in (WORK, MODEL, PUB):
        dest.mkdir(parents=True, exist_ok=True)
    original = read(SOURCE / "corpus.json")
    if any(not (RAW / Path(route).name).is_file() for route in original["files"]):
        sources()
    for route, expected in original["files"].items():
        if sha(RAW / Path(route).name) != expected:
            raise ValueError("Changed CrossWOZ object: " + route)
    catalog = {d: dict(read(RAW / (f + "_db.json"))) for d, f in DOMAINS.items()}
    sets = {}
    counts = {}
    for split in ("train", "val", "test"):
        singles, _ = extract(split, catalog)
        singles = [dict(r, keys=[r["key"]]) for r in singles]
        pairs = double_rows(split, catalog)
        sets[split] = singles + pairs
        counts[split] = [len(singles), len(pairs)]
        if split == "train":
            sets["joined"] = paired_training(singles)
    inherited = read(PARENT / "readback.json")
    chars = list(inherited)
    raster = Rasterizer()
    # Reuse every parent action and learn only genuinely new physical symbols.
    palette = list(np.load(PARENT / "palette.npy"))
    bitmap = {bytes(tile): i for i, tile in enumerate(palette[1:], 1)}
    code = {c: i for i, c in enumerate(chars)}
    for c in [SECOND, *sorted({c for r in sets["train"] for c in r["masked"]})]:
        if c in code:
            continue
        tile = raster.tile(c)
        idx = bitmap.get(tile)
        if idx is None:
            idx = len(chars)
            bitmap[tile] = idx
            palette.append(np.frombuffer(tile, np.uint8))
            chars.append(c)
        code[c] = idx
    if code[SECOND] != len(inherited):
        raise ValueError("Second pointer needs distinct ink feedback")
    dump(MODEL / "alphabet.json", chars)
    np.save(MODEL / "palette.npy", np.asarray(palette))
    vocabulary = [""] + sorted(
        {c for rows in sets.values() for r in rows for c in r["q"]}
        | set("".join((*DOMAINS, *FIELDS)))
    )
    lookup = {c: i for i, c in enumerate(vocabulary)}

    def enc(s):
        return [lookup[c] for c in s] + [0] * (WIDTH - len(s))

    bits = np.stack(
        [
            np.frombuffer(raster.tile(c) if c else bytes(128), np.uint8)
            for c in vocabulary
        ]
    )
    octets = np.zeros((len(vocabulary), 4), np.uint8)
    for i, c in enumerate(vocabulary):
        octets[i, : len(c.encode())] = list(c.encode())
    np.savez_compressed(
        WORK / "bank.npz",
        tiles=bits,
        bytes=octets,
        labels=np.array([enc(k) for k in (*FIELDS, *DOMAINS)]),
    )
    hashes = {}
    for split, rows in sets.items():
        dump(WORK / (split + ".json"), rows)
        y = np.full((len(rows), 1 + max(len(r["masked"]) for r in rows)), -1, np.int16)
        for i, r in enumerate(rows):
            if split in ("train", "joined"):
                seq = [code[c] for c in r["masked"]] + [0]
                y[i, : len(seq)] = seq
        np.savez_compressed(
            WORK / (split + ".npz"),
            query=np.asarray([enc(r["q"]) for r in rows], np.int16),
            domain=np.array([list(DOMAINS).index(r["domain"]) for r in rows], np.int16),
            fields=np.array(
                [
                    [
                        FIELDS.index(r["keys"][0]),
                        FIELDS.index(r["keys"][1])
                        if len(r["keys"]) == 2
                        else len(FIELDS),
                    ]
                    for r in rows
                ],
                np.int16,
            ),
            available=np.array([[k in r["card"] for k in FIELDS] for r in rows], bool),
            answer=y,
        )
        hashes[split] = sha(WORK / (split + ".npz"))
    dump(
        PUB / "corpus.json",
        dict(
            splits=counts,
            joined=len(sets["joined"]),
            archives=hashes,
            bank=sha(WORK / "bank.npz"),
            palette=sha(MODEL / "palette.npy"),
            source=original["source"],
            classes=len(chars),
        ),
    )
    print(counts, "joined", len(sets["joined"]), len(chars), flush=True)


if __name__ == "__main__":
    main()
