"""Build human Chinese question, complete catalogue card and reply examples from CrossWOZ."""

import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import requests
from scripts.prepare_semantic_page import ROOT, sha, dump, Rasterizer
from scripts.train_semantic_page import read

REV = "df82c9fdff91b9b130f2d6b89110d3870ba6260e"
RAW = ROOT / "data/raw/crosswoz"
WORK = ROOT / "artifacts/chinese_reply_v1"
PUB = ROOT / "references/evidence/chinese_reply_v1"
MODEL = ROOT / "models/chinese_card_reply"
DOMAINS = {"景点": "attraction", "餐馆": "restaurant", "酒店": "hotel"}
FIELDS = (
    "地址",
    "评分",
    "营业时间",
    "人均消费",
    "酒店类型",
    "价格",
    "门票",
    "游玩时间",
)
WIDTH = 80
COPY = ("◇", "◎")


def sources():
    routes = [f"data/crosswoz/{s}.json.zip" for s in ("train", "val", "test")]
    routes += [f"data/crosswoz/database/{d}_db.json" for d in DOMAINS.values()]
    routes += ["LICENSE", "data/crosswoz/README.md"]
    url = "https://api.github.com/repos/thu-coai/CrossWOZ/git/trees/" + REV
    reply = requests.get(url, params={"recursive": 1}, timeout=30)
    reply.raise_for_status()
    tree = {i["path"]: i["sha"] for i in reply.json()["tree"]}
    RAW.mkdir(parents=True, exist_ok=True)
    result = {}
    for route in routes:
        p = RAW / Path(route).name
        if p.exists():
            b = p.read_bytes()
        else:
            resp = requests.get(
                "https://raw.githubusercontent.com/thu-coai/CrossWOZ/"
                + REV
                + "/"
                + route,
                timeout=60,
            )
            resp.raise_for_status()
            b = resp.content
        if hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest() != tree[route]:
            raise ValueError("Invalid original object " + route)
        if not p.exists():
            p.write_bytes(b)
        result[route] = sha(p)
    return result


def value(field, v):
    if v is None or v == "" or isinstance(v, (list, dict)):
        return ""
    if field == "评分":
        return f"{float(v):g}分"
    if field == "门票" and v == 0:
        return "免费"
    if field in ("门票", "价格", "人均消费"):
        return str(v) + "元"
    return str(v)


def extract(split, db):
    rows = []
    nturn = 0
    with zipfile.ZipFile(RAW / (split + ".json.zip")) as z:
        dialogues = json.load(z.open(split + ".json"))
    for ident, dialog in dialogues.items():
        question = None
        for turn, utter in enumerate(dialog["messages"]):
            if utter["role"] == "usr":
                question = utter
                continue
            nturn += 1
            if question is None:
                continue
            requests_ = [
                (d, s) for act, d, s, _ in question["dialog_act"] if act == "Request"
            ]
            info = [
                (d, s, v) for act, d, s, v in utter["dialog_act"] if act != "General"
            ]
            if len(requests_) != 1 or len(info) != 1 or requests_[0] != info[0][:2]:
                continue
            domain, key, annotation = info[0]
            if key not in FIELDS or domain not in db:
                continue
            names = {
                v
                for a, d, s, v in question["dialog_act"]
                if a == "Inform" and d == domain and s == "名称"
            }
            names.update(utter["sys_state"][domain].get("selectedResults", []))
            if len(names) != 1:
                continue
            name = next(iter(names))
            entry = db[domain].get(name)
            if not entry:
                continue
            fields = {k: value(k, entry.get(k)) for k in FIELDS}
            fields = {
                k: v for k, v in fields.items() if v and len(v) + len(k) + 1 <= WIDTH
            }
            v = fields.get(key)
            answer = utter["content"]
            q = question["content"]
            if (
                not v
                or v != annotation
                or v not in answer
                or len(set(fields.values())) < 2
            ):
                continue
            if (
                list(fields.values()).count(v) != 1
                or not 1 <= len(q) <= WIDTH
                or len(answer) > 120
            ):
                continue
            # The test artifact keeps the original reply. Fitting labels use two image-span actions.
            if any(m in q + answer + name for m in COPY):
                continue
            masked = answer.replace(name, COPY[0]).replace(v, COPY[1])
            if COPY[1] not in masked:
                continue
            rows.append(
                dict(
                    dialog=ident,
                    turn=turn,
                    q=q,
                    domain=domain,
                    name=name,
                    card=fields,
                    key=key,
                    answer=answer,
                    masked=masked,
                )
            )
    return rows, {
        "dialogues": len(dialogues),
        "system_turns": nturn,
        "eligible": len(rows),
    }


def main():
    for d in (MODEL, WORK, PUB):
        d.mkdir(exist_ok=True, parents=True)
    manifests = sources()
    db = {d: dict(read(RAW / (f + "_db.json"))) for d, f in DOMAINS.items()}
    sets = {}
    summary = {}
    for s in ("train", "val", "test"):
        sets[s], summary[s] = extract(s, db)
    train = sets["train"]
    alphabet = (
        [""]
        + sorted({c for r in train for c in r["masked"] if c not in COPY})
        + list(COPY)
    )
    ix = {c: i for i, c in enumerate(alphabet)}
    keys = list(FIELDS)
    domains = list(DOMAINS)
    raster = Rasterizer()
    cells = [""] + sorted(
        set("".join(keys + domains))
        | {c for rows in sets.values() for r in rows for c in r["q"]}
    )
    code = {c: i for i, c in enumerate(cells)}

    def pix(c):
        return np.frombuffer(raster.tile(c) if c else bytes(128), np.uint8)

    # The upstream corpus uses a small number of unsupported ornamental characters.
    # Map an unsupported input cell to blank and record it. Unrenderable training targets are excluded.
    def make(chars):
        pixels = []
        skips = []
        for c in chars:
            try:
                pixels.append(pix(c))
            except ValueError:
                pixels.append(pix(""))
                skips.append(c)
        return np.array(pixels), skips

    images, dropped = make(cells)
    out, missing = make(alphabet)
    if missing:
        alphabet = [c for c in alphabet if c not in missing]
        ix = {c: i for i, c in enumerate(alphabet)}
        out, _ = make(alphabet)
    reps = [""]
    seen = {}
    indices = {}
    keep = [0]
    for i, c in enumerate(alphabet[1:], 1):
        bitmap = bytes(out[i])
        if bitmap not in seen:
            seen[bitmap] = len(reps)
            reps.append(c)
            keep.append(i)
        indices[c] = seen[bitmap]
    out = out[keep]
    alphabet = reps
    ix = indices
    if alphabet[-2:] != list(COPY):
        raise ValueError("Copy markers alias a printable glyph")
    np.save(MODEL / "palette.npy", out)
    dump(MODEL / "readback.json", alphabet)

    def ids(line):
        return [code[c] for c in line] + [0] * (WIDTH - len(line))

    byte = np.zeros((len(cells), 4), np.uint8)
    for c, i in code.items():
        byte[i, : len(c.encode())] = list(c.encode())
    label = np.array([ids(k) for k in keys + domains], np.int32)
    np.savez_compressed(WORK / "bank.npz", tiles=images, bytes=byte, labels=label)
    np.save(MODEL / "label-pixels.npy", images[label])
    digests = {}
    for s, items in sets.items():
        if s == "train":
            items = [r for r in items if not any(c in missing for c in r["masked"])]
        dump(WORK / (s + ".json"), items)
        n = len(items)
        queries = np.array([ids(r["q"]) for r in items], np.int32)
        ds = np.array([domains.index(r["domain"]) for r in items], np.int16)
        ys = np.full((n, max(len(r["masked"]) for r in items) + 1), -1, np.int16)
        for i, r in enumerate(items):
            seq = [ix.get(c, -2) for c in r["masked"]] + [0]
            ys[i, : len(seq)] = seq
        np.savez_compressed(
            WORK / (s + ".npz"),
            question=queries,
            domain=ds,
            answer=ys,
            label=np.array([keys.index(r["key"]) for r in items], np.int16),
            available=np.array([[k in r["card"] for k in keys] for r in items], bool),
        )
        digests[s] = sha(WORK / (s + ".npz"))
        summary[s]["used"] = n
    dump(
        PUB / "corpus.json",
        dict(
            source=REV,
            files=manifests,
            splits=summary,
            archives=digests,
            bank=sha(WORK / "bank.npz"),
            palette=sha(MODEL / "palette.npy"),
            input_unknown=dropped,
            output_unsupported=missing,
            classes=len(alphabet),
            fields=keys,
            domains=domains,
        ),
    )
    from scripts.write_chinese_card_reply import save_layout

    save_layout()
    print(summary, len(alphabet), flush=True)


if __name__ == "__main__":
    main()
