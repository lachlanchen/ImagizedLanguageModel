"""Pair human questions with evidence from other conversations about the same venue."""

import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from scripts.prepare_chinese_card_reply import (
    RAW,
    DOMAINS,
    FIELDS,
    ROOT,
    value,
    sources,
    sha,
    read,
    dump,
)
from scripts.prepare_chinese_two_fact import double_rows
from scripts.prepare_chinese_card_reply import extract

BASE = Path(os.environ.get("ILM_PROSE_RUN", ROOT / "artifacts/prose_fields_v1"))
PUBLIC = (
    BASE / "scores"
    if os.environ.get("ILM_PROSE_RUN")
    else ROOT / "references/evidence/prose_fields_v1"
)
WEIGHTS = (
    BASE / "weights"
    if os.environ.get("ILM_PROSE_RUN")
    else ROOT / "models/prose_fields"
)
WIDTH = 80
LIMIT = 160


def norm(s):
    return "".join(s.split())


def resolve(msg, domain):
    names = set(msg["sys_state"][domain].get("selectedResults", []))
    names.update(
        v
        for a, d, k, v in msg["dialog_act"]
        if a == "Inform" and d == domain and k == "名称"
    )
    return next(iter(names)) if len(names) == 1 else None


def positions(text, card):
    spans = {}
    for key, val in card.items():
        if not val or list(card.values()).count(val) != 1:
            continue
        for match in re.finditer(re.escape(val), text):
            lo, hi = match.span()
            if val[0].isdigit() and lo and text[lo - 1] in "0123456789.":
                continue
            spans[key] = [lo, hi]
            break
    return spans


def passages(split, catalog, seen):
    with zipfile.ZipFile(RAW / (split + ".json.zip")) as f:
        dialogs = json.load(f.open(split + ".json"))
    found = []
    for did, dialog in dialogs.items():
        for i, msg in enumerate(dialog["messages"]):
            if msg["role"] != "sys":
                continue
            text = msg["content"]
            if not 1 <= len(text) <= LIMIT or norm(text) in seen:
                continue
            acts = msg["dialog_act"]
            domains = {d for a, d, k, v in acts if a not in ("General",)}
            if len(domains) != 1:
                continue
            domain = next(iter(domains))
            if domain not in DOMAINS:
                continue
            name = resolve(msg, domain)
            if not name or name not in catalog[domain]:
                continue
            entry = catalog[domain][name]
            card = {k: value(k, entry.get(k)) for k in FIELDS}
            spans = positions(text, card)
            if not spans:
                continue
            # Exclude entity confusion and an explicitly annotated competing value.
            if any(
                a == "Inform"
                and d == domain
                and k in FIELDS
                and v in text
                and card[k] != v
                for a, d, k, v in acts
            ):
                continue
            found.append(
                dict(
                    dialog=did, turn=i, domain=domain, name=name, text=text, spans=spans
                )
            )
    return found


def joined(rows):
    """Combine two distinct human utterances. Their seam carries no model feature."""
    a, b = rows
    p = a["text"]
    sep = "" if p[-1] in "。，！？；?!." else "。"
    text = p + sep + b["text"]
    if len(text) > LIMIT:
        return None
    spans = {k: [len(p) + len(sep) + x for x in v] for k, v in b["spans"].items()}
    spans.update(a["spans"])
    return dict(
        text=text,
        domain=a["domain"],
        name=a["name"],
        spans=spans,
        pieces=[[r["dialog"], r["turn"]] for r in rows],
    )


def rank(query, source):
    fields = set(source["spans"])
    needed = set(query["keys"])
    return (len(fields - needed), len(fields), -len(source["text"]))


def prepare_pairs(queries, docs, training_text):
    by_name = defaultdict(list)
    for d in docs:
        by_name[d["domain"], d["name"]].append(d)
    out = defaultdict(list)
    for r in queries:
        eligible = [
            p
            for p in by_name[r["domain"], r["name"]]
            if p["dialog"] != r["dialog"] and norm(p["text"]) != norm(r["answer"])
        ]
        needs = set(r["keys"])

        def add(group, items):
            if items:
                page = max(items, key=lambda p: rank(r, p))
                pieces = page.get("pieces") or [[page["dialog"], page["turn"]]]
                out[group].append(
                    dict(
                        q=r["q"],
                        keys=r["keys"],
                        name=r["name"],
                        domain=r["domain"],
                        card=r["card"],
                        answer=r["answer"],
                        dialog=r["dialog"],
                        turn=r["turn"],
                        text=page["text"],
                        spans=page["spans"],
                        pieces=pieces,
                    )
                )

        add("natural", [p for p in eligible if needs <= set(p["spans"])])
        add("missing", [p for p in eligible if not needs <= set(p["spans"])])
        if len(needs) == 2:
            first, second = r["keys"]
            pa = [
                p for p in eligible if first in p["spans"] and second not in p["spans"]
            ]
            pb = [
                p for p in eligible if second in p["spans"] and first not in p["spans"]
            ]
            # Alternate the written evidence order independent of the requested fields.
            flip = (
                int(
                    hashlib.sha256((r["dialog"] + str(r["turn"])).encode()).hexdigest()[
                        -1
                    ],
                    16,
                )
                % 2
            )
            pages = (
                joined((b, a) if flip else (a, b))
                for a in pa
                for b in pb
                if a["dialog"] != b["dialog"]
            )
            add(
                "assembled",
                [p for p in pages if p and norm(p["text"]) not in training_text],
            )
    return out


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    old = read(ROOT / "references/evidence/chinese_reply_v1/corpus.json")
    if any(not (RAW / Path(k).name).exists() for k in old["files"]):
        sources()
    for k, h in old["files"].items():
        if sha(RAW / Path(k).name) != h:
            raise ValueError(k)
    catalog = {d: dict(read(RAW / (f + "_db.json"))) for d, f in DOMAINS.items()}
    seen = set()
    summary = {}
    for split in ("train", "val", "test"):
        docs = passages(split, catalog, seen)
        dump(BASE / (split + "-passages.json"), docs)
        singles, _ = extract(split, catalog)
        queries = [dict(r, keys=[r["key"]]) for r in singles] + double_rows(
            split, catalog
        )
        summary[split] = dict(
            passages=len(docs),
            unique=len({norm(r["text"]) for r in docs}),
            multi=sum(len(r["spans"]) >= 2 for r in docs),
        )
        if split == "train":
            seen = {norm(r["text"]) for r in docs}
            # All answer targets from the older frozen writer are historical exposures.
            prior = {norm(r["answer"]) for r in queries}
            seen |= prior
        else:
            groups = prepare_pairs(queries, docs, seen)
            dump(BASE / (split + "-questions.json"), groups)
            summary[split]["questions"] = {
                k: [sum(len(r["keys"]) == n for r in rs) for n in (1, 2)]
                for k, rs in groups.items()
            }
    dump(
        BASE / "corpus.json",
        dict(
            source=old["source"],
            splits=summary,
            sha256={
                p.name: sha(p)
                for p in sorted(BASE.glob("*-*.json"))
                if p.name.endswith(("-questions.json", "-passages.json"))
            },
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
