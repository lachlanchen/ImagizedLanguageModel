"""Extract pictured pronoun questions from the official GOLEMcoref work splits."""

from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import re

import requests
from scripts.prepare_dialogue_ink import ROOT, dump, sha
from scripts.train_semantic_page import read as read

REV = "4f038383971cc45fe5901dcb18b5b0ab83a67066"
RAW = ROOT / "data/raw/golemcoref" / REV
RUN = Path(os.environ.get("ILM_MENTION_RUN", ROOT / "artifacts/story_mentions_v1"))
PUBLIC = (
    RUN / "scores"
    if "ILM_MENTION_RUN" in os.environ
    else ROOT / "references/evidence/story_mentions_v1"
)
MODEL = (
    RUN / "weights"
    if "ILM_MENTION_RUN" in os.environ
    else ROOT / "models/story_mentions"
)
LANGS = ("chinese", "english")
SPLITS = ("train", "dev", "test")
BLOBS = {
    "chinese": (
        "4866dcca1f1ca1cb72f35f1645e539370c9dc998",
        "dcff48b5437e09d63409603a0c09264ec4a5682a",
        "c9cd1baf07a55da71e05d1109fc27776a5585800",
    ),
    "english": (
        "d2d5c99c29d23f51adcd9fb25f2a54b375ba147a",
        "f868b70b8bc5482d9847be5d6e3c7286ba17e477",
        "bb7df081bc2d05ac5630817e6f218072fb8bd65f",
    ),
}
PRONOUNS = set(
    "我 你 您 他 她 它 我们 你们 他们 她们 咱 咱们 俺 祂 I me my mine myself you your yours yourself he him his himself she her hers herself they them their theirs themselves we us our ours ourselves".lower().split()
)
BLOCK = 128
CELLS = 288
MAX_SPAN = 24
BRACKETS = re.compile(r"\(e\d+\)|\(e\d+|e\d+\)")


def source(lang, split):
    p = RAW / lang / (split + ".conllu")
    if p.exists():
        data = p.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/GOLEM-lab/GOLEMcoref/{REV}/data/gold_annotations/{lang}/conllu/{split}.conllu"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.content
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != BLOBS[lang][SPLITS.index(split)]:
        raise ValueError(f"Changed {lang} {split} source")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return data.decode()


def sentence(block):
    lines = block.splitlines()
    text = next(line[9:] for line in lines if line.startswith("# text = "))
    tokens = []
    pending = defaultdict(list)
    mentions = []
    groups = set()
    offset = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        token, form, _, pos, _, _, _, _, _, misc = line.split("\t")
        tags = dict(p.split("=", 1) for p in misc.split("|") if "=" in p)
        if "SplitAnte" in tags:
            groups.update(x.split("<")[-1] for x in tags["SplitAnte"].split(","))
        if not token.isdigit():
            continue
        start = text.find(form, offset)
        if start < 0 or text[offset:start].strip():
            raise ValueError("Token does not align with the printed sentence")
        offset = start + len(form)
        idx = len(tokens)
        tokens.append((start, offset, pos))
        spec = tags.get("Entity", "")
        actions = BRACKETS.findall(spec)
        if "".join(actions) != spec:
            raise ValueError(f"Unrecognized entity brackets: {spec}")
        for action in actions:
            entity = action.strip("()")
            if action.startswith("("):
                pending[entity].append(idx)
            if action.endswith(")"):
                if not pending[entity]:
                    raise ValueError("Unmatched mention close")
                begin = pending[entity].pop()
                left = tokens[begin][0]
                tagset = {t[2] for t in tokens[begin : idx + 1]}
                name = "PROPN" in tagset or (
                    "X" in tagset and bool(re.search(r"[A-Z]", text[left:offset]))
                )
                name = (
                    name
                    and not tagset.intersection({"PRON", "CCONJ"})
                    and offset - left <= MAX_SPAN
                )
                mentions.append((left, offset, entity, name))
    if text[offset:].strip() or any(pending.values()):
        raise ValueError("Incomplete alignment or mention")
    return text, mentions, groups


def documents(raw):
    doc = None
    texts, mentions, groups = [], [], set()
    position = 0
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        if "# newdoc id = " in block:
            if doc is not None:
                yield doc, " ".join(texts), mentions, groups
            doc = next(
                s[14:] for s in block.splitlines() if s.startswith("# newdoc id = ")
            )
            texts, mentions, groups = [], [], set()
            position = 0
        if doc is None:
            raise ValueError("Missing initial document identifier")
        text, spans, split_groups = sentence(block)
        texts.append(text)
        mentions.extend((a + position, b + position, e, n) for a, b, e, n in spans)
        groups.update(split_groups)
        position += len(text) + 1
    if doc is not None:
        yield doc, " ".join(texts), mentions, groups


def questions(doc, text, mentions, groups, lang):
    from bisect import bisect_left

    names = sorted(
        (a, b, e)
        for a, b, e, is_name in mentions
        if is_name and e not in groups and text[a:b].lower() not in PRONOUNS
    )
    starts = [a for a, b, e in names]
    seen = set()
    for a, b, e, _ in sorted(mentions):
        if e in groups or (a, b) in seen or text[a:b].lower() not in PRONOUNS:
            continue
        seen.add((a, b))
        block = a // BLOCK * BLOCK
        left = max(0, block - BLOCK)
        right = min(len(text), block + BLOCK + 32)
        candidates = [
            (i - left, j - left, owner)
            for i, j, owner in names[bisect_left(starts, left) : bisect_left(starts, a)]
            if j <= a
        ]
        if len({owner for _, _, owner in candidates}) < 2:
            continue
        yield dict(
            doc=doc,
            lang=lang,
            offset=left,
            text=text[left:right].rstrip(),
            query=[a - left, b - left],
            entity=e,
            names=candidates,
        )


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    seen = set()
    evidence = {}
    pages = {}
    for split in SPLITS:
        rows = []
        works = {}
        for lang in LANGS:
            for doc, text, mentions, groups in documents(source(lang, split)):
                digest = hashlib.sha256(text.encode()).hexdigest()
                if (lang, doc) in seen or digest in seen:
                    raise ValueError("Repeated work")
                seen.update(((lang, doc), digest))
                cases = list(questions(doc, text, mentions, groups, lang))
                works[f"{lang}/{doc}"] = dict(
                    cases=len(cases),
                    present=sum(
                        any(e == r["entity"] for _, _, e in r["names"]) for r in cases
                    ),
                )
                rows.extend(cases)
        file = RUN / (split + ".json")
        dump(file, rows)
        current = {hashlib.sha256(r["text"].encode()).hexdigest() for r in rows}
        overlap = {s: len(current & past) for s, past in pages.items()}
        if any(overlap.values()):
            raise ValueError("Repeated pictured passage across work splits")
        pages[split] = current
        evidence[split] = dict(
            sha256=sha(file),
            counts=dict(Counter(r["lang"] for r in rows)),
            works=works,
            overlap=overlap,
        )
    dump(PUBLIC / "corpus.json", dict(revision=REV, blobs=BLOBS, splits=evidence))
    print({s: x["counts"] for s, x in evidence.items()})


if __name__ == "__main__":
    main()
