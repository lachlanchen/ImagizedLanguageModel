"""Build physical train/test windows for formerly excluded pronoun senses."""

from collections import Counter, defaultdict
from functools import lru_cache
import random
import re

from scripts.prepare_asked_gloss import ROOT, PRIOR, HELD, SOURCES, dump, read, sha
from scripts.train_typeface_writer import Face, FACES

WORK = ROOT / "artifacts/pronoun_writer_v1"
PUBLIC = ROOT / "references/evidence/pronoun_writer_v1"
CANONICAL = {"[1PRON]": "me", "[2PRON]": "you", "[3PRON]": "it"}
SEED = 167031


def window(page, i):
    return (" " * 8 + page + " " * 8)[i : i + 17]


def main():
    WORK.mkdir(exist_ok=True)
    face = Face(FACES[0])
    physical = {}

    @lru_cache(None)
    def cell(c):
        return physical.setdefault(face.cell(c), len(physical))

    def key(w):
        return tuple(map(cell, w))

    base = read(ROOT / "references/evidence/visual_word_senses_20261124/split.json")
    assert sha(PRIOR / "train.json") == base["data"]["train"]["meta"]
    prior = read(PRIOR / "train.json")
    prior_keys = {
        key(left.rjust(8) + c + right.ljust(8)) for _, _, _, left, c, right, _ in prior
    }
    old = read(ROOT / "references/evidence/everyday_gloss_v1/positions.json")
    reserved = {key(window(r["page"], i)) for r in old for i in range(len(r["page"]))}
    records = defaultdict(dict)
    for part, digest in SOURCES.items():
        path = ROOT / f"data/raw/ud_classical_kyoto_r216/lzh_kyoto-ud-{part}.conllu"
        assert sha(path) == digest
        for block in path.read_text().strip().split("\n\n"):
            sid = re.search(r"(?m)^# sent_id = (.+)$", block)
            if not sid:
                continue
            book = sid[1].split("_")[0]
            split = (
                "train"
                if part == "train" and book not in HELD
                else "held_book"
                if book in HELD
                else "held_section"
            )
            tokens = [
                l.split("\t") for l in block.splitlines() if l.split("\t")[0].isdigit()
            ]
            page = "".join(t[1] for t in tokens)
            i = 0
            for t in tokens:
                w = window(page, i)
                i += len(t[1])
                g = re.search(r"(?:^|\|)Gloss=([^|]+)", t[-1])
                if (
                    len(t[1]) != 1
                    or not g
                    or not all(
                        c == " "
                        or ord(c) in face.coverage
                        and "\u3400" <= c <= "\u9fff"
                        for c in w
                    )
                ):
                    continue
                if g[1] in CANONICAL and t[3:4] == ["PRON"] and t[7] == "obj":
                    label, kind = CANONICAL[g[1]], "object_person"
                elif g[1] == "go" and t[1] == "之" and t[3] == "VERB":
                    label, kind = "go", "motion"
                else:
                    continue
                k = key(w)
                records[k].setdefault(
                    (split, label, kind), [w, label, kind, book, sid[1]]
                )
    train, test = [], []
    for k, group in records.items():
        if len(group) != 1:
            continue
        (split, _, _), row = next(iter(group.items()))
        if split == "train" and k not in reserved:
            train.append(row)
        elif split != "train" and k not in prior_keys:
            test.append([*row, split])
    heads = {r[0][8] for r in train}
    train_keys = {key(r[0]) for r in train}
    test_keys = {key(r[0]) for r in test}
    anchors = []
    for book, sid, _, left, glyph, right, word in prior:
        w = left.rjust(8) + glyph + right.ljust(8)
        k = key(w)
        if (
            book in HELD
            or k in reserved
            or k in train_keys
            or k in test_keys
            or k in records
        ):
            continue
        anchors.append([w, word, "lexical", book, sid])
    rng = random.Random(SEED)
    ordinary = [r for r in anchors if r[0][8] not in heads]
    train += [r for r in anchors if r[0][8] in heads] + rng.sample(
        ordinary, min(8192, len(ordinary))
    )
    assert not {key(r[0]) for r in train} & (test_keys | reserved)
    dump(WORK / "train.json", train)
    dump(PUBLIC / "held.json", test)
    dump(
        PUBLIC / "split.json",
        dict(
            train_sha256=sha(WORK / "train.json"),
            test_sha256=sha(PUBLIC / "held.json"),
            train=dict(Counter(r[2] for r in train)),
            held=dict(Counter(r[-1] + "/" + r[2] for r in test)),
            heads=sorted(heads),
            sources=SOURCES,
            old_writer=base["data"]["train"]["meta"],
        ),
    )
    print(read(PUBLIC / "split.json")["train"])


if __name__ == "__main__":
    main()
