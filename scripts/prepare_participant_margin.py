"""Prepare name exchanges from annotated DuIE passages with two competing credits."""

from collections import Counter
from itertools import combinations
import json
import random

from scripts.grounded_proposition_corpus import (
    admissible,
    corpora,
    exchange,
    label,
    query,
)
from scripts.prepare_visual_facts import Face, FONTS, LANG, ROOT, simplify, write_json
from scripts.train_visual_facts import rows
from scripts.train_pointing_causal_writer import sha

OUT = ROOT / "artifacts/participant_margin_v1"
PUBLIC = ROOT / "references/evidence/participant_margin_20261123"
SEED = 231109
VERB = {
    "作词": "作词",
    "作曲": "作曲",
    "歌手": "演唱",
    "编剧": "编剧",
    "导演": "执导",
    "主演": "主演",
    "制片人": "制片",
}
FRESH = {
    "作曲": "谁谱写了旋律",
    "作词": "哪位写下了歌词",
    "导演": "执导者是哪一位",
    "编剧": "剧本由谁执笔",
    "歌手": "哪位歌手演唱",
    "制片人": "制片的是哪位",
    "主演": "哪位演员主演",
    "作者": "谁创作了它",
    "祖籍": "祖上来自哪里",
    "毕业院校": "读的是哪所学校",
    "父亲": "父亲姓甚名谁",
    "母亲": "母亲叫什么名字",
}


def pick(split, count, blocked, rng, face):
    sources = corpora(split)
    rng.shuffle(sources)
    result = []
    used_s, used_p = map(set, blocked)
    for key, p, facts in sources:
        if simplify(p) in used_p or any(
            not c.isspace() and ord(c) not in face.coverage for c in p
        ):
            continue
        for a, b in combinations(facts, 2):
            s, r, o = a
            if s != b[0] or r == b[1] or p.count(s) != 1 or simplify(s) in used_s:
                continue
            if not admissible(p, (a, b)):
                continue
            if set((r, b[1])) - VERB.keys():
                continue
            # Rejected if exchanging different-length names changes substring uniqueness.
            pg = exchange(p, o, b[2])
            if any(pg.count(t) != 1 for t in (s, o, b[2])):
                continue
            result.append((key, p, a, b))
            used_s.add(simplify(s))
            used_p.add(simplify(p))
            break
        if len(result) == count:
            return result
    raise RuntimeError(
        f"Only {len(result)} eligible source pages for {split}, requested {count}"
    )


def make_rows(boards, group, controlled, test):
    result = []
    for i, (key, p, a, b) in enumerate(boards):
        s, r, one = a
        _, u, two = b
        for reverse in range(2 if controlled else 1):
            for swap in range(2):
                names = (two, one) if swap else (one, two)
                if controlled:
                    clauses = [(names[j], VERB[role]) for j, role in enumerate((r, u))]
                    if reverse:
                        clauses.reverse()
                    (x, v), (y, w) = clauses
                    if test:
                        pg = f"登记册注明，《{s}》由{x}{v}，{y}{w}。"
                    elif i % 2:
                        pg = f"《{s}》的制作名单：{v}是{x}，{w}是{y}。"
                    else:
                        pg = f"《{s}》由{x}{v}，{y}{w}。"
                else:
                    pg = exchange(p, one, two) if swap else p
                if any(pg.count(t) != 1 for t in (s, *names)):
                    break
                items = []
                for j, role in enumerate((r, u)):
                    q = (
                        f"据此，{label(s,role)}，{FRESH[role]}？"
                        if test
                        else query(s, role, i % 2)
                    )
                    items.append(
                        dict(
                            group=group,
                            board=str(key),
                            q=q,
                            p=pg,
                            s=s,
                            r=role,
                            o=names[j],
                            rival=names[1 - j],
                            reverse=reverse,
                            swap=swap,
                            job=j,
                        )
                    )
                result.extend(items)
    groups = Counter(x["board"] for x in result)
    return [v for v in result if groups[v["board"]] == (8 if controlled else 4)]


def main():
    rng = random.Random(SEED)
    face = Face((FONTS[0], 2))
    base = rows("training")
    old = rows("held")
    prior = json.loads(
        (ROOT / "artifacts/grounded_proposition_v1/boards.json").read_text()
    )
    subjects = {simplify(x["s"]) for x in base + old}
    pages = {simplify(x["p"]) for x in base + old}
    for boards in prior.values():
        for board in boards:
            subjects.update(simplify(s) for s, _, _ in board["facts"])
            pages.update(simplify(p) for p in board["p"].split("　"))
    dev = pick("dev", 176, (subjects, pages), rng, face)
    test = make_rows(dev[:128], "new_source", False, True) + make_rows(
        dev[128:], "clause_order", True, True
    )
    subjects = {simplify(x["s"]) for x in old + test}
    pages = {simplify(x["p"]) for x in old + test}
    sources = pick("train", 1200, (subjects, pages), rng, face)
    train = make_rows(sources, "paired_source", False, False) + make_rows(
        sources[:600], "paired_clauses", True, False
    )
    seen_subject = {simplify(x["s"]) for x in train + base}
    seen_page = {simplify(x["p"]) for x in train + base}
    assert all(
        simplify(x["s"]) not in seen_subject and simplify(x["p"]) not in seen_page
        for x in test
    )
    human = [i for i, x in enumerate(base) if x["group"] in ("cmrc", "drcd")]
    other = rng.sample(
        [i for i, x in enumerate(base) if x["group"] not in ("cmrc", "drcd")],
        len(train) - len(human),
    )
    replay = human + other
    OUT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, ds in [("paired", train), ("new", test), ("replay", replay)]:
        path = OUT / (name + ".json")
        with path.open("x") as f:
            json.dump(ds, f, ensure_ascii=False)
        hashes[name] = sha(path)
    write_json(
        PUBLIC / "corpus.json",
        dict(
            sha256=hashes,
            counts=dict(Counter(x["group"] for x in train + test)),
            replay=len(replay),
            normalized_subject_and_page_overlap=0,
            base_split="references/evidence/role_binding_20261123/corpus.json",
        ),
    )
    print("records", len(train), len(test), len(replay), flush=True)


if __name__ == "__main__":
    main()
