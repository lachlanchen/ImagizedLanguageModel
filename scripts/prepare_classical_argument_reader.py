"""Build visual role pairs from the annotated Classical Chinese Kyoto treebank."""

from collections import Counter, defaultdict
from itertools import combinations
import csv
import re
import unicodedata
from urllib.request import urlopen

from scripts.prepare_role_question_binding import (
    reserve as shishuo,
    digest,
    read,
    swap,
    WEIGHTS,
)
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_typeface_writer import Face, FACES

PUBLIC = ROOT / "references/evidence/classical_argument_20261124"
WORK = ROOT / "artifacts/classical_argument_reader_v1"
UD = ROOT / "data/raw/ud_classical_kyoto_r216/lzh_kyoto-ud-train.conllu"
UD_SHA = "dde87435a4590bde8346cc6b792ff0c6dcd301f876815143187bd9b2c839e02b"
ZUO = ROOT.parent / "ZhJpBook/books/zuozhuan/markdown/wenyan.md"


def normalized(s):
    return "".join(c for c in unicodedata.normalize("NFKC", s) if c.isalnum())


def reserved():
    source = ZUO.read_text()
    assert (
        sha(ZUO) == "ca86d6e12ee171f17fa4a268231b1a14bcdad91fbb5fef6dbc3888bdcc61f868"
    )
    cases = []
    with (PUBLIC / "zuo.tsv").open() as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            p, a, b = r["page"], r["a"], r["b"]
            assert source.count(p) == 1 and a in p and b in p
            other = swap(p, a, b)
            assert swap(other, a, b) == p
            qs = [
                dict(answer=r[k], en=r["en_" + k], zh=r["zh_" + k]) for k in ("a", "b")
            ]
            assert all(n not in q["zh"] for n in (a, b) for q in qs)
            cases.append(
                dict(group="zuo", key=f"zuo-{i + 1:02}", pages=[p, other], qs=qs)
            )
    assert len(cases) == 16
    return cases + [dict(**r, group="shishuo") for r in shishuo()]


def person(head, rows):
    if head[3] != "PROPN" or ",人," not in head[4]:
        return
    ids = {int(head[0])}
    for _ in range(len(rows)):
        more = {
            int(r[0])
            for r in rows
            if int(r[6]) in ids and r[7] == "flat" and r[3] in ("PROPN", "NOUN")
        }
        if more <= ids:
            break
        ids.update(more)
    if len(ids) != max(ids) - min(ids) + 1:
        return
    word = "".join(rows[i - 1][1] for i in sorted(ids))
    if 0 < len(word) <= 8:
        return word


def prepare():
    WORK.mkdir(exist_ok=True)
    if not UD.exists():
        base = "https://raw.githubusercontent.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/r2.16/"
        UD.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(base + UD.name, timeout=60) as response:
            tmp = UD.with_suffix(".part")
            tmp.write_bytes(response.read())
        assert sha(tmp) == UD_SHA
        tmp.replace(UD)
    assert sha(UD) == UD_SHA
    cases = reserved()
    held = {normalized(p) for r in cases for p in r["pages"]}
    face = Face(FACES[0])

    def permitted(p):
        return all(c.isspace() or ord(c) in face.coverage for c in p)

    assert all(
        permitted(v)
        for r in cases
        for v in r["pages"] + [q[l] for q in r["qs"] for l in ("zh", "en")]
    )
    with (PUBLIC / "predicates.tsv").open() as f:
        wording = {r["verb"]: r for r in csv.DictReader(f, delimiter="\t")}
    facts = []
    rows_out = []
    joined_used = set()
    seen = set()

    def make(x, y, kind):
        a, b = x["answer"], y["answer"]
        if a in b or b in a or any(x[l] == y[l] for l in ("en", "zh")):
            return False
        if any(n in q["zh"] for n in (a, b) for q in (x, y)):
            return False
        if kind == "single":
            p = x["clause"]
        else:
            if a in y["clause"] or b in x["clause"] or x["verb"] == y["verb"]:
                return False
            order = [x, y] if int(digest(x["id"] + y["id"])[0], 16) % 2 else [y, x]
            p = "。".join(q["clause"] for q in order) + "。"
        if p in seen or not permitted(p):
            return False
        p2 = swap(p, a, b)
        if p2 in seen or any(h in normalized(v) for v in (p, p2) for h in held):
            return False
        seen.update((p, p2))
        qs = [
            {k: q[k] for k in ("zh", "en", "answer", "id", "role", "verb")}
            for q in (x, y)
        ]
        rows_out.append(dict(kind=kind, pages=[p, p2], qs=qs))
        return True

    for block in UD.read_text().split("\n\n"):
        lines = block.splitlines()
        rows = [l.split("\t") for l in lines if l[:1].isdigit()]
        if not rows or not all(
            len(r) == 10 and r[0].isdigit() and r[6].isdigit() for r in rows
        ):
            continue
        if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)):
            continue
        clause = "".join(r[1] for r in rows)
        if not 3 <= len(clause) <= 160 or not permitted(clause):
            continue
        sid = next(l[12:] for l in lines if l.startswith("# sent_id = "))
        local = []
        for r in rows:
            role = {"nsubj": "subject", "obj": "object", "iobj": "object"}.get(r[7])
            if not role or r[6] == "0":
                continue
            verb = rows[int(r[6]) - 1]
            w = wording.get(verb[1])
            if not w or verb[3] != "VERB" or not w[role + "_zh"]:
                continue
            n = person(r, rows)
            if not n or clause.count(n) != 1:
                continue
            lab = dict(
                id=sid + "/" + r[0],
                clause=clause,
                verb=verb[1],
                role=role,
                head=verb[0],
                answer=n,
                zh=w[role + "_zh"],
                en=w[role + "_en"],
            )
            local.append(lab)
            facts.append(lab)
        for x, y in combinations(local, 2):
            if (
                x["head"] == y["head"]
                and x["role"] != y["role"]
                and make(x, y, "single")
            ):
                break
    facts.sort(key=lambda r: digest(r["id"]))
    # Pair every fact at most once for the joined population. Match the same
    # grammatical role where possible so identifying the role alone is insufficient.
    for i, x in enumerate(facts):
        if x["id"] in joined_used:
            continue
        for y in facts[i + 1 : i + 100]:
            if (
                y["id"] in joined_used
                or x["id"].split("/")[0] == y["id"].split("/")[0]
                or x["role"] != y["role"]
            ):
                continue
            if make(x, y, "joined"):
                joined_used.update((x["id"], y["id"]))
                break
    assert len(rows_out) > 500
    record(WORK / "learn.json", rows_out)
    record(WORK / "reserved.json", cases)
    record(
        PUBLIC / "corpus.json",
        dict(
            treebank=UD_SHA,
            zuo_source=sha(ZUO),
            question_wordings=sha(PUBLIC / "predicates.tsv"),
            source_facts=len(facts),
            kinds=dict(Counter(r["kind"] for r in rows_out)),
            verb_count=len({q["verb"] for r in rows_out for q in r["qs"]}),
            learn_sha=sha(WORK / "learn.json"),
            test_sha=sha(WORK / "reserved.json"),
            initial=sha(WEIGHTS),
            works=dict(Counter(r["group"] for r in cases)),
        ),
    )


if __name__ == "__main__":
    prepare()
