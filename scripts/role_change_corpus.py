"""Prepare paired passages whose truth changes while the requested roles stay fixed."""

from collections import Counter
import csv
from hashlib import sha256
from itertools import combinations
import json
import random

from scripts.grounded_proposition_corpus import (
    ARCHIVE,
    ARCHIVE_HASH,
    LANG,
    admissible,
    corpora,
    exchange,
)
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import FACES, Face

SEED = 242110
EN = {
    "导演": "Who directed it?",
    "主演": "Who starred in it?",
    "编剧": "Who wrote the screenplay?",
    "制片人": "Who produced it?",
    "作词": "Who wrote the lyrics?",
    "作曲": "Who composed the music?",
    "歌手": "Who sang it?",
    "作者": "Who wrote the original work?",
}
SURNAMES = "林周许温韩张陈孙吴郑胡陆顾朱唐岳谢苏程宋方杜江罗马"
GIVEN = "明竹远溪山青雪雨文礼和宁芳亭清阳芸松秋舟"


def number(s):
    return int.from_bytes(sha256(s.encode()).digest()[:8], "big")


def duie(split, limit, blocked):
    face = Face(FACES[0])
    source = corpora(split)
    random.Random(SEED + (split == "dev")).shuffle(source)
    rows = []
    titles = set()
    pages = set()
    for idx, p, fs in source:
        if p in pages or any(
            ord(c) not in face.coverage and not c.isspace() for c in p
        ):
            continue
        for a, b in combinations(fs, 2):
            title, r, x = a
            _, s, y = b
            if (
                title != b[0]
                or title in titles
                or title in blocked
                or r == s
                or r not in EN
                or s not in EN
            ):
                continue
            if not admissible(p, (a, b)):
                continue
            roles = [LANG[r][0], LANG[s][0]]
            zh = [f"《{title}》{LANG[k][1+idx%2]}？" for k in (r, s)]
            rows.append(
                dict(
                    key=f"duie-{split}-{idx}",
                    kind="duie",
                    title=title,
                    pages=[p, exchange(p, x, y)],
                    names=[x, y],
                    actions=roles,
                    qs=dict(zh=zh, en=[EN[r], EN[s]]),
                    style="credit",
                )
            )
            pages.add(p)
            titles.add(title)
            break
        if len(rows) == limit:
            break
    if len(rows) != limit:
        raise ValueError(f"Insufficient eligible pages for {split}: {len(rows)}")
    return rows


def fresh(path=None, prefix="fresh"):
    path = path or PUBLIC / "fresh.tsv"
    rows = []
    with path.open() as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            rows.append(
                dict(
                    key=f"{prefix}-{i:02}",
                    kind=r["kind"],
                    pages=[r["page_zero"], r["page_one"]],
                    names=[r["person_a"], r["person_b"]],
                    actions=[r["action_a"], r["action_b"]],
                    qs={
                        l: [r["question_a_" + l], r["question_b_" + l]]
                        for l in ("zh", "en")
                    },
                    style="active",
                )
            )
    return rows


def fictional(key, pair, names, kind):
    rng = random.Random(number(key + kind))
    x, y = names
    a, b = [p[0] for p in pair]
    right_first = rng.randrange(2)
    first_assignment = rng.randrange(2)

    def claims(world):
        va, vb = (a, b) if world == 0 else (b, a)
        parts = [x + va, y + vb]
        if right_first:
            parts.reverse()
        return "，".join(parts)

    layout = rng.randrange(3)
    pages = []
    for world in (0, 1):
        kept = world == first_assignment
        if kind == "role":
            p = (
                "当天的记录写着：" + claims(world) + "。这份记录由两人签字。",
                "记者记下了两件事：" + claims(world) + "。后来照片也印证了这些细节。",
                "在场的人回忆，" + claims(world) + "。这就是当时的分工。",
            )[layout]
        elif kind == "time":
            start = claims(first_assignment)
            p = (
                "头一天，"
                + start
                + "。第二天，两人"
                + ("继续原来的差事。" if kept else "互换了差事。"),
                "一开始是这样安排的："
                + start
                + "。到了最后，二人的分工"
                + ("没有改变。" if kept else "正好反过来。"),
                "原计划让"
                + start
                + "。实际执行时，二人"
                + ("照原计划办。" if kept else "对调了工作。"),
            )[layout]
        else:
            claim = claims(first_assignment)
            p = (
                "起初传出的消息是："
                + claim
                + "。核对结果表明，传言"
                + ("完全属实。" if kept else "把两个人弄反了。"),
                "有人说，"
                + claim
                + "。当事人"
                + (
                    "证实这个说法准确无误。"
                    if kept
                    else "否认这一说法，指出双方正好说反了。"
                ),
                "初稿说"
                + claim
                + "。定稿时，编辑"
                + (
                    "保留了这两个人名的位置。"
                    if kept
                    else "把这两个人名互相交换了位置。"
                ),
            )[layout]
        pages.append(p)
    zhprefix = {
        "role": "当时",
        "time": ("第二天", "到最后", "实际执行时")[layout],
        "negation": "实际上",
    }[kind]
    enprefix = {
        "role": "in the account",
        "time": "in the final arrangement",
        "negation": "in fact",
    }[kind]
    return dict(
        key=key + "-" + kind,
        kind=kind,
        pages=pages,
        names=names,
        actions=[a, b],
        qs=dict(
            zh=[f"{zhprefix}谁{v}？" for v in (a, b)],
            en=[f"Who {e} {enprefix}?" for z, e in pair],
        ),
        style=("active", "cleft")[rng.randrange(2)],
    )


def prepare():
    if sha(ARCHIVE) != ARCHIVE_HASH:
        raise ValueError("DuIE archive changed")
    held = duie("dev", 128, set())
    tr = duie("train", 1024, {c["title"] for c in held})
    new = fresh()
    nms = {n for c in new for n in c["names"]}
    teacher = WORK / "actions.jsonl"
    if sha(teacher) != json.loads((PUBLIC / "teacher.json").read_text())["sha256"]:
        raise ValueError("Action file changed")
    names = [s + g for s in SURNAMES for g in GIVEN if s + g not in nms]
    seen = set()
    for line in teacher.read_text().splitlines():
        v = json.loads(line)
        if not v["pair"]:
            continue
        acts = tuple(z for z, e in v["pair"])
        if acts in seen:
            continue
        seen.add(acts)
        for kind in ("role", "time", "negation"):
            rng = random.Random(number(v["key"] + kind))
            ns = rng.sample(names, 2)
            if any(n in a for n in ns for a in acts):
                continue
            tr.append(fictional(v["key"], v["pair"], ns, kind))
    groups = dict(train=tr, duie=held, fresh=new)
    used = set()
    for split, rows in groups.items():
        pages = {p for c in rows for p in c["pages"]}
        if len(pages) != 2 * len(rows) or used & pages:
            raise ValueError("Repeated input passage: " + split)
        used.update(pages)
    path = WORK / "scenes.json"
    record(path, groups)
    record(
        PUBLIC / "corpus.json",
        dict(
            sha256=sha(path),
            duie_sha=ARCHIVE_HASH,
            teacher_sha=sha(teacher),
            fresh_sha=sha(PUBLIC / "fresh.tsv"),
            counts={k: dict(Counter(c["kind"] for c in v)) for k, v in groups.items()},
            overlap_training_dev_titles=len(
                {c["title"] for c in tr if "title" in c} & {c["title"] for c in held}
            ),
        ),
    )


def load():
    p = WORK / "scenes.json"
    if sha(p) != json.loads((PUBLIC / "corpus.json").read_text())["sha256"]:
        raise ValueError("Scenes changed")
    return json.loads(p.read_text())


def target(c, world, rev):
    names = c["names"][::-1] if world else c["names"]
    if c["style"] == "credit":
        clauses = [
            verb + "是" + name for verb, name in zip(c["actions"], names, strict=True)
        ]
    elif c["style"] == "cleft":
        clauses = [
            verb + "的是" + name for verb, name in zip(c["actions"], names, strict=True)
        ]
    else:
        clauses = [name + verb for verb, name in zip(c["actions"], names, strict=True)]
    return "，".join(clauses[::-1] if rev else clauses) + "。"


if __name__ == "__main__":
    prepare()
