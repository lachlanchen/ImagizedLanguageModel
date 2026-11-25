"""Prepare visual translation contrasts and independent wording and document cohorts."""

from collections import Counter, defaultdict
from functools import lru_cache
import itertools
import re
import numpy as np
from scripts.prepare_ordered_translation import (
    ROOT,
    WORK as OLD,
    MODEL as BASE,
    PUB as PREV,
    Ink,
    read,
    dump,
    sha,
    hashed,
    key,
)
from scripts.evaluate_ordered_translation import CHALLENGE, material
from scripts.evaluate_news_translation import cohort as previous_news

WORK = ROOT / "artifacts/proposition_translation_v1"
PUB = ROOT / "references/evidence/proposition_translation_v1"
MODEL = ROOT / "models/proposition_translation"
NAMES = [
    ("Tom", "汤姆"),
    ("Mary", "玛丽"),
    ("John", "约翰"),
    ("Peter", "彼得"),
    ("Alice", "爱丽丝"),
    ("David", "大卫"),
    ("the teacher", "老师"),
    ("the doctor", "医生"),
    ("the driver", "司机"),
    ("the student", "学生"),
    ("the nurse", "护士"),
    ("the writer", "作家"),
]
VERBS = [
    ("helped", "help", "帮助了", "帮助"),
    ("visited", "visit", "拜访了", "拜访"),
    ("thanked", "thank", "感谢了", "感谢"),
    ("called", "call", "打电话给", "打电话给"),
    ("invited", "invite", "邀请了", "邀请"),
    ("followed", "follow", "跟着", "跟着"),
    ("saw", "see", "看到了", "看到"),
    ("found", "find", "找到了", "找到"),
    ("hugged", "hug", "拥抱了", "拥抱"),
    ("praised", "praise", "表扬了", "表扬"),
]
NUMS = [
    ("three", "三"),
    ("four", "四"),
    ("five", "五"),
    ("six", "六"),
    ("seven", "七"),
    ("eight", "八"),
    ("nine", "九"),
]
OBJECTS = [
    ("books", "本书"),
    ("tickets", "张票"),
    ("letters", "封信"),
    ("apples", "个苹果"),
    ("maps", "张地图"),
    ("pens", "支笔"),
    ("photos", "张照片"),
    ("keys", "把钥匙"),
    ("cups", "个杯子"),
    ("coins", "枚硬币"),
]
ACTIONS = [
    ("bought", "买了"),
    ("found", "找到了"),
    ("sold", "卖了"),
    ("lost", "丢了"),
    ("needed", "需要"),
    ("counted", "数了"),
]

# Independent authored pairs use constructions absent from the factorial grammar.
PROBES = {
    "role": [
        (
            "It was Mary who thanked Tom.",
            "是玛丽感谢了汤姆。",
            "It was Tom who thanked Mary.",
            "是汤姆感谢了玛丽。",
        ),
        (
            "The doctor was invited by the nurse.",
            "医生受到了护士的邀请。",
            "The nurse was invited by the doctor.",
            "护士受到了医生的邀请。",
        ),
        (
            "After lunch, John called Peter.",
            "午饭后，约翰给彼得打了电话。",
            "After lunch, Peter called John.",
            "午饭后，彼得给约翰打了电话。",
        ),
        (
            "The teacher showed the map to the driver.",
            "老师把地图给司机看了。",
            "The driver showed the map to the teacher.",
            "司机把地图给老师看了。",
        ),
        (
            "Alice received a letter from David.",
            "爱丽丝收到了大卫的信。",
            "David received a letter from Alice.",
            "大卫收到了爱丽丝的信。",
        ),
        (
            "Before leaving, the writer helped the student.",
            "离开前，作家帮助了学生。",
            "Before leaving, the student helped the writer.",
            "离开前，学生帮助了作家。",
        ),
    ],
    "polarity": [
        (
            "The nurse has already arrived.",
            "护士已经到了。",
            "The nurse has not arrived yet.",
            "护士还没到。",
        ),
        (
            "I remember the doctor's name.",
            "我记得医生的名字。",
            "I do not remember the doctor's name.",
            "我不记得医生的名字。",
        ),
        (
            "Peter agreed to help.",
            "彼得同意帮忙。",
            "Peter refused to help.",
            "彼得拒绝帮忙。",
        ),
        (
            "The driver can read the map.",
            "司机能看懂地图。",
            "The driver cannot read the map.",
            "司机看不懂地图。",
        ),
        (
            "There was a key in the box.",
            "盒子里有一把钥匙。",
            "There was no key in the box.",
            "盒子里没有钥匙。",
        ),
        (
            "Mary saw the letter before dinner.",
            "玛丽在晚饭前看到了那封信。",
            "Mary did not see the letter before dinner.",
            "玛丽在晚饭前没有看到那封信。",
        ),
    ],
    "quantity": [
        (
            "Only three tickets remained.",
            "只剩下三张票。",
            "Only eight tickets remained.",
            "只剩下八张票。",
        ),
        (
            "Of the books on the table, five are mine.",
            "桌上的书有五本是我的。",
            "Of the books on the table, nine are mine.",
            "桌上的书有九本是我的。",
        ),
        (
            "She paid for four cups but received two.",
            "她付了四个杯子的钱，却只收到两个。",
            "She paid for six cups but received two.",
            "她付了六个杯子的钱，却只收到两个。",
        ),
        (
            "There are seven more photos in the box.",
            "盒子里还有七张照片。",
            "There are three more photos in the box.",
            "盒子里还有三张照片。",
        ),
        (
            "The last train leaves at five.",
            "末班车五点出发。",
            "The last train leaves at nine.",
            "末班车九点出发。",
        ),
        (
            "Two of the eight letters were missing.",
            "八封信中少了两封。",
            "Two of the four letters were missing.",
            "四封信中少了两封。",
        ),
    ],
}


def capitalize(s):
    return s[:1].upper() + s[1:]


def item(kind, a, b, origin):
    return dict(kind=kind, forms=[list(a), list(b)], origin=origin)


def factorial():
    pools = defaultdict(list)
    for i, j in itertools.permutations(range(len(NAMES)), 2):
        a, x = NAMES[i]
        b, y = NAMES[j]
        fold = int(hashed(f"proposition-names/{min(i,j)}/{max(i,j)}")[:4], 16) % 5
        for v, past in enumerate(VERBS):
            vp, base, zp, zb = past

            def sentence(a, x, b, y, neg):
                return (
                    capitalize(f"{a} did not {base} {b}." if neg else f"{a} {vp} {b}."),
                    f"{x}{'没有'+zb if neg else zp}{y}。",
                )

            for neg in (False, True):
                if i < j:
                    pools[fold].append(
                        item(
                            "role",
                            sentence(a, x, b, y, neg),
                            sentence(b, y, a, x, neg),
                            "factorial",
                        )
                    )
            pools[fold].append(
                item(
                    "polarity",
                    sentence(a, x, b, y, False),
                    sentence(a, x, b, y, True),
                    "factorial",
                )
            )
    for i, (a, x) in enumerate(NAMES):
        for j, (obj, zo) in enumerate(OBJECTS):
            fold = int(hashed(f"proposition-quant/{i}/{j}")[:4], 16) % 5
            for v, (verb, zv) in enumerate(ACTIONS):
                for n in range(len(NUMS)):
                    ns = (NUMS[n], NUMS[(n + 1 + (i + j + v) % 5) % 7])
                    variants = [
                        (capitalize(f"{a} {verb} {word} {obj}."), f"{x}{zv}{num}{zo}。")
                        for word, num in ns
                    ]
                    pools[fold].append(item("quantity", *variants, "factorial"))
    return pools


def real_edits(row):
    en, zh = row["text"]
    for m, ma in (("汤姆", "玛丽"), ("湯姆", "瑪麗")):
        if all(
            s.count(v) == 1 for s, v in ((en, "Tom"), (en, "Mary"), (zh, m), (zh, ma))
        ) and not re.search(
            r"\b(his|her|wife|husband|mother|father|he|she)\b", en, re.I
        ):
            swap = lambda s, a, b: s.replace(a, "\0").replace(b, a).replace("\0", b)
            yield item(
                "role",
                (en, zh),
                (swap(en, "Tom", "Mary"), swap(zh, m, ma)),
                "human-edited",
            )
            break
    # Restrict edits to explicit count expressions having one numeral in each side.
    es = re.findall(r"\b(?:three|four|five|six|seven|eight|nine)\b", en)
    zs = re.findall(
        r"(?<![一二两三四五六七八九十百千万])([三四五六七八九])(?=[个個本张張年天人位名次小时小時封支把条條枚周辆輛])",
        zh,
    )
    if len(es) == len(zs) == 1 and (es[0], zs[0]) in NUMS and zh.count(zs[0]) == 1:
        new = NUMS[(NUMS.index((es[0], zs[0])) + 1 + int(hashed(en)[:4], 16) % 6) % 7]
        yield item(
            "quantity",
            (en, zh),
            (re.sub(r"\b" + es[0] + r"\b", new[0], en), zh.replace(zs[0], new[1])),
            "human-edited",
        )
    if (
        re.match(r"^(I|We|You|They|Tom|Mary|He|She) (can|should) ", en)
        and not re.search(r"\b(not|never|only|hardly|barely|no)\b", en)
        and not re.search(r"[不没沒有無无]", zh)
    ):
        options = (
            ("can", "cannot", ("可以", "能")),
            ("should", "should not", ("应该", "應該")),
        )
        for word, neg, mods in options:
            if " " + word + " " not in en:
                continue
            mod = next((a for a in mods if zh.count(a) == 1), None)
            if mod:
                yield item(
                    "polarity",
                    (en, zh),
                    (
                        en.replace(" " + word + " ", " " + neg + " ", 1),
                        zh.replace(mod, "不" + mod, 1),
                    ),
                    "human-edited",
                )
                break


def prepare():
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    MODEL.mkdir(exist_ok=True, parents=True)
    pools = factorial()
    held = []
    for kind in PROBES:
        candidates = sorted(
            (p for p in pools[0] if p["kind"] == kind), key=lambda p: hashed(repr(p))
        )
        held.extend(dict(p, group="combination") for p in candidates[:12])
        held.extend(
            dict(item(kind, (a, b), (c, d), "authored"), group="wording")
            for a, b, c, d in PROBES[kind]
        )
    held.extend(
        dict(item(k, CHALLENGE[i], CHALLENGE[i + 1], "regression"), group="regression")
        for i, k in ((0, "role"), (2, "polarity"), (4, "action"), (6, "quantity"))
    )
    previous = {x["family"] for x in previous_news()} | {
        x["family"] for x in material() if x["sample"]
    }
    natural = []
    for domain, source in (("conversation", "test"), ("news", "news-held")):
        groups = defaultdict(list)
        for r in read(OLD / (source + ".json")):
            if r["family"] not in previous:
                groups[r["family"]].append(r)
        for fid in sorted(groups, key=lambda s: hashed("proposition-holdout/" + s))[
            :16
        ]:
            r = min(groups[fid], key=lambda r: hashed(repr(r["text"])))
            natural.append(
                dict(domain=domain, family=fid, text=r["text"], credit=r.get("credit"))
            )
    heldkeys = [
        {key(f[t]) for a in held for f in a["forms"]}
        | {key(a["text"][t]) for a in natural}
        for t in (0, 1)
    ]
    ink = Ink(np.load(BASE / "palette.npy"))
    inv = [set(x) for x in read(BASE / "inventories.json")]

    @lru_cache(None)
    def encode(s):
        cells = ink.encode(s)
        return cells

    def safe(pair):
        return all(
            0 < len(encode(s)) <= 96
            and not any(c in ink.bad for c in s)
            and set(encode(s)) <= inv[t]
            and key(s) not in heldkeys[t]
            for t, s in enumerate(pair)
        )

    originals = read(OLD / "train.json")
    news = read(OLD / "news-train.json")
    train = [a for f in (1, 2, 3, 4) for a in pools[f]]
    for r in itertools.chain(originals, news):
        train.extend(real_edits(r))
    clean = []
    seen = set()
    for r in train:
        a, b = r["forms"]
        ident = tuple(sorted((tuple(map(key, a)), tuple(map(key, b)))))
        if ident in seen or not all(safe(z) for z in (a, b)):
            continue
        seen.add(ident)
        clean.append(
            dict(
                kind=r["kind"],
                origin=r["origin"],
                cells=[[encode(s) for s in f] for f in (a, b)],
            )
        )
    replay = []
    for source in (originals, news):
        replay.append(
            [
                r["cells"]
                for r in source
                if all(key(s) not in heldkeys[t] for t, s in enumerate(r["text"]))
            ]
        )
    dump(WORK / "contrasts.json", clean)
    dump(WORK / "replay.json", replay)
    dump(WORK / "held.json", dict(pairs=held, natural=natural))
    dump(
        PUB / "design.json",
        dict(
            training=dict(Counter(r["origin"] + "/" + r["kind"] for r in clean)),
            replay=list(map(len, replay)),
            held=dict(Counter(r["group"] + "/" + r["kind"] for r in held)),
            natural=dict(Counter(r["domain"] for r in natural)),
            files={
                k: sha(WORK / (k + ".json")) for k in ("contrasts", "held", "replay")
            },
        ),
    )
    print(read(PUB / "design.json"))


if __name__ == "__main__":
    prepare()
