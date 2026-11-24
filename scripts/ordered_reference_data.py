"""Prepare fixed natural reference probes and fit-only order examples."""

import hashlib
import re
import requests
import numpy as np
from scripts.prepare_story_mentions import ROOT, read, sha
from scripts.check_mention_transfer import crop
from scripts.train_story_cast import partitions, story_rows

FILES = {
    "en": "squad_v1.1/dev-v1.1.json",
    "zh-hant": "drcd/DRCD_test.json",
    "zh-hans": "cmrc2018/cmrc2018_dev.json",
}


# Freeze the source bytes as well as the order of the screened mentions.
SOURCES = {
    "en": (
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    ),
    "zh-hant": (
        "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/b944790de5af02c5fbb7cd9cb1473d27d169eebf/DRCD_test.json",
        "d9de0b4d247a8391ac8daee6c02c2c8272b75562a1e2377b50ae04ecc09a7438",
    ),
    "zh-hans": (
        "https://raw.githubusercontent.com/ymcui/cmrc2018/c0eb1b6ba219847457e6af3180da722bbeb656af/squad-style-data/cmrc2018_dev.json",
        "e9ff74231f05c230c6fa88b84441ee334d97234cbb610991cd94b82db00c7f1f",
    ),
}


def verified_source(lang, path):
    url, digest = SOURCES[lang]
    if not path.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != digest:
            raise ValueError(f"Changed download for {lang}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if sha(path) != digest:
        raise ValueError(f"Changed reference corpus for {lang}")
    return read(path)


def natural():
    labels = read(ROOT / "references/ordered_reference_cases.json")
    out = {}
    screen = {}
    for lang, name in FILES.items():
        path = ROOT / "data/raw" / name
        seen = set()
        hits = []
        for d in verified_source(lang, path)["data"]:
            for p in d["paragraphs"]:
                text = p["context"]
                if text in seen:
                    continue
                seen.add(text)
                for m in re.finditer(
                    r"前者|後者|后者|\b(?:the|The) (former|latter)\b", text
                ):
                    hits.append(
                        (text, m.start(1) if m.lastindex else m.start(), m.end())
                    )
        rows = []
        decisions = []
        for i, (label, (text, a, b)) in enumerate(zip(labels[lang], hits)):
            if label is None:
                decisions.append("no distinct attributable visible pair")
                continue
            x, y, t = label
            options = [(text.rfind(s, 0, a), s) for s in (x, y)]
            if any(i < 0 for i, s in options):
                decision = "candidate absent before mention"
            else:
                decision = crop(f"{lang}-{i}", text, (a, text[a:b]), options, t)
            if isinstance(decision, str):
                decisions.append(decision)
                continue
            rows.append(decision)
            decisions.append("eligible")
        out[lang] = rows
        screen[lang] = dict(
            source_sha256=sha(path), occurrences=len(hits), first_sixteen=decisions
        )
    return out, screen


def teaching():
    rng = np.random.default_rng(7301)
    old = story_rows("train")
    out = []
    seen = set()
    for lang, rows in old.items():
        for r in rows:
            names = list(dict.fromkeys(r["text"][a:b] for a, b in r["cast"]))
            if len(names) < 2:
                continue
            pair = rng.choice(names, size=2, replace=False)
            if any(s in t for s, t in (pair, pair[::-1])):
                continue
            for left, right in (pair, pair[::-1]):
                if lang == "english":
                    intro = rng.choice(
                        [
                            f"{left} and {right} came in.",
                            f"The two entries are {left} and {right}.",
                            f"We considered {left} alongside {right}.",
                        ]
                    )
                    ref = ("former", "latter")
                    frames = [
                        (" The ", " spoke at the meeting."),
                        (" We heard from the ", " yesterday."),
                        (" They described the ", " in detail."),
                    ]
                else:
                    intro = rng.choice(
                        [
                            f"{left}和{right}都到了。",
                            f"这里比较{left}与{right}。",
                            f"这两项是{left}、{right}。",
                        ]
                    )
                    ref = ("前者", rng.choice(["后者", "後者"]))
                    frames = [
                        ("据说", "先到。"),
                        ("其中", "很可靠。"),
                        ("谈起", "，大家都点头。"),
                    ]
                pre, post = frames[rng.integers(len(frames))]
                for target, word in enumerate(ref):
                    lead = intro + pre
                    text = lead + word + post
                    if text in seen:
                        continue
                    seen.add(text)
                    out.append(
                        dict(
                            text=text,
                            query=[len(lead), len(lead) + len(word)],
                            cast=[
                                (intro.index(s), intro.index(s) + len(s))
                                for s in (left, right)
                            ],
                            truth=target,
                        )
                    )
                if lang == "english":
                    pre = intro + " They spoke to the "
                    word = "former"
                    text = pre + word + " manager."
                    if text not in seen:
                        seen.add(text)
                        out.append(
                            dict(
                                text=text,
                                query=[len(pre), len(pre) + len(word)],
                                cast=[
                                    (intro.index(s), intro.index(s) + len(s))
                                    for s in (left, right)
                                ],
                                truth=None,
                            )
                        )
    return out


def ordinary():
    other, _ = partitions()
    return sum(story_rows("train").values(), []) + other["clue"][0] + other["gap"][0]
