"""Find naturally paired event clauses and disjoint role reversals in the archived corpus."""

from collections import Counter
from pathlib import Path
import hashlib
import re
import tarfile

from scripts.align_visual_translation import ROOT, BASE, load, sha, write_new
from scripts.prepare_visual_edit_translation import pairs
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_causal_glyph_translator import read_rows, tiles

OUT = ROOT / "artifacts/visual_event_roles_v1"
STATES = "齐楚秦赵魏燕韩吴越鲁卫宋郑晋周陈蜀汉辽唐夏蔡邾莒虞虢"
FIT_VERBS = "伐攻围灭侵击败入朝会取袭助逐"
RESERVED = "救聘如奔"
EVENT = re.compile(
    f"([{STATES}])(?:国|人|师|军|王|侯|公)?([{FIT_VERBS+RESERVED}])([{STATES}])"
)
CLAUSE = re.compile(r"[；。，、：！？;，：“”「」！？\n]+")
OLD = [
    ("pictured_clause_support_v1", "rows", "copy"),
    ("causal_glyph_translation_v1/evaluation", "rows", "copy"),
    ("visual_feedback_recovery_v1/evaluation", "rows", "copy"),
    ("source_grounded_revision_v1/evaluation", "rows", "source"),
    ("paired_gloss_student_v1/evaluation", "roster", "source"),
    ("visible_image_likelihood_v1/evaluation", "roster", "source"),
]


def order(s):
    return hashlib.sha256(s.encode()).hexdigest()


def swap(text, a, b):
    return text.translate(str.maketrans({a: b, b: a}))


def role_rows(s, t):
    modern = [c.strip() for c in CLAUSE.split(t) if c.strip()]
    for x in (v.strip() for v in CLAUSE.split(s)):
        matches = list(EVENT.finditer(x))
        if len(matches) != 1 or len(x) > 32:
            continue
        a, v, b = matches[0].groups()
        if a == b or x.count(a) != 1 or x.count(b) != 1:
            continue
        # A unique clause containing both participants avoids guessing a clause alignment.
        candidates = [
            c
            for c in modern
            if c.count(a) == c.count(b) == 1
            and c.index(a) < c.index(b)
            and len(c) <= 48
        ]
        if len(candidates) == 1:
            yield dict(source=x + "。", target=candidates[0] + "。", a=a, b=b, verb=v)


def prepare():
    import numpy as np

    archive = BASE / "archive.tar.gz"
    assert (
        sha(archive)
        == "dd5c20960f224b6f8cb18a077e546f57f7509c0524d0ab70722d455bcb73c62c"
    )
    OUT.mkdir()
    bank = tiles()
    renderer = Rasterizer()

    def physical(s):
        return hashlib.sha256(b"".join(renderer.tile(c) for c in s)).digest()

    old = set()
    chapters = set()
    parents = read_rows("train")
    training_chapters = {r["group"] for r in parents}
    parent_verbs = Counter()
    for r in parents + read_rows("heldout"):
        chapters.add(r["group"])
        old.add(hashlib.sha256(bank[r["x"]].tobytes()).digest())
    for r in parents:
        for m in EVENT.finditer(r["source"]):
            parent_verbs[m[2]] += 1
    chapters.update(
        r["chapter"]
        for r in load(
            ROOT
            / "references/evidence/block_translation_20261118/calibration_mapping.json"
        )
    )
    for folder, roster, px in OLD:
        d = ROOT / "artifacts" / folder
        chapters.update(r["chapter"] for r in load(d / (roster + ".json")))
        with np.load(d / (px + ".npz")) as f:
            cells = f["packed"]
            o = f["offsets"]
            old.update(
                hashlib.sha256(cells[a:b].tobytes()).digest() for a, b in zip(o, o[1:])
            )
    collected = []
    broad = []
    nchapter = 0
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            p = Path(item.name)
            if len(p.parts) < 4 or p.parts[1] != "双语数据" or p.name != "bitext.txt":
                continue
            ch = "/".join(p.parts[2:-1])
            nchapter += 1
            held = (
                int(order("role-split/" + ch)[:8], 16) % 3 == 0 and ch not in chapters
            )
            for i, (src, tgt) in enumerate(pairs(tar.extractfile(item).read())):
                if held and 5 <= len(src) <= 32 and 3 <= len(tgt) <= 60:
                    broad.append((order(f"role-broad/{ch}/{i}"), ch, i, src, tgt))
                if not EVENT.search(src):
                    continue
                for r in role_rows(src, tgt):
                    r.update(chapter=ch, ordinal=i)
                    if ch in chapters:
                        # Keep only the actual parent training chapters eligible for added training.
                        if ch not in training_chapters:
                            continue
                    r["held"] = held
                    collected.append(r)
    collected.sort(
        key=lambda r: order(f"role-pair/{r['chapter']}/{r['ordinal']}/{r['source']}")
    )
    used = set(old)
    training = []
    evaluation = []
    for held, dest in [(True, evaluation), (False, training)]:
        for r in collected:
            if held != r["held"] or (not held and r["verb"] in RESERVED):
                continue
            a, b = r["a"], r["b"]
            try:
                keys = {physical(r["source"]), physical(swap(r["source"], a, b))}
                physical(r["target"])
                physical(swap(r["target"], a, b))
            except ValueError:
                continue
            if keys & used:
                continue
            used.update(keys)
            dest.append(r)
    event_chapters = {r["chapter"] for r in evaluation}
    test = []
    groups = Counter()
    for _, ch, i, x, y in sorted(broad):
        if ch in event_chapters or groups[ch] >= 8:
            continue
        try:
            k = physical(x)
            physical(y)
        except ValueError:
            continue
        if k in used:
            continue
        used.add(k)
        groups[ch] += 1
        test.append(dict(chapter=ch, ordinal=i, source=x, target=y))
        if len(test) == 1024:
            break
    report = {}
    for name, data in [("train", training), ("roles", evaluation), ("broad", test)]:
        write_new(OUT / (name + ".json"), data)
        report[name] = dict(
            rows=len(data),
            chapters=len({r["chapter"] for r in data}),
            verbs=dict(Counter(r["verb"] for r in data if "verb" in r)),
            sha256=sha(OUT / (name + ".json")),
        )
    write_new(
        OUT / "preparation.json",
        dict(
            splits=report,
            archive=sha(archive),
            archive_chapters=nchapter,
            previous_chapters=len(chapters),
            parent_pattern_counts=parent_verbs,
        ),
    )
    print(report, flush=True)


if __name__ == "__main__":
    prepare()
