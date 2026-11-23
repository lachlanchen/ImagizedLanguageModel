"""Build book-passage supervision from the Chinese and English BiPaR questions."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from tempfile import NamedTemporaryFile

from scripts.evaluate_print_routing import paragraphs
from scripts.interval_teacher_corpus import fingerprints
from scripts.span_reservation import reservation
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import ROOT, FACES, Face

BASE = "https://raw.githubusercontent.com/sharejing/BiPaR/f6f01ee166871bf8a9b47afeb5f418f48bc61815/"
SHA = {
    "Crosslingual_QEN_train.json": "4caf9ff6fdfef3e122f560bc7fa79a30a8a55251350f19992f4628cdf5f24aad",
    "Crosslingual_QEN_valid.json": "4d4d05044a6e4d2815330dfc08a5ff31dd35b80ee0409ed92fe96086f554a29b",
    "Monolingual_ZH_train.json": "0bae5c234d8cd7768b66902c0077c9dbbb7628e1272d483cb41f327169bb2c88",
    "Monolingual_ZH_valid.json": "9fbae513fc9bae94860e2afd289d03e341d4bdf6f41d9b038ee61947ac60abfe",
}
WORK = ROOT / "artifacts/bilingual_novel_v1"
PUBLIC = ROOT / "references/evidence/bilingual_novel_20261124"


def source(kind, split):
    filename = f"{kind}_{split}.json"
    path = ROOT / "data/raw/bipar" / filename
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        sub = {
            "Crosslingual_QEN": "Crosslingual/QEN",
            "Monolingual_ZH": "Monolingual/ZH",
        }[kind]
        with urlopen(BASE + sub + "/" + filename, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != SHA[filename]:
            raise ValueError(f"Unexpected remote content for {filename}")
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
        Path(tmp.name).replace(path)
    if sha(path) != SHA[filename]:
        raise ValueError(f"Source changed: {path}")
    return path


def align(split):
    english = {}
    for p in paragraphs(source("Crosslingual_QEN", split)):
        for q in p["qas"]:
            key = q["id"].removesuffix("_CROSS_QEN")
            if key in english:
                raise ValueError("Duplicate bilingual key")
            english[key] = p["context"], q
    for p in paragraphs(source("Monolingual_ZH", split)):
        items = []
        for q in p["qas"]:
            key = q["id"].removesuffix("_ZH")
            passage, other = english.pop(key)
            if passage != p["context"] or other["answers"] != q["answers"]:
                raise ValueError(
                    "Parallel sources disagree on context or answers: " + key
                )
            items.append((key, q["question"], other["question"], q["answers"]))
        yield p["context"], items
    if english:
        raise ValueError("Unpaired English questions")


def prepare():
    WORK.mkdir(exist_ok=True)
    public_xquad = ROOT / "data/raw/xquad/xquad.zh.json"
    blocked = {
        f for v in paragraphs(public_xquad) for f in fingerprints(v["context"][:416])
    }
    for p in (ROOT / "data/raw/cmrc2018").glob("cmrc2018_*.json"):
        for v in paragraphs(p):
            blocked.update(fingerprints(v["context"][:416]))
    guard = reservation()
    canonical = Face(FACES[0])
    report = {}
    for split in ("valid", "train"):
        seen = set()
        stats = Counter()
        groups = []
        for passage, rows in align(split):
            stats["source_questions"] += len(rows)
            p = passage[:416]
            fp = fingerprints(p)
            if fp & blocked or fp & seen or guard.overlaps(passage):
                stats["blocked_passages"] += 1
                continue
            seen.update(fp)
            cases = []
            for key, zh, en, answers in rows:
                if not zh or not en or len(p) + 2 * max(len(en), len(zh)) + 52 > 1024:
                    stats["input_length"] += 1
                    continue
                refs = []
                for a in answers:
                    text, at = a["text"], a["answer_start"]
                    if (
                        at >= 0
                        and 0 < len(text) <= 48
                        and p[at : at + len(text)] == text
                        and text not in refs
                    ):
                        refs.append(text)
                if not refs:
                    stats["no_visible_annotated_answer"] += 1
                    continue
                if any(
                    not c.isspace() and ord(c) not in canonical.coverage
                    for s in [p, zh, en, *refs]
                    for c in s
                ):
                    stats["missing_font_glyph"] += 1
                    continue
                cases.append(dict(id=key, zh=zh, en=en, refs=refs))
            if not any(
                a["zh"] != b["zh"]
                and a["en"] != b["en"]
                and all(x not in y and y not in x for x in a["refs"] for y in b["refs"])
                for j, a in enumerate(cases)
                for b in cases[j + 1 :]
            ):
                stats["without_distinct_pair"] += 1
                continue
            groups.append(dict(page=p, questions=cases))
        blocked.update(seen)
        path = WORK / (split + ".jsonl")
        content = "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in groups)
        if path.exists() and path.read_text() != content:
            raise ValueError("Prepared data differs")
        if not path.exists():
            path.write_text(content)
        report[split] = dict(
            **stats,
            passages=len(groups),
            parallel_questions=sum(len(v["questions"]) for v in groups),
            sha256=sha(path),
        )
    record(
        PUBLIC / "corpus.json",
        dict(source=BASE, source_sha256=SHA, splits=report, license="CC BY-NC 4.0"),
    )
    print(report)


if __name__ == "__main__":
    prepare()
