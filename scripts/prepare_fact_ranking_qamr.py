"""Recover source passages for the human QAMR statements in QA2D."""

from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from scripts.prepare_region_writer import ROOT, example, phrase_key, regions, sha
from scripts.prepare_visual_statements import collect, detok

REVISION = "956b0eb2d5857f4aafa1af257ba06fa6cb683b99"
SOURCES = {
    "wiki-sentences.tsv": "2e90fd8d8014ec2ade69148ff2187c26684366dbd3d6f8ad6bb6bf1ba1f04ea9",
    "filtered/train.tsv": "24239718e759523a7d1138583361ed35180177ab0b04f63b5911c2bf6a04ffd9",
    "filtered/dev.tsv": "d507f22b1348324ad3d5c9208308e49e91d409222c080eb6d224a8ecab4bfb93",
    "filtered/test.tsv": "c065e2e28551ae2eb94994c928de18cfb6ae1425ec3ed975e45607265f54eff4",
}


def source(name):
    p = ROOT / "data/raw/qamr" / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=p.parent, delete=False) as tmp:
            new = Path(tmp.name)
        try:
            url = f"https://raw.githubusercontent.com/uwnlp/qamr/{REVISION}/data/{name}"
            with urlopen(url, timeout=45) as r:
                new.write_bytes(r.read())
            if sha(new) != SOURCES[name]:
                raise ValueError("Changed QAMR source " + name)
            new.replace(p)
        finally:
            new.unlink(missing_ok=True)
    if sha(p) != SOURCES[name]:
        raise ValueError("QAMR checksum mismatch")
    return p


def statements_from_qamr():
    sentences = {}
    paragraphs = defaultdict(list)
    for line in source("wiki-sentences.tsv").read_text().splitlines():
        key, words = line.split("\t", 1)
        sentences[key] = detok(words)
        article, paragraph, index = key.rsplit(":", 2)
        paragraphs[article, paragraph].append((int(index), key))
    lookup = defaultdict(set)
    for split in ("train", "dev", "test"):
        for line in source(f"filtered/{split}.tsv").read_text().splitlines():
            fields = line.split("\t")
            lookup[phrase_key(fields[5])].add(fields[0])
    for split in ("train", "dev"):
        for r in collect(split):
            if r["dataset"] != "QAMR":
                continue
            q, fact, answer = [
                detok(r[k]) for k in ("question", "answer", "turker_answer")
            ]
            matches = [
                k
                for k in lookup[phrase_key(q)]
                if k in sentences and fact and fact in sentences[k]
            ]
            if len(matches) != 1:
                continue
            article, para, _ = matches[0].rsplit(":", 2)
            parts = sorted(paragraphs[article, para])
            blocks = [b for _, key in parts for b in regions(sentences[key])]
            valid = [j for j, b in enumerate(blocks) if fact in b]
            if valid:
                yield example(
                    "statements", article, q, blocks, valid, [answer], {"answer": fact}
                )
