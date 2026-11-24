"""Build an optical English meaning corpus from public lexical data."""

import csv
import gzip
import os
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen
import shutil
import numpy as np
from scripts.pictured_action_paraphrases import PARAPHRASES
from scripts.prepare_visual_clause_roles import ROOT, read, dump, sha

STUDY = ROOT / "references/evidence/visual_action_meaning_v1"
WORK = Path(
    os.environ.get("ILM_MEANING_WORK", ROOT / "artifacts/visual_action_meaning_v1")
)
RAW = ROOT / "data/raw/visual_action_semantics"
PARENT = ROOT / "references/evidence/pictured_action_lookup_v1"
FILES = {
    "glove50.gz": (
        "https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz",
        "5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24",
    ),
    "wordnet.zip": (
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
        "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
    ),
    "webster.json": (
        "https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/dictionary_compact.json",
        "16b12847a47cc1202e5e40a3a44b9ed2f749d23c0a5f0ad11fcbd4e8c6c3e33d",
    ),
}


def table(name):
    with (STUDY / name).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def clean(s):
    return " ".join(re.findall("[a-z]+", s.lower()))


def sources():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, h) in FILES.items():
        path = RAW / name
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=RAW, delete=False) as f:
                tmp = Path(f.name)
                try:
                    with urlopen(url, timeout=90) as response:
                        shutil.copyfileobj(response, f)
                    f.flush()
                    if sha(tmp) != h:
                        raise ValueError("Downloaded bytes differ for " + name)
                    tmp.replace(path)
                finally:
                    tmp.unlink(missing_ok=True)
        if sha(path) != h:
            raise ValueError("Changed source " + name)


def build(rows):
    WORK.mkdir(parents=True, exist_ok=True)
    sources()
    reg = read(PARENT / "split.json")
    if sha(rows) != reg["train_sha256"]:
        raise ValueError("Wrong Kyoto training split")
    labels = sorted({r["gloss"][0][0] for r in read(rows)})
    held = table("webster.tsv")
    dictionary = read(RAW / "webster.json")
    assert all(r["description"] in dictionary[r["action"]].lower() for r in held)
    forbidden = {clean(r["description"]) for r in held} | {
        clean(v) for v in PARAPHRASES.values()
    }
    reject = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in sorted(forbidden)) + r")\b"
    )
    examples = {v: set() for v in labels}
    wanted = set(labels)
    removed = 0
    with zipfile.ZipFile(RAW / "wordnet.zip") as z:
        for line in z.read("wordnet/data.verb").decode().splitlines():
            if not line or not line[0].isdigit():
                continue
            data, explanation = line.split(" | ", 1)
            meta = data.split()
            lemmas = [
                clean(x.replace("_", " "))
                for x in meta[4 : 4 + 2 * int(meta[3], 16) : 2]
            ]
            definition = clean(explanation.split('; "')[0].strip())
            for phrase in [*lemmas, definition]:
                wanted.update(phrase.split())
            for v in set(lemmas) & examples.keys():
                for phrase in set(lemmas + [definition]):
                    if not phrase or len(phrase) > 110 or phrase == v:
                        continue
                    if reject.search(phrase):
                        removed += 1
                        continue
                    examples[v].add(phrase)
    names = []
    vectors = []
    alphabetic = 0
    with gzip.open(RAW / "glove50.gz", "rt") as f:
        next(f)
        for line in f:
            name, _, value = line.partition(" ")
            if not re.fullmatch("[a-z]+", name):
                continue
            alphabetic += 1
            if alphabetic <= 24000 or name in wanted:
                names.append(name)
                vectors.append(np.fromstring(value, sep=" ", dtype=np.float32))
    x = np.stack(vectors)
    xc = x.astype("float64") - x.mean(0, dtype="float64")
    lam, basis = np.linalg.eigh(xc.T @ xc / len(xc))
    target = (xc @ basis) / np.sqrt(lam.clip(0.0001))
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    np.save(WORK / "targets.npy", target.astype("float32"), allow_pickle=False)
    dump(WORK / "words.json", names)
    examples = {k: sorted(v) for k, v in examples.items()}
    dump(WORK / "descriptions.json", examples)
    vocabulary = set(names)
    result = dict(
        training_rows=reg["training"],
        training_sha256=sha(rows),
        words=len(names),
        meaning_sha256=sha(WORK / "targets.npy"),
        vocab_sha256=sha(WORK / "words.json"),
        dictionary_sha256=sha(WORK / "descriptions.json"),
        action_labels=len(labels),
        descriptions=sum(map(len, examples.values())),
        actions_with_descriptions=sum(bool(v) for v in examples.values()),
        removed=removed,
        held_complete_phrase_collisions=sum(
            bool(reject.search(s)) for v in examples.values() for s in v
        ),
        held_webster_singletons_in_glove=[
            r["description"] for r in held if r["description"] in names
        ],
        held_webster_singletons_in_kyoto=[
            r["description"] for r in held if r["description"] in labels
        ],
        held_description_word_coverage=[
            sum(w in vocabulary for w in set(clean(r["description"]).split()))
            for r in held
        ],
        sources={k: v[1] for k, v in FILES.items()},
    )
    dump(WORK / "preparation.json", result)
    print("prepared", len(names), result["descriptions"], flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True, type=Path)
    build(p.parse_args().rows)
