"""Build article-disjoint SQuAD splits and character-exact visual span labels."""

from collections import Counter
import hashlib
import json
import re

from scripts.prepare_questioned_events import ROOT, sha, dump

WORK = ROOT / "artifacts/optical_passage_qa_v1"
PUB = ROOT / "references/evidence/optical_passage_qa_v1"
SOURCES = [
    (
        ROOT / "data/raw/visual_semantic_teacher/squad_train.json",
        "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955",
    ),
    (
        ROOT / "data/raw/squad_v1.1/dev-v1.1.json",
        "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    ),
]


def signature(s):
    return hashlib.sha256(s.encode()).hexdigest()


def printable(s):
    return all(32 <= ord(c) <= 126 or c.isspace() for c in s)


def prepare():
    pages, rows = {}, []
    rejected = Counter()
    raw = Counter()
    for file_no, (file, expected) in enumerate(SOURCES):
        if sha(file) != expected:
            raise ValueError("Invalid SQuAD source")
        for article in json.loads(file.read_text())["data"]:
            title = article["title"]
            group = (
                "dev"
                if file_no
                else ("reserved" if int(signature(title), 16) % 17 == 0 else "train")
            )
            for paragraph in article["paragraphs"]:
                text = paragraph["context"]
                words = list(re.finditer(r"\S+", text))
                page = " ".join(w[0] for w in words)
                for qa in paragraph["qas"]:
                    raw[group] += 1
                    question = " ".join(qa["question"].split())
                    if not (
                        printable(text)
                        and printable(question)
                        and 1 <= len(words) <= 192
                        and 1 <= len(question.split()) <= 32
                    ):
                        rejected[group + "_geometry"] += 1
                        continue
                    a = qa["answers"][0]
                    i = a["answer_start"]
                    j = i + len(a["text"])
                    if text[i:j] != a["text"]:
                        raise ValueError(qa["id"])
                    touched = [
                        k for k, w in enumerate(words) if i < w.end() and w.start() < j
                    ]
                    if not touched:
                        rejected[group + "_empty"] += 1
                        continue
                    l, r = touched[0], touched[-1]
                    target = [
                        l,
                        r,
                        max(0, i - words[l].start()),
                        max(0, words[r].end() - j),
                    ]
                    if group == "train" and (r - l >= 16 or max(target[2:]) >= 24):
                        rejected["train_target"] += 1
                        continue
                    key = signature(page)
                    pages[key] = page
                    rows.append(
                        [
                            group,
                            title,
                            key,
                            qa["id"],
                            question,
                            target,
                            [x["text"] for x in qa["answers"]],
                        ]
                    )
    train = {r[2] for r in rows if r[0] == "train"}
    kept = []
    for r in rows:
        if r[0] != "train" and r[2] in train:
            rejected[r[0] + "_duplicate"] += 1
        else:
            kept.append(r)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "corpus.json").write_text(
        json.dumps(dict(pages=pages, rows=kept), ensure_ascii=False)
    )
    groups = ("train", "reserved", "dev")
    result = dict(
        raw=raw,
        excluded=rejected,
        examples=Counter(r[0] for r in kept),
        pages={g: len({r[2] for r in kept if r[0] == g}) for g in groups},
        articles={g: len({r[1] for r in kept if r[0] == g}) for g in groups},
        corpus_sha256=sha(WORK / "corpus.json"),
    )
    dump(PUB / "corpus.json", result)
    print(result)


if __name__ == "__main__":
    prepare()
