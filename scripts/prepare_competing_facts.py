"""Reserve new passage families and index all exact occurrences of trainable facts."""

from collections import defaultdict
import numpy as np
from scripts.prepare_paired_evidence import (
    ROOT,
    WORK as PAIRED,
    source,
    hash_text,
)
from scripts.prepare_recurrent_evidence import (
    read,
    dump,
    sha,
    fragments,
    prepared,
    record_corpus,
)
from scripts.evaluate_visual_cursor import normalize

WORK = ROOT / "artifacts/competing_facts_v1"
PUB = ROOT / "references/evidence/competing_facts_v1"
MODEL = ROOT / "models/competing_facts"
KINDS = ("squad", "cmrc", "coqa", "drcd")


def train_groups():
    with np.load(PAIRED / "corpus.npz") as z:
        po, yo, p, y = [z[k] for k in ("po", "yo", "p", "y")]
        labels = z["labels"]
        groups = read(PAIRED / "groups.json")
        out = defaultdict(list)
        for k in KINDS:
            for group in groups[k]:
                doc = int(labels[group[0], 0])
                page = p[po[doc] : po[doc + 1]]
                facts = {}
                for i in group:
                    ans = y[yo[i] : yo[i + 1] - 1]
                    if not 1 <= len(ans) <= 48:
                        continue
                    key = tuple(map(int, ans))
                    if key not in facts:
                        starts = [
                            int(j)
                            for j in np.flatnonzero(page == ans[0])
                            if np.array_equal(page[j : j + len(ans)], ans)
                        ]
                        facts[key] = [starts, []]
                    if facts[key][0]:
                        facts[key][1].append(i)
                facts = [
                    dict(y=yv, at=starts, rows=rows)
                    for yv, (starts, rows) in facts.items()
                    if rows
                ]
                # Mark the permissible alternatives once; indexing is by physical glyphs.
                keys = ["".join(map(chr, f["y"])) for f in facts]
                for j, f in enumerate(facts):
                    f["rivals"] = [
                        i
                        for i, key in enumerate(keys)
                        if key not in keys[j] and keys[j] not in key
                    ]
                if any(f["rivals"] for f in facts):
                    out[k].append(dict(doc=doc, facts=facts))
    return out


def reserve(salt="competing-facts-v1", pages_per_corpus=64, exclude_files=()):
    blocked = set()
    known = set()
    used_queries = set()
    for file in (
        "paired_evidence_v1/held.json",
        "visual_cursor_v1/held.json",
        "visual_statements_v1/held.json",
        "visual_statements_v1/fresh.json",
        "recurrent_evidence_v1/held.json",
        *exclude_files,
    ):
        for row in read(ROOT / "artifacts" / file):
            if "doc" in row:
                known.add(row["doc"])
            if row.get("source"):
                blocked.update(fragments(row["source"], 24))
            if row.get("question"):
                used_queries.add(normalize(row["question"]))
            for q in row.get("rows", []):
                used_queries.add(normalize(q["question"]))
    for k in KINDS:
        for text, _ in source(k, 0):
            known.add(hash_text(text))
            blocked.update(fragments(text[:600], 32))
    held = []
    for k in KINDS:
        count = 0
        for text, qs in sorted(
            source(k, 1), key=lambda r: hash_text(salt + " " + r[0])
        ):
            doc = hash_text(text)
            p = text[:600]
            if doc in known or fragments(p, 1) & blocked:
                continue
            sel = []
            claimed = set()
            for question, history, refs, _ in qs:
                visible = (question + (" Earlier: " + history if history else ""))[:160]
                answers = {normalize(v) for v in refs if normalize(v)}
                if (
                    not visible
                    or not answers
                    or answers & claimed
                    or normalize(question) in used_queries
                    or any(question == u["question"] for u in sel)
                ):
                    continue
                sel.append(dict(question=question, query=visible, refs=refs))
                claimed.update(answers)
                if len(sel) == 2:
                    break
            if len(sel) != 2:
                continue
            held.append(dict(kind=k, doc=doc, source=p, rows=sel))
            known.add(doc)
            used_queries.update(normalize(r["question"]) for r in sel)
            blocked.update(fragments(p, 24))
            count += 1
            if count == pages_per_corpus:
                break
        if count != pages_per_corpus:
            raise ValueError((k, count))
    return held


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    receipt = PUB / "corpus.json"
    if prepared(receipt, WORK):
        return
    groups = train_groups()
    held = reserve()
    dump(WORK / "train.json", groups)
    dump(WORK / "held.json", held)
    record_corpus(
        receipt,
        dict(
            training_pages={k: len(v) for k, v in groups.items()},
            parents=sha(PAIRED / "corpus.npz"),
            held={k: [r["doc"] for r in held if r["kind"] == k] for k in KINDS},
            files={f: sha(WORK / f) for f in ("held.json", "train.json")},
        ),
    )
    print({k: len(v) for k, v in groups.items()}, len(held), flush=True)


if __name__ == "__main__":
    main()
