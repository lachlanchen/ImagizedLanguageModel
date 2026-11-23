"""Prepare short two-clause training replies to human questions about Chinese novels."""

import hashlib
from itertools import combinations
import json
import re

from scripts.local_translation_teacher import local_teacher
from scripts.mine_stopped_book_replies import BOOK, BOOK_PUBLIC
from scripts.train_bilingual_novel_writer import opposite
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_span_choice import gpu_owner

WORK = ROOT / "artifacts/ordered_book_replies_v1"
PUBLIC = ROOT / "references/evidence/ordered_book_replies_20261124"
MODEL = "qwen3:4b-q8_0"
PAGES = 1024
SYSTEM = """读给定的小说摘录和两组人工问答。为每组写一句独立、简短、主谓齐全的中文陈述句，逐字保留该组给定的答案。不要加入摘录未提供的信息。每句至多22字。不要重复问题或给答案加解释。只输出两行，每行一个陈述句。"""


def sample(split):
    path = BOOK / f"{split}.jsonl"
    manifest = json.loads((BOOK_PUBLIC / "corpus.json").read_text())
    if sha(path) != manifest["splits"][split]["sha256"]:
        raise ValueError("Book corpus changed")
    choices = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for a, b in combinations(row["questions"], 2):
            refs = [a["refs"][0], b["refs"][0]]
            if not opposite(a, b) or max(map(len, refs)) > 10:
                continue
            if not all(re.search(r"[\u3400-\u9fff]", x) for x in refs):
                continue
            if len(row["page"]) + 2 * (len(a["en"]) + len(b["en"])) > 775:
                continue
            choices.append(
                dict(page=row["page"], qs=[a, b], key=a["id"] + "+" + b["id"])
            )
            break
    choices.sort(key=lambda r: hashlib.sha256(r["key"].encode()).digest())
    return choices[:PAGES] if split == "train" else choices[:96]


def clean(lines, refs):
    raw = [
        s.strip().rstrip("。！？.!?") for s in lines.strip().splitlines() if s.strip()
    ]
    if len(raw) != 2:
        return None
    for i, s in enumerate(raw):
        if not 2 <= len(s) <= 22 or s.count(refs[i]) != 1 or refs[1 - i] in s:
            return None
        extra = s.replace(refs[i], "", 1)
        if not re.search(r"[\u3400-\u9fff]", extra) or re.search(r"^[\d*#-]|[:：]", s):
            return None
    return raw


def main():
    WORK.mkdir(exist_ok=True)
    path = WORK / "teacher.jsonl"
    rows = sample("train")
    prior = (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )
    if [r["key"] for r in prior] != [v["key"] for v in rows[: len(prior)]]:
        raise ValueError("Teacher resume prefix differs")
    with local_teacher(MODEL, WORK) as (ask, digest, size), path.open("a") as out:
        if any(r["teacher_sha"] != digest for r in prior):
            raise ValueError("Teacher changed")
        for i, row in enumerate(rows[len(prior) :], len(prior)):
            qs = row["qs"]
            user = (
                row["page"]
                + "\n"
                + "\n".join(q["zh"] + " 答案：" + q["refs"][0] for q in qs)
            )
            result = ask(
                SYSTEM,
                dict(role="user", content=user),
                think=False,
                options=dict(temperature=0, num_predict=104, num_ctx=2048, seed=241209),
            )
            clauses = (
                clean(result["text"], [q["refs"][0] for q in qs])
                if result["stop"] == "stop"
                else None
            )
            entry = dict(
                key=row["key"], teacher_sha=digest, clauses=clauses, result=result
            )
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(i + 1, flush=True)
    all_rows = [json.loads(s) for s in path.read_text().splitlines()]
    record(
        PUBLIC / "teacher.json",
        dict(
            model=MODEL,
            sha=digest,
            weight_file_bytes=size,
            attempted=len(rows),
            accepted=sum(bool(v["clauses"]) for v in all_rows),
            duration_ns=sum(v["result"]["duration_ns"] for v in all_rows),
            replies_sha=sha(path),
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        main()
