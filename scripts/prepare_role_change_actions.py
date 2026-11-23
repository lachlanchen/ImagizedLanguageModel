"""Collect bilingual activity phrases from book passages for controlled fictional edits."""

import json
import re

from scripts.local_translation_teacher import local_teacher
from scripts.prepare_ordered_book_replies import MODEL, ROOT
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_ordered_book_replies import cases
from scripts.train_span_choice import gpu_owner

WORK = ROOT / "artifacts/role_change_writer_v1"
PUBLIC = ROOT / "references/evidence/role_change_writer_20261124"
SYSTEM = """取材于提供的故事场景，给两个不同的人设计两项不同、具体、合理的动作。每项中文动作是三至八个字的动宾短语，不写人名、主语、标点、否定、时间或“了”。给出可直接放在 Who 后面的英文过去式动词短语。避免近义的动作。只输出两行，每行用竖线分隔中文和英文，例如：修补封面|repaired the cover"""


def parse(text):
    pairs = []
    for line in text.splitlines():
        fields = [v.strip() for v in line.strip().split("|")]
        if len(fields) != 2:
            return None
        z, e = fields
        if not re.fullmatch(r"[\u3400-\u9fff]{3,8}", z) or not re.fullmatch(
            r"[A-Za-z][A-Za-z '’-]{3,55}", e
        ):
            return None
        if any(c in z for c in "没不未是否谁了"):
            return None
        pairs.append([z, e])
    if (
        len(pairs) != 2
        or pairs[0][0][0] == pairs[1][0][0]
        or pairs[0][1] == pairs[1][1]
    ):
        return None
    return pairs


def run():
    WORK.mkdir(exist_ok=True)
    rows = cases()["train"]
    path = WORK / "actions.jsonl"
    prior = (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )
    if [v["key"] for v in prior] != [c["key"] for c in rows[: len(prior)]]:
        raise ValueError("Source order changed")
    with local_teacher(MODEL, WORK) as (ask, digest, size), path.open("a") as dst:
        if any(v["teacher_sha"] != digest for v in prior):
            raise ValueError("Teacher manifest changed")
        for c in rows[len(prior) :]:
            r = ask(
                SYSTEM,
                dict(role="user", content=c["page"]),
                think=False,
                options=dict(seed=242110, temperature=0, num_ctx=2048, num_predict=90),
            )
            v = dict(
                key=c["key"],
                teacher_sha=digest,
                pair=parse(r["text"]) if r["stop"] == "stop" else None,
                raw=r,
            )
            dst.write(json.dumps(v, ensure_ascii=False) + "\n")
            dst.flush()
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    record(
        PUBLIC / "teacher.json",
        dict(
            count=len(rows),
            accepted=sum(bool(v["pair"]) for v in rows),
            sha256=sha(path),
            model=MODEL,
            model_sha=digest,
            bytes=size,
            seconds=sum(v["raw"]["duration_ns"] for v in rows) * 1e-9,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
