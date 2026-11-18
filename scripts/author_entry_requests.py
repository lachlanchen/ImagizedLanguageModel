"""Elicit Chinese questions from a separate local writer without answers or training wording."""

from collections import defaultdict
import fcntl
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.local_translation_teacher import local_teacher
from scripts.prepare_pictured_entry_answers import (
    DEFAULT,
    STAGES,
    load,
    sha,
    write_json,
)
from scripts.train_pictured_entry_writer import ROOT

OUT = ROOT / "references/evidence/independent_entry_20261119"
WORK = ROOT / "artifacts/independent_entry_v1"
MODEL = "qwen3:8b-q8_0"
SYSTEM = """你替参观汉字展的人写一句简短自然的中文提问。读者能看到两条字典资料，里面有本义的英文释义、现代普通话主要读音、以及古文字字形。根据意图写一句能独立理解的问题，指明要查哪个字，问所需的一个信息并要求展示指定的一种古文字。不要作答，不要猜测字源，不要多求资料。允许用生活化说法。只返回问题本身，不加编号。"""
STYLES = (
    "像参观者随口问。",
    "像读书时写在页边的问题。",
    "像向老师请教。",
    "请用简洁的一句话。",
)


def choose():
    rows = load(DEFAULT / "development/index.json")
    grouped = defaultdict(list)
    for i in range(0, len(rows), 8):
        grouped[tuple(sorted((rows[i]["stage"], rows[i + 1]["stage"])))].append(i)
    rng = np.random.default_rng(191177)
    starts = []
    for group in sorted(grouped):
        picked = rng.choice(grouped[group], size=4, replace=False)
        starts.extend(int(i + 4 * rng.integers(2)) for i in picked)
    rng.shuffle(starts)
    assert len(starts) == len({rows[i]["card"] for i in starts}) == 24
    for group, i in enumerate(starts):
        for j in range(4):
            r = rows[i + j]
            field = "最初的意思" if r["field"] == "本义" else "现代普通话主要读音"
            wanted = STAGES[r["stage"]][0]
            instruction = f"要查的字：{r['head']}。想知道：{field}。请对方展示：{wanted}字形。{STYLES[(group+j)%4]}"
            if group % 2:
                other = STAGES[rows[i + (j ^ 1)]["stage"]][0]
                known = "普通话发音" if r["field"] == "本义" else "最初的意义"
                instruction += f"提问中自然说明已经知道{known}，已经看过{other}的字形，两者都无须再答。"
            yield dict(row=i + j, group=group, instruction=instruction)


def author():
    destination = OUT / "questions.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    schedule = list(choose())
    ledger = WORK / "author.jsonl"
    existing = (
        [json.loads(line) for line in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    assert [r["row"] for r in existing] == [r["row"] for r in schedule[: len(existing)]]
    with local_teacher(MODEL, WORK) as (ask, digest, size):
        previous = WORK / "model.json"
        model = dict(
            model=MODEL,
            digest=digest,
            bytes=size,
            system=SYSTEM,
            code_sha=sha(Path(__file__)),
        )
        if previous.exists():
            assert load(previous) == model
        else:
            write_json(previous, model)
        with ledger.open("a", encoding="utf-8") as f:
            for index, row in enumerate(schedule[len(existing) :], start=len(existing)):
                result = ask(
                    SYSTEM,
                    dict(role="user", content=row["instruction"]),
                    think=False,
                    options=dict(
                        seed=7111 + index,
                        temperature=0.75,
                        num_predict=145,
                        num_ctx=2048,
                    ),
                )
                f.write(json.dumps(row | result, ensure_ascii=False) + "\n")
                f.flush()
                print(index + 1, flush=True)
    raw = ledger.read_bytes()
    assert len(raw.splitlines()) == 96
    destination.write_bytes(raw)
    write_json(
        OUT / "authorship.json",
        model
        | dict(
            questions=96,
            cards=24,
            source_sha=sha(DEFAULT / "development/index.json"),
            output_sha=sha256(raw).hexdigest(),
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        author()
