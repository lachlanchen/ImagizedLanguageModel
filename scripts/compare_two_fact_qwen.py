"""Run the local quantized Qwen text comparator or blinded whole-sentence ratings."""

import argparse
import gzip
import json
import random
from time import monotonic

from scripts.evaluate_chinese_two_fact import grade
from scripts.local_translation_teacher import local_teacher
from scripts.prepare_chinese_two_fact import WORK, PUB, read, dump
from scripts.train_semantic_page import gpu_owner, snapshot, consumed

TAG = "qwen3:8b-q8_0"
ANSWER = "请用自然、简洁的中文回答顾客的问题。只依据这条场所资料，逐项回答用户所问。保留名称、数值和地址的原文，不添加资料之外的事实。只输出最终答复。"
JUDGE = """你是独立的中文编辑。根据场所资料、顾客的问题和匿名答复，分别判断三件事。complete：所有被问到的事实是否都正确且角色清楚，意思相同即可。grounded：答复是否没有凭空增加事实或判断。fluent：语言是否通顺可直接发给顾客，允许日常简短说法。三项应分别判断。只输出指定格式的JSON。"""
OPTIONS = dict(temperature=0, seed=14071, num_ctx=2048, num_predict=160)


def content(row):
    return json.dumps(
        {
            "场所": row["name"],
            "类别": row["domain"],
            "资料": row["card"],
            "顾客": row["q"],
        },
        ensure_ascii=False,
    )


def generated(ask, rows):
    outputs = []
    for r in rows:
        response = ask(
            ANSWER, dict(role="user", content=content(r)), think=False, options=OPTIONS
        )
        outputs.append(response | {"score": grade(response["text"], [], r)})
    return outputs


def judged(ask, rows):
    outputs = json.loads(gzip.decompress((PUB / "replies.json.gz").read_bytes()))[
        "test"
    ]
    covered = read(PUB / "coverage.json")["test"]["lines"]
    qwen = read(PUB / "qwen-answer.json")["outputs"]
    aligned = read(PUB / "alignment-ink.json")["test"]["lines"]
    separate = read(PUB / "aligned-separate.json")["lines"]
    arms = (
        "ink-joined",
        "separate",
        "byte-joined",
        "covered",
        "aligned",
        "aligned-separate",
        "human",
        "qwen",
    )
    fields = ("complete", "grounded", "fluent")
    schema = {
        "type": "object",
        "properties": {k: {"type": "boolean"} for k in fields},
        "required": list(fields),
        "additionalProperties": False,
    }
    candidates = {}
    for j, r in enumerate(rows):
        for arm in arms:
            if arm == "aligned":
                line = aligned[j]["line"]
            elif arm == "aligned-separate":
                line = separate[j]["line"]
            elif arm == "qwen":
                line = qwen[j]["text"]
            elif arm == "human":
                line = r["answer"]
            elif arm == "covered":
                line = covered[j]["line"]
            else:
                line = outputs[arm][len(outputs[arm]) - len(rows) + j]["line"]
            candidates.setdefault((j, line), []).append(arm)
    items = list(candidates)
    random.Random(84318).shuffle(items)
    out = []
    for j, line in items:
        user = content(rows[j]) + "\n匿名答复：" + line
        response = ask(
            JUDGE,
            dict(role="user", content=user),
            think=False,
            format=schema,
            options=OPTIONS,
        )
        verdict = json.loads(response["text"])
        if any(not isinstance(verdict[k], bool) for k in fields):
            raise ValueError(verdict)
        out.append(dict(index=j, line=line, arms=candidates[j, line], verdict=verdict))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("answer", "judge"))
    mode = ap.parse_args().mode
    rows = [r for r in read(WORK / "test.json") if len(r["keys"]) == 2]
    dest = PUB / ("qwen-" + mode + ".json")
    if dest.exists():
        raise FileExistsError(dest)
    directory = WORK / ("qwen-" + mode)
    directory.mkdir(exist_ok=True)
    with gpu_owner(), local_teacher(TAG, directory) as (ask, manifest, size):
        start = monotonic()
        energy = snapshot()
        out = generated(ask, rows) if mode == "answer" else judged(ask, rows)
        cost = consumed(energy, snapshot())
        seconds = monotonic() - start
    dump(
        dest,
        dict(
            tag=TAG,
            manifest=manifest,
            size_bytes=size,
            seconds=seconds,
            energy=cost,
            outputs=out,
        ),
    )
    print(mode, len(out), flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
