"""Prepare crossed Chinese clauses from actual training or validation claim predicates."""
from collections import Counter
from dataclasses import dataclass
import json
import re

import numpy as np

from scripts.prepare_visual_inference_pairs import DEST, RAW, RESPONSES
from scripts.train_visual_inference_writer import DECIDE, PRIVATE, load

POLARITIES = (
    ("喜欢","不喜欢"), ("知道","不知道"), ("了解","不了解"), ("熟悉","不熟悉"),
    ("愿意","不愿意"), ("同意","不同意"), ("想要","不想要"), ("需要","不需要"),
    ("相信","不相信"), ("支持","不支持"), ("能","不能"), ("会","不会"),
    ("可以","不可以"), ("有","没有"), ("想","不想"), ("认识","不认识"),
    ("记得","不记得"), ("打算","不打算"), ("见过","没见过"), ("去过","没去过"),
)
FORMS = {form: pair for pair in POLARITIES for form in pair}
PREFIX = re.compile(
    r"^(?:他们|她们|我们|你们|他|她|我|你)(?:也|并|都|其实|还)?("
    + "|".join(sorted(FORMS, key=len, reverse=True)) + r")([\u3400-\u9fff]{2,18})[。！!]?$"
)
TRAIN_NAMES = tuple("方宁 安文 唐菲 吴可 李晨 马俊 田静 孙敏 陈慧 黄莉 冯源 周航 赵凡 郑云 宋清 高亮 徐川 梁平 杨帆 林蓉 罗芳 沈康 白雪 江月".split())
TEST_NAMES = tuple("邓琪 贺松 曹虹 严欣 季明 段哲 姚雪 温岚".split())
ASK = ("请把{}的情况写成一句话，把姓名放在“的是”后面。", "请用“……的是某人。”写出{}的情况。")
NEW_ASK = "根据上文，{}的情况如何？写一句话，用“的是”引出姓名。"


@dataclass(frozen=True)
class Predicate:
    positive: str
    negative: str
    complement: str


def predicates():
    rows = load(DEST/"train/rows.json")
    raw = [json.loads(line)["sentence2"] for line in (RAW/"train.50k.json").read_text().splitlines()]
    samples = {}
    seen = set()
    with np.load(PRIVATE) as splits:
        for split in ("fit", "validation"):
            found = set()
            for idx in splits[split]:
                match = PREFIX.fullmatch(raw[rows[idx]["line"]])
                if not match:
                    continue
                form, tail = match.groups()
                # Exclude lexicalized/modifier uses that do not license mechanical negation.
                if tail in seen or (form in ("有","没有") and tail.startswith(("些","点","时候","关","所"))):
                    continue
                pair = FORMS[form]
                if tail.startswith(("不","没","法","论","疑","何")):
                    continue
                found.add((*pair, tail))
            seen.update(tail for _,_,tail in found)
            samples[split] = [Predicate(*row) for row in sorted(found)]
    assert not ({p.complement for p in samples["fit"]} & {p.complement for p in samples["validation"]})
    return samples


def situation(pred, names, positive_first, layout, task, negative_claim=False):
    """Order is passage orientation, then queried entity; the four targets change on both axes."""
    pos, neg = pred.positive+pred.complement, pred.negative+pred.complement
    records = []
    for first,second in (names,names[::-1]):
        pos_name, neg_name = (first,second) if positive_first else (second,first)
        facts = {pos_name:pos, neg_name:neg}
        if layout == 0:
            passage = first+facts[first]+"，"+second+facts[second]+"。"
        elif layout == 1:
            passage = first+facts[first]+"；而"+second+facts[second]+"。"
        else:
            passage = "据记录，"+first+facts[first]+"。另据记录，"+second+facts[second]+"。"
        for name in names:
            if task == "decision":
                claim = name+(neg if negative_claim else pos)+"。"
                question = DECIDE+claim
                true = (name==pos_name) != negative_claim
                answer = RESPONSES[0 if true else 1]
            else:
                question = (NEW_ASK if layout==2 else ASK[layout]).format(name)
                answer = facts[name]+"的是"+name+"。"
            records.append((passage,question,answer))
    return records
