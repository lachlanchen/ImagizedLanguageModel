"""Build screened, paired glyph-writing examples from Chinese fact passages."""
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts.grounded_proposition_corpus import LANG, noun
from scripts.prepare_visual_facts import ROOT
from scripts.span_reservation import normalized, reservation
from scripts.train_fact_realizer import COMMANDS, authored_clause, target
from scripts.train_visual_facts import rows

DEFAULT = ROOT / "artifacts/page_evidence_writer_v1"
SEED = 241129
YEAR = re.compile(r"(?<![0-9])([12][0-9]{3})年")
FOUR_DIGITS = re.compile(r"(?<![0-9])[0-9]{4}(?![0-9])")


def unique_year(x):
    years = YEAR.findall(x["p"])
    if len(years) != 1 or len(FOUR_DIGITS.findall(x["p"])) != 1:
        return None
    y = years[0]
    if y in x["s"] or y in x["o"] or str(int(y) + 1) in x["p"]:
        return None
    return y


def item(task, board, x, p, q, y, *, year="", rival="", request=0, edit=0):
    return dict(task=task, board=board, key=x["key"], p=p, q=q, y=y,
                s=x["s"], r=x["r"], o=x["o"], year=year, rival=rival,
                request=request, edit=edit)


def date_item(task, board, x, p, wanted, other, held, request, edit):
    name = noun(x["s"], x["r"])
    q = (f"请介绍{name}，并指出同一段落写的是哪一年。" if held else
         f"先说明{name}，再写同段提到的年份。")
    answer = target(x, 0) + f"该段提到{wanted}年。"
    return item(task, board, x, p, q, answer, year=wanted, rival=other,
                request=request, edit=edit)


def fits(v):
    # The deployed decoder reserves 48 output positions. Its layout repeats q.
    return len(v["p"]) + 2 * len(v["q"]) + 4 + 48 <= 1024 and len(v["y"]) <= 48


def single_cases(x, held):
    year = unique_year(x)
    other = str(int(year) + 1)
    alt = x["p"].replace(year + "年", other + "年", 1)
    if reservation().overlaps(alt):
        return []
    return [date_item("single", x["key"], x, p, good, bad, held, 0, e)
            for e, (p, good, bad) in enumerate(((x["p"], year, other), (alt, other, year)))]


def dual_cases(a, b, held):
    left, right = a["p"], b["p"]
    first, second = unique_year(a), unique_year(b)
    if first == second or any(x in right for x in (a["s"], first)) or any(
        x in left for x in (b["s"], second)
    ):
        return []
    board = a["key"] + "/" + b["key"]
    records = []
    for edit, (y1, y2) in enumerate(((first, second), (second, first))):
        page = left.replace(first + "年", y1 + "年", 1) + "　" + right.replace(second + "年", y2 + "年", 1)
        if reservation().overlaps(page):
            return []
        records += [date_item("dual", board, a, page, y1, y2, held, 0, edit),
                    date_item("dual", board, b, page, y2, y1, held, 1, edit)]
    return records if all(map(fits, records)) else []


def build(data, held, seed):
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    duie = [x for x in data if x["group"] == "duie"]
    out = []
    for x in duie[:256 if held else 4800] + ([x for x in data if x["group"] == "cmrc"] if held else []):
        for k in (0, 1):
            out.append(item(x["group"], x["key"], x, x["p"], x["q"] + COMMANDS[k], target(x, k), request=k))
    n = 0
    for x in duie:
        c = authored_clause(x)
        if c:
            out.append(item("clause", x["key"], x, x["p"], x["q"] + "请摘录原文中同时提到两者的句子。", c))
            n += 1
            if n == (128 if held else 2048):
                break
    previous = set(json.loads((ROOT / "references/evidence/span_list_screened_20261124/details.json").read_text())["held_keys"])
    dated = [x for x in duie if unique_year(x) and (not held or x["key"] not in previous)]
    single_limit = 96 if held else 1800
    consumed = set()
    for x in dated:
        group = single_cases(x, held)
        if group and all(map(fits, group)):
            out.extend(group)
            consumed.add(x["key"])
        if len(consumed) == single_limit:
            break
    remaining = [x for x in dated if x["key"] not in consumed]
    # Each physical two-paragraph board uses two distinct source records.
    paired = 0
    while len(remaining) >= 2 and paired < (64 if held else 1152):
        a = remaining.pop()
        for i in range(len(remaining) - 1, -1, -1):
            group = dual_cases(a, remaining[i], held)
            if group:
                out.extend(group)
                remaining.pop(i)
                paired += 1
                break
    return [v for v in out if fits(v)]


def make_splits():
    train = [v for v in rows("training", eligible_only=True) if v["group"] == "duie"]
    held = [v for v in rows("held", eligible_only=True) if v["group"] in ("duie", "cmrc")]
    held_pages = {normalized(x["p"]) for x in held}
    held_names = {normalized(x["s"]) for x in held}
    windows = {p[i:i+64] for p in held_pages for i in range(len(p)-63)}
    def eligible(x):
        p = normalized(x["p"])
        return (p not in held_pages and normalized(x["s"]) not in held_names
                and all(p[i:i+64] not in windows for i in range(len(p)-63)))
    train = [x for x in train if eligible(x)]
    return build(train, False, SEED), build(held, True, SEED + 1)


def save_split(folder, name, data):
    import hashlib
    path = Path(folder) / (name + ".jsonl")
    raw = "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in data).encode()
    with path.open("xb") as f:
        f.write(raw)
    return dict(rows=len(data), groups=dict(Counter(x["task"] for x in data)),
                sha256=hashlib.sha256(raw).hexdigest())
