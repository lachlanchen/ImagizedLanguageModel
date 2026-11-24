"""Score Chinese sentence generation under actual questions and same-card exchanges."""

from collections import Counter, defaultdict
import gzip
import json
import re
from time import monotonic
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from scripts.prepare_chinese_card_reply import WORK, PUB, MODEL, FIELDS, COPY
from scripts.train_chinese_card_reply import router, output_bank, load_writer
from scripts.train_semantic_page import (
    read,
    archive,
    gpu_owner,
    snapshot,
    consumed,
    dump,
)


CUES = {
    "地址": "地址|位置|位于|坐落|地点",
    "评分": "评分|评价|打分|分数",
    "营业时间": "营业|开放|开门|营业时段",
    "人均消费": "人均|每人|消费",
    "酒店类型": "类型|型|档次|星级",
    "价格": "价格|价位|房价|元一晚|每晚|每夜",
    "门票": "门票|票价|入场|免费",
    "游玩时间": "游玩|游览|游园|玩儿|建议玩",
}


def includes(line, s):
    pattern = (
        (r"(?<![\d.])" if s[0].isdigit() else "")
        + re.escape(s)
        + (r"(?![\d.])" if s[-1].isdigit() else "")
    )
    return re.search(pattern, line) is not None


def grade(row, chosen, seq, chars):
    masked = "".join(chars[c] for c in seq)
    value = row["card"][chosen]
    line = masked.replace(COPY[0], row["name"]).replace(COPY[1], value)
    gold = row["card"][row["key"]]
    good = includes(line, gold)
    competing = any(
        includes(line, v)
        for k, v in row["card"].items()
        if k != row["key"] and v != gold and not includes(gold, v)
    )
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", gold + " " + row["name"]))
    numbers = set(re.findall(r"\d+(?:\.\d+)?", line))
    extra = bool(numbers - allowed_numbers)
    return dict(
        line=line,
        masked=masked,
        selection=chosen,
        route=chosen == row["key"],
        value=good,
        clean_value=good and not competing and not extra,
        added_number=extra,
        competing=competing,
        relation=bool(re.search(CUES[row["key"]], line)),
        exact=line == row["answer"],
        empty=not line,
        span=COPY[1] in masked,
    )


def score(records, idx):
    subset = [records[i] for i in idx]
    result = {
        k: sum(r[k] for r in subset)
        for k in (
            "route",
            "value",
            "clean_value",
            "exact",
            "competing",
            "added_number",
            "span",
            "empty",
        )
    }
    return result | dict(
        n=len(subset),
        explicit=sum(r["route"] and r["clean_value"] and r["relation"] for r in subset),
        distinct_masked=len({r["masked"] for r in subset}),
    )


def paired(rows):
    # First distinct human question per field on the identical complete database card.
    groups = defaultdict(dict)
    for i, r in enumerate(rows):
        key = (r["domain"], r["name"], tuple(sorted(r["card"].items())))
        groups[key].setdefault(r["key"], i)
    return [
        (a, b)
        for group in groups.values()
        for j, a in enumerate(group.values())
        for b in list(group.values())[j + 1 :]
        if rows[a]["q"] != rows[b]["q"]
    ]


def main():
    meta = read(PUB / "corpus.json")
    bank = archive(WORK / "bank.npz", meta["bank"])
    data = archive(WORK / "test.npz", meta["archives"]["test"])
    rows = read(WORK / "test.json")
    training = read(WORK / "train.json")
    chars = read(MODEL / "readback.json")
    qs = {r["q"] for r in training}
    names = {r["name"] for r in training}
    splits = {
        "all": list(range(len(rows))),
        "unseen_question": [i for i, r in enumerate(rows) if r["q"] not in qs],
        "unseen_entity": [i for i, r in enumerate(rows) if r["name"] not in names],
        "both_unseen": [
            i for i, r in enumerate(rows) if r["q"] not in qs and r["name"] not in names
        ],
    }
    pairs = paired(rows)
    scores = {}
    outputs = {}
    by_domain = defaultdict(Counter)
    for r in training:
        by_domain[r["domain"]][r["key"]] += 1
    scores["prior"] = {
        k: sum(
            max(rows[i]["card"], key=lambda f: by_domain[rows[i]["domain"]][f])
            == rows[i]["key"]
            for i in subset
        )
        for k, subset in splits.items()
    }
    with gpu_owner(), torch.inference_mode():
        out = output_bank("cuda")
        for mode in ("ink", "byte"):
            net = load_writer(mode, "cuda")
            ink = torch.as_tensor(observations(bank, mode), device="cuda")
            label_states = net.encode_rows(ink[bank["labels"]])
            for erase in (False, True):
                records = []
                clock = monotonic()
                energy = snapshot()
                for j in range(0, len(rows), 32):
                    selected = slice(j, j + 32)
                    query = ink[data["question"][selected]]
                    if erase:
                        query = torch.zeros_like(query)
                    domains = torch.as_tensor(
                        data["domain"][selected], device="cuda"
                    ).long()
                    available = torch.as_tensor(
                        data["available"][selected], device="cuda"
                    )
                    logits, mem = router(net, query, label_states, domains, available)
                    valid = torch.ones(mem.shape[:2], dtype=torch.bool, device="cuda")
                    ids, ns = beam_write(net, mem, valid, out, None, limit=120)
                    for i, (seq, n, relation) in enumerate(
                        zip(ids.tolist(), ns.tolist(), logits.argmax(1).tolist())
                    ):
                        records.append(
                            grade(rows[j + i], FIELDS[relation], seq[:n], chars)
                        )
                cost = dict(
                    seconds=monotonic() - clock, energy=consumed(energy, snapshot())
                )
                key = mode + ("-erased" if erase else "")
                scores[key] = {k: score(records, idx) for k, idx in splits.items()}
                scores[key]["pairs"] = dict(
                    n=len(pairs),
                    both_route=sum(
                        records[i]["route"] and records[j]["route"] for i, j in pairs
                    ),
                    both_clean=sum(
                        records[i]["clean_value"] and records[j]["clean_value"]
                        for i, j in pairs
                    ),
                    both_explicit=sum(
                        all(
                            records[k]["route"]
                            and records[k]["clean_value"]
                            and records[k]["relation"]
                            for k in (i, j)
                        )
                        for i, j in pairs
                    ),
                )
                scores[key]["cost"] = cost
                outputs[key] = records
                print(key, scores[key]["all"], scores[key]["pairs"], flush=True)
    scores["human"] = {
        "explicit_relation_verbalized": sum(
            bool(re.search(CUES[r["key"]], r["answer"])) for r in rows
        ),
        "n": len(rows),
    }
    dump(PUB / "results.json", scores)
    # Keep dialogue IDs and actual outputs. The corpus is separately fetched from its authors.
    ledger = dict(
        keys=[[r["dialog"], r["turn"]] for r in rows], pairs=pairs, outputs=outputs
    )
    (PUB / "outputs.json.gz").write_bytes(
        gzip.compress(json.dumps(ledger, ensure_ascii=False).encode(), mtime=0)
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
