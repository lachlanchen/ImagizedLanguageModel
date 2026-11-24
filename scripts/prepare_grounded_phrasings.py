"""Collect training-only human formulations with identical annotated facts."""

import csv
import json
import re
import unicodedata
from collections import defaultdict

from scripts.prepare_chinese_card_reply import DOMAINS, RAW as CROSS, extract
from scripts.prepare_evidence_views import PUB, WORK
from scripts.prepare_open_exam import ROOT, read, dump, sha


def phrase_key(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def groups():
    english = ROOT / "data/raw/e2e_cleaned/train-fixed.no-ol.csv"
    english_receipt = read(ROOT / "references/evidence/semantic_page_v1/corpus.json")
    assert sha(english) == english_receipt["files"][english.name]
    stock = {"en": defaultdict(dict), "zh": defaultdict(dict)}
    with english.open(encoding="utf8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            facts = tuple(
                sorted(
                    (k.strip(), v)
                    for k, v in re.findall(r"([^,\[]+)\[([^\]]*)\]", row["mr"])
                )
            )
            if not facts or len(facts) != row["mr"].count("["):
                raise ValueError("Unparsed field in E2E meaning representation")
            target = row["ref"].strip()
            stock["en"][facts].setdefault(
                phrase_key(target), dict(answer=target, row=idx)
            )
    ref = read(ROOT / "references/evidence/chinese_reply_v1/corpus.json")
    originals = [CROSS / "train.json.zip"] + [
        CROSS / (name + "_db.json") for name in DOMAINS.values()
    ]
    for f in originals:
        k = (
            "data/crosswoz/database/" if f.suffix == ".json" else "data/crosswoz/"
        ) + f.name
        assert sha(f) == ref["files"][k]
    db = {d: dict(read(CROSS / (name + "_db.json"))) for d, name in DOMAINS.items()}
    rows, _ = extract("train", db)
    for row in rows:
        field = row["key"]
        facts = (row["domain"], row["name"], field, row["card"][field])
        target = row["answer"]
        stock["zh"][facts].setdefault(
            phrase_key(target),
            dict(
                answer=target, question=row["q"], dialog=row["dialog"], turn=row["turn"]
            ),
        )
    return stock, {f.name: sha(f) for f in [english, *originals]}


def main():
    stock, sources = groups()
    report, pooled = {}, []
    for lang, data in stock.items():
        multi = {k: v for k, v in data.items() if len(v) >= 2}
        eligible = {
            k: [a for a in v.values() if 0 < len(a["answer"]) < 64]
            for k, v in multi.items()
        }
        under_cap = {k: v for k, v in eligible.items() if len(v) >= 2}
        by_budget = {
            decisions: sum(
                sum(0 < len(r["answer"]) < decisions for r in alternatives.values())
                >= 2
                for alternatives in multi.values()
            )
            for decisions in (64, 128, 256, 512)
        }
        report[lang] = dict(
            pairs_by_decision_budget=by_budget,
            annotated_facts=len(data),
            multiple_wordings=len(multi),
            distinct_formulations=sum(map(len, multi.values())),
            two_under_current_cap=len(under_cap),
            distinct_under_cap=sum(map(len, under_cap.values())),
        )
        pooled.extend(
            dict(language=lang, facts=k, responses=list(v.values()))
            for k, v in multi.items()
        )
    path = WORK / "training-fact-wordings.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as f:
        for r in pooled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    dump(
        PUB / "training-fact-wordings.json",
        dict(
            sources=sources,
            sha256=sha(path),
            counts=report,
            scope="Original training partitions only. Distinct wording after case, punctuation and whitespace folding. Grouping uses equality of the annotated fact record. Additional unannotated claims or conversational differences require separate semantic screening.",
        ),
    )
    print(report)


if __name__ == "__main__":
    main()
