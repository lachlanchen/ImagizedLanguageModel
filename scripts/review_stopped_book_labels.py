"""Build a fixed, stratified review packet before either fitting condition runs."""

from collections import defaultdict
from hashlib import sha256
import json

from scripts.mine_stopped_book_replies import PUBLIC, WORK, training_cases
from scripts.train_causal_glyph_writer import record, sha


def packet():
    items = training_cases()
    accepted = json.loads((WORK / "accepted.json").read_text())
    report = json.loads((PUBLIC / "adjudication.json").read_text())
    assert sha(WORK / "accepted.json") == report["accepted_sha"]
    strata = defaultdict(list)
    for v in accepted:
        id_, lang = v["id"], v["lang"]
        rank = sha256(("independent-review-1/" + id_ + "/" + lang).encode()).hexdigest()
        strata[lang, v["kind"]].append((rank, v))
    rows = []
    for group in sorted(strata):
        for _, v in sorted(strata[group])[:12]:
            row, q = items[v["id"], v["lang"]]
            rows.append(dict(**v, question=q[v["lang"]], page=row["page"]))
    record(WORK / "review_packet.json", rows)
    return rows


def apply():
    rows = packet()
    audit = json.loads((PUBLIC / "review_decisions.json").read_text())
    assert [a["id"] for a in audit] == [v["id"] + "/" + v["lang"] for v in rows]
    assert all(a["decision"] in ("supported", "exclude") and a["reason"] for a in audit)
    rejected = [
        (v["id"], v["lang"])
        for a, v in zip(audit, rows, strict=True)
        if a["decision"] == "exclude"
    ]
    report = dict(
        n=len(rows),
        supported=len(rows) - len(rejected),
        exclude=rejected,
        admit=(len(rows) - len(rejected)) / len(rows) >= 0.90,
        packet_sha=sha(WORK / "review_packet.json"),
        accepted_sha=sha(WORK / "accepted.json"),
        reviewer="model-assisted reading of pictured-training source transcripts",
    )
    record(PUBLIC / "label_review.json", report)
    print(report)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--apply"]:
        apply()
    else:
        packet()
