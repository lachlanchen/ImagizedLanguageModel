"""Count known protected-source exposure from saved records without rendering or inference."""

import json
from pathlib import Path
from scripts.span_reservation import Reservation, ROOT
from scripts.prepare_span_list import PUBLIC


def main():
    reserve = Reservation()
    ids = set()
    paras = set()
    counts = {}
    splits = {
        "fact_" + s: ROOT / f"artifacts/role_binding_v1/{s}.jsonl"
        for s in ("training", "held")
    }
    for name in ("span_choice", "span_list"):
        for s in ("train", "validation", "fresh"):
            splits[name + "_" + s] = ROOT / f"artifacts/{name}_v1/{s}.json"
    splits["earlier_transfer"] = (
        ROOT / "artifacts/role_binding_v1/transfer_questions.json"
    )
    for name, path in splits.items():
        data = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text())
        )
        hits = {
            r["key"]
            for r in data
            if r.get("group") == "drcd" and r.get("key") in reserve.by_question
        }
        matched = {reserve.by_question[k] for k in hits}
        ids.update(hits)
        paras.update(matched)
        counts[name] = dict(
            records=len(data),
            reserved_ids=len(hits),
            known_paragraphs=len(matched),
            screened_out=sum(reserve.rejects(r) for r in data)
            if data and "p" in data[0]
            else None,
            rows_with_reserved_text=sum(reserve.overlaps(r["p"]) for r in data)
            if data and "p" in data[0]
            else None,
        )
    out = dict(
        scope="Named saved cohorts only; counts are a lower bound across project history.",
        counts=counts,
        distinct_question_ids=len(ids),
        distinct_paragraph_ids=len(paras),
        affected_article_ids=len({reserve.by_paragraph[p] for p in paras}),
    )
    dest = PUBLIC / "reservation_audit.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
