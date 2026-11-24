"""Recount the stated strict reading of the first 24 fresh, hash-ordered outputs."""

from hashlib import sha256
import gzip
import json
from scripts.evaluate_visual_cursor import normalize
from scripts.prepare_visual_statements import PUB, WORK, read, dump

# A strict pass needs a grammatical, standalone proposition that answers the
# question as supported by the pictured excerpt. Partial facts stay visible.
# Ordinals follow the reserved passage order, not an ordering by model score.
LABELS = [
    "circular",
    "wrong entity",
    "one of three",
    "wrong cause",
    "incomplete predicate",
    "missing place",
    "role in place of actor",
    "correct",
    "correct",
    "count without referent",
    "incomplete name",
    "wrong person",
    "correct",
    "broken syntax",
    "circular",
    "circular",
    "date in place of programme",
    "place in place of rank",
    "broken syntax",
    "unreadable adjective",
    "missing list",
    "correct",
    "circular",
    "incomplete place",
]


FROZEN_ANSWERS = {
    "before": "22f0864d01f02716a5af39b0cb796a9687872675005a323f27ad877b6e456f45",
    "trained": "17b80b65f169ac45c9aa97d98605f36dcc56f2702abfba1fe32068d5321ca0e2",
}


def main():
    fresh = read(WORK / "fresh.json")
    source = fresh[: len(LABELS)]
    visibility = {}
    result = {}
    for arm in ("before", "trained"):
        payload = gzip.decompress((PUB / (arm + "-answers.json.gz")).read_bytes())
        if sha256(payload).hexdigest() != FROZEN_ANSWERS[arm]:
            raise ValueError("The stored readings belong to different model outputs")
        raw = json.loads(payload)["fresh"]
        conditioned = {
            key: {"pages": 0, "fact_read": 0, "output_contains_fact": 0}
            for key in ("visible", "outside_crop")
        }
        for page, output in zip(fresh, raw, strict=True):
            assert page["id"] == output["id"]
            visible = any(
                normalize(ref) and normalize(ref) in normalize(page["source"])
                for ref in page["refs"]
            )
            bucket = conditioned["visible" if visible else "outside_crop"]
            bucket["pages"] += 1
            bucket["fact_read"] += int(output["fact_read"])
            bucket["output_contains_fact"] += int(output["contains"])
        visibility[arm] = conditioned
        rows = []
        for i, (page, v) in enumerate(zip(source, raw)):
            assert v["id"] == page["id"]
            label = LABELS[i] if arm == "trained" else "short fragment"
            rows.append(
                dict(
                    id=page["id"],
                    order=i + 1,
                    shape_sha256=v["shape_sha256"],
                    stopped=v["stopped"],
                    judgement=label,
                    absent_from_both_inputs=all(
                        "".join(v["output"].lower().split())
                        not in "".join(t.lower().split())
                        for t in (page["source"], page["question"])
                    ),
                    strict=label == "correct" and v["stopped"],
                )
            )
        result[arm] = rows
    dump(
        PUB / "fresh-reading.json",
        dict(
            selection="First 24 fresh passage families in fixed hash order. Exploratory after fitting.",
            assessment="One project model-based reviewer with access to arm identity, source, prompt and reference. No independent human adjudication.",
            requirement="Stopped, grammatical, standalone proposition giving the requested grounded answer; a right bare name or number earns no full-sentence credit.",
            counts={k: sum(x["strict"] for x in v) for k, v in result.items()},
            fresh_reference_visibility=visibility,
            rows=result,
        ),
    )


if __name__ == "__main__":
    main()
