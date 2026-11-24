"""Count natural held successes achievable with a previously observed answer frame."""

import re
from collections import defaultdict
from scripts.prepare_region_writer import WORK, PUB, inventory
from scripts.prepare_open_exam import dump
from scripts.evaluate_visual_discourse import unpack
from scripts.audit_region_writer import wording


def frame(reply, value):
    s, n = re.subn(re.escape(value), lambda m: " SIGNEDFACT ", reply, flags=re.I)
    return wording(s) if n else None


def main():
    train, rows = inventory()
    seen = defaultdict(set)
    frames = defaultdict(set)
    for r in train:
        if r["task"] == "restaurant":
            continue
        v = next(iter(r["facts"].values()))
        for s in r["answers"]:
            seen[r["task"]].add(wording(s))
            if f := frame(s, v):
                frames[r["task"]].add(f)
    out = {}
    for arm in ("direct", "selected"):
        results = unpack(WORK / (arm + "-predictions.json.gz"))["original"]
        data = {}
        for t in ("statements", "catalogue"):
            hits = []
            for r, p in zip(rows, results, strict=True):
                if (
                    r["task"] != t
                    or not p["stop"]
                    or wording(p["text"]) not in [wording(s) for s in r["answers"]]
                ):
                    continue
                s = p["text"]
                f = frame(s, next(iter(r["facts"].values())))
                hits.append(
                    dict(
                        family=r["family"],
                        whole_seen=wording(s) in seen[t],
                        value_span_found=f is not None,
                        frame_seen=f in frames[t],
                    )
                )
            data[t] = dict(
                hits=len(hits),
                training_target_matches=sum(r["whole_seen"] for r in hits),
                matched_value=sum(r["value_span_found"] for r in hits),
                seen_frame=sum(r["frame_seen"] for r in hits),
                cases=hits,
            )
        out[arm] = data
    dump(PUB / "frames.json", out)


if __name__ == "__main__":
    main()
