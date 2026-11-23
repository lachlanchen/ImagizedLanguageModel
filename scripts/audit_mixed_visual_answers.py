"""Independently score the packed pixels for both questions, their blank control and 言."""

import json
from collections import Counter
from functools import lru_cache

from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import WORK, PUBLIC


def audit():
    out = WORK / "evaluation"
    report = json.loads((PUBLIC / "evaluation.json").read_text())
    data = out / "extra.json"
    assert sha(data) == report["extra_sha256"]
    cases = json.loads(data.read_text())
    f = Face((FONTS[0], 2))
    face = lru_cache(maxsize=1000)(f)
    pairs = [cases[i : i + 2] for i in range(0, len(cases), 2)]
    assert len({p[0]["p"] for p in pairs}) == len(pairs)
    assert all(
        a["p"] == b["p"]
        and a["q"] != b["q"]
        and not {face(r) for r in a["refs"]} & {face(r) for r in b["refs"]}
        for a, b in pairs
    )

    def get(arm, key):
        path = out / f"{arm}_{key}.npz"
        stats = report["results"][arm][key]
        assert sha(path) == stats["sha256"]
        return list(zip(*load_streams(path), strict=True)), stats

    result = {}
    for arm, key in [
        ("mixed", "extra_cmrc"),
        ("parent", "extra_cmrc"),
        ("extractor", "extra_cmrc"),
        ("mixed", "blind_cmrc"),
    ]:
        streams, prior = get(arm, key)
        exact = [
            stop and any(y == face(ref) for ref in v["refs"])
            for (y, stop), v in zip(streams, cases, strict=True)
        ]
        both = [a and b for a, b in zip(exact[::2], exact[1::2], strict=True)]
        changed = [a != b for a, b in zip(streams[::2], streams[1::2], strict=True)]
        assert all(not ok or changed[i] for i, ok in enumerate(both))
        assert (sum(exact), sum(both)) == (prior["exact"], prior["complete"])
        result[arm + "_" + key] = dict(
            correct=sum(exact), both=sum(both), changed=sum(changed)
        )
    mixed, _ = get("mixed", "extra_cmrc")
    teacher, _ = get("extractor", "extra_cmrc")
    failures = Counter()
    for (y, stop), (t, _), v in zip(mixed, teacher, cases, strict=True):
        refs = [face(s) for s in v["refs"]]
        ok = stop and y in refs
        failures["both_right"] += int(ok and t in refs)
        failures["mixed_only"] += int(ok and t not in refs)
        failures["reader_only"] += int(not ok and t in refs)
        failures["neither"] += int(not ok and t not in refs)
        if not ok:
            failures["wrong_first_ink"] += int(
                all(y[:128] != ref[:128] for ref in refs)
            )
            failures["right_first_ink"] += int(
                any(y[:128] == ref[:128] for ref in refs)
            )
            failures["correct_prefix_then_longer"] += int(
                any(y.startswith(ref) and len(y) > len(ref) for ref in refs)
            )
            failures["incomplete_prefix"] += int(
                any(ref.startswith(y) and len(y) < len(ref) for ref in refs)
            )
    result["mixed_reading_errors"] = dict(failures)
    rows = [r for r in plates("development", f) if r.task == "yan"]
    streams, stats = get("mixed", "yan")
    answers = [stop and y == r.y for (y, stop), r in zip(streams, rows, strict=True)]
    assert sum(answers) == stats["exact"]
    result["catalogue_card"] = answers
    result["additional_questions"] = dict(Counter(v["group"] for v in cases))
    record(PUBLIC / "recount.json", result)


if __name__ == "__main__":
    audit()
