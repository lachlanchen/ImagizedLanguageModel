"""Complete the bilingual book training split with Chinese clause targets from a local teacher."""

import json
from scripts.prepare_ordered_book_replies import (
    sample,
    MODEL,
    SYSTEM,
    clean,
    WORK as OLD,
    PUBLIC as LAST,
)
from scripts.local_translation_teacher import local_teacher
from scripts.test_pictured_action_binding import WORK, PUBLIC
from scripts.train_causal_glyph_writer import sha, record
from scripts.train_span_choice import gpu_owner


def run():
    old = OLD / "teacher.jsonl"
    assert sha(old) == json.loads((LAST / "teacher.json").read_text())["replies_sha"]
    initial = [json.loads(s) for s in old.read_text().splitlines()]
    cases = sample("train", 100000)
    assert [c["key"] for c in cases[: len(initial)]] == [r["key"] for r in initial]
    path = WORK / "book_extra.jsonl"
    extra = (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )
    remaining = cases[len(initial) :]
    if [c["key"] for c in remaining[: len(extra)]] != [r["key"] for r in extra]:
        raise ValueError("Stored book ordering changed")
    with local_teacher(MODEL, WORK) as (ask, digest, size), path.open("a") as f:
        assert digest == json.loads((LAST / "teacher.json").read_text())["sha"]
        for c in remaining[len(extra) :]:
            user = (
                c["page"]
                + "\n"
                + "\n".join(q["zh"] + " 答案：" + q["refs"][0] for q in c["qs"])
            )
            r = ask(
                SYSTEM,
                dict(role="user", content=user),
                think=False,
                options=dict(seed=241109, temperature=0, num_ctx=2048, num_predict=96),
            )
            clauses = (
                clean(r["text"], [q["refs"][0] for q in c["qs"]])
                if r["stop"] == "stop"
                else None
            )
            f.write(
                json.dumps(
                    dict(key=c["key"], clauses=clauses, result=r), ensure_ascii=False
                )
                + "\n"
            )
            f.flush()
    extra = [json.loads(s) for s in path.read_text().splitlines()]
    record(
        PUBLIC / "book_targets.json",
        dict(
            teacher=MODEL,
            manifest=digest,
            initial_sha=sha(old),
            extra_sha=sha(path),
            initial=len(initial),
            extra=len(extra),
            accepted=sum(bool(r["clauses"]) for r in initial + extra),
            teacher_seconds_added=sum(r["result"]["duration_ns"] for r in extra) * 1e-9,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
