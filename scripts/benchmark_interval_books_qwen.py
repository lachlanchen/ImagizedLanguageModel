"""Score the local text-input reference on the same frozen 92 book questions."""

import json
from time import monotonic
from scripts.benchmark_interval_teacher import MODEL, SYSTEM, DIGEST
from scripts.local_translation_teacher import local_teacher
from scripts.prepare_bilingual_book_intervals import PUBLIC, WORK
from scripts.reserve_bilingual_interval_books import PAGES, CHECK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner


def run():
    pages = json.loads(PAGES.read_text())
    receipt = json.loads(CHECK.read_text())
    assert sha(PAGES) == receipt["pages_sha256"]
    dest = WORK / "qwen"
    dest.mkdir(exist_ok=True)
    ledger = dest / "answers.jsonl"
    jobs = [
        (k, i, lang)
        for k in receipt["eligible"]
        for i in range(2)
        for lang in ("zh", "en")
    ]
    rows = (
        [json.loads(line) for line in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    assert [tuple(r["job"]) for r in rows] == jobs[: len(rows)]
    with local_teacher(MODEL, dest) as (ask, digest, size), ledger.open("a") as f:
        assert digest == DIGEST
        opts = dict(
            seed=241179, temperature=0, num_predict=80, num_ctx=2048, num_thread=2
        )
        ask(
            SYSTEM,
            dict(role="user", content="材料：小河流入大海。\n问题：小河流入哪里？"),
            think=False,
            options=opts,
        )
        for key, i, lang in jobs[len(rows) :]:
            page = pages[key]
            q = page["questions"][i]
            start = monotonic()
            reply = ask(
                SYSTEM,
                dict(role="user", content=f"材料：{page['page']}\n问题：{q[lang]}"),
                think=False,
                options=opts,
            )
            reply.update(job=[key, i, lang], seconds=monotonic() - start)
            f.write(json.dumps(reply, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(reply)
    results = {}
    for register in ("classical", "modern"):
        flags = {lang: [] for lang in ("zh", "en")}
        for row in rows:
            key, i, lang = row["job"]
            if pages[key]["register"] == register:
                flags[lang].append(
                    row["stop"] == "stop"
                    and row["text"] in pages[key]["questions"][i]["refs"]
                )
        both = [a and b for a, b in zip(flags["zh"], flags["en"], strict=True)]
        results[register] = dict(
            zh=sum(flags["zh"]),
            en=sum(flags["en"]),
            n=len(both),
            four=sum(a and b for a, b in zip(both[::2], both[1::2], strict=True)),
        )
    record(
        PUBLIC / "text_baseline.json",
        dict(
            scores=results,
            model=MODEL,
            sha256=digest,
            weight_bytes=size,
            seconds=sum(r["seconds"] for r in rows),
            answers_sha256=sha(ledger),
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
