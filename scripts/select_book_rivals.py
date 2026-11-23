"""Select and adjudicate self-generated alternatives on human-annotated training pages."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from time import monotonic

from scripts.benchmark_interval_teacher import MODEL, DIGEST
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.local_translation_teacher import local_teacher
from scripts.mine_stopped_book_replies import PUBLIC, WORK, training_cases
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

CAP = 192
PROMPT = (
    "根据给出的原文，判断两个选项中哪一个单独就能正确完整地回答问题。"
    "同义或等价答案均可，包括简称。如果两者都对答 BOTH；如果都不对或不确定答 UNCERTAIN。"
    "如果仅一个正确，答 A 或 B。只输出一个标签。"
)


def select():
    items = training_cases()
    mining = json.loads((PUBLIC / "mining.json").read_text())
    face = Face(FACES[0])
    pool = defaultdict(list)
    rejected = Counter()
    for name, h in mining["files"].items():
        receipt = WORK / "mining" / name
        assert sha(receipt) == h
        m = json.loads(receipt.read_text())
        path = receipt.with_suffix(".npz")
        assert sha(path) == m["sha256"]
        ys, ends = load_streams(path)
        for (key, lang), y, end in zip(m["ids"], ys, ends, strict=True):
            row, q = items[(key, lang)]
            refs = q["refs"]
            if not end or not y or y in [face(t) for t in refs]:
                rejected["exact_empty_or_capped"] += 1
                continue
            src = row["page"]
            width = len(y) // 128
            cells = in_cells(face(src))
            first = y[:128]
            spans = {
                src[i : i + width]
                for i, c in enumerate(cells)
                if c == first and b"".join(cells[i : i + width]) == y
            }
            if len(spans) != 1:
                rejected["not_unique_source_spelling"] += 1
                continue
            alt = spans.pop()
            if any(alt in r or r in alt for r in refs) or not any(
                c.isalnum() for c in alt
            ):
                rejected["possibly_partial_or_punctuation"] += 1
                continue
            evidence = [
                d["id"] for d in row["questions"] if d["id"] != key and alt in d["refs"]
            ]
            kind = "other_answer" if evidence else "other_span"
            rank = sha256((key + "/" + lang + "/book-rival-1").encode()).hexdigest()
            pool[lang, kind].append(
                (
                    rank,
                    dict(
                        id=key,
                        lang=lang,
                        kind=kind,
                        wrong=alt,
                        refs=refs,
                        other_ids=evidence,
                        pixels=sha256(y).hexdigest(),
                    ),
                )
            )
    selected = [item for group in sorted(pool) for _, item in sorted(pool[group])[:CAP]]
    report = dict(
        total=sum(map(len, pool.values())),
        eligible={str(k): len(v) for k, v in pool.items()},
        rejected=dict(rejected),
        selected=len(selected),
        cap_per_group=CAP,
    )
    record(WORK / "proposed.json", selected)
    record(
        PUBLIC / "selection.json", dict(**report, proposals=sha(WORK / "proposed.json"))
    )
    return selected, items


def adjudicate():
    selected, items = select()
    folder = WORK / "adjudication"
    folder.mkdir(exist_ok=True)
    ledger = folder / "votes.jsonl"
    jobs = [(v, order) for v in selected for order in (0, 1)]
    votes = (
        [json.loads(s) for s in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    for r, (v, order) in zip(votes, jobs):
        assert (r["id"], r["lang"], r["order"], r["pixels"]) == (
            v["id"],
            v["lang"],
            order,
            v["pixels"],
        )
    if len(votes) > len(jobs):
        raise ValueError("Unexpected extra votes")
    start = monotonic()
    with local_teacher(MODEL, folder) as (ask, digest, size), ledger.open("a") as f:
        assert digest == DIGEST
        for v, order in jobs[len(votes) :]:
            row, q = items[v["id"], v["lang"]]
            a, b = v["refs"][0], v["wrong"]
            if order:
                a, b = b, a
            message = json.dumps(
                dict(passage=row["page"], question=q[v["lang"]], A=a, B=b),
                ensure_ascii=False,
            )
            reply = ask(
                PROMPT,
                dict(role="user", content=message),
                think=False,
                options=dict(
                    seed=84371,
                    temperature=0,
                    num_predict=12,
                    num_ctx=2048,
                    num_thread=2,
                ),
            )
            vote = dict(
                id=v["id"],
                lang=v["lang"],
                order=order,
                pixels=v["pixels"],
                text=reply["text"],
                stop=reply["stop"],
                duration_ns=reply["duration_ns"],
            )
            f.write(json.dumps(vote, ensure_ascii=False) + "\n")
            f.flush()
            votes.append(vote)
            if len(votes) % 128 == 0:
                print("teacher judgments", len(votes), flush=True)
    keep = []
    labels = Counter()
    for v, a, b in zip(selected, votes[::2], votes[1::2], strict=True):
        correct = (a["text"], b["text"]) == ("A", "B") and a["stop"] == b[
            "stop"
        ] == "stop"
        labels[v["lang"] + "/" + v["kind"] + "/" + str(correct)] += 1
        if correct:
            keep.append(v)
    record(WORK / "accepted.json", keep)
    record(
        PUBLIC / "adjudication.json",
        dict(
            proposed=len(selected),
            accepted=len(keep),
            groups=dict(labels),
            accepted_sha=sha(WORK / "accepted.json"),
            ledger_sha=sha(ledger),
            teacher=digest,
            teacher_bytes=size,
            requests=len(votes),
            teacher_seconds=sum(r["duration_ns"] for r in votes) / 1e9,
            session_seconds=monotonic() - start,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        adjudicate()
