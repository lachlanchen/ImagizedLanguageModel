"""Prepare and grade hidden-arm semantic readings on the newly reserved story pairs."""

import argparse
from statistics import median
from time import monotonic
from scripts.measure_writer_energy import snapshot, consumed
from collections import defaultdict
import json
from hashlib import sha256
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.inspect_role_change_writer import score as score_hidden
from scripts.local_translation_teacher import local_teacher
from scripts.role_change_corpus import target, load
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_individual_clause_writer import PUBLIC, WORK, new_stories
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def samples():
    for i, c in enumerate(new_stories()):
        a = i % 2
        for w in (0, 1):
            for j, lang in enumerate(("zh", "en")):
                yield c, w, lang, a, 8 * i + 4 * w + 2 * j, f"{c['key']}/{w}/{lang}/{a}"


def qwen():
    rows = []
    model = "qwen3:8b-q8_0"
    with local_teacher(model, WORK) as (ask, digest, size):
        system = "读短文，按问题先后用中文写两个完整、简短的分句。每个分句都写明人物和对应的事情。只写答案。"
        settings = dict(temperature=0, num_ctx=2048, num_predict=96, seed=242110)
        ask(
            system,
            dict(role="user", content="林方收信，丁波寄信。谁收信？谁寄信？"),
            think=False,
            options=settings,
        )
        latency = []
        first_energy = snapshot()
        for c, w, lang, a, ix, key in samples():
            qs = c["qs"][lang][::-1] if a else c["qs"][lang]
            start = monotonic()
            r = ask(
                system,
                dict(role="user", content=c["pages"][w] + "\n" + "\n".join(qs)),
                think=False,
                options=settings,
            )
            latency.append(monotonic() - start)
            rows.append(dict(key=key, reply=r))
        energy = consumed(first_energy, snapshot())
    record(WORK / "qwen.json", rows)
    record(
        PUBLIC / "text_control.json",
        dict(
            device_energy=energy,
            seconds_sum=sum(latency),
            seconds_median=median(latency),
            model=model,
            model_sha=digest,
            bytes=size,
            answers=len(rows),
            sha256=sha(WORK / "qwen.json"),
        ),
    )


def prepare():
    arms = ["parent", "independent", "aligned"]
    face = Face(FACES[0])
    chars = set("，。！？：；、□ ")
    for group in [load()["train"], new_stories()]:
        for c in group:
            chars.update(
                "".join(
                    c["pages"]
                    + c["qs"]["zh"]
                    + c["qs"]["en"]
                    + c["names"]
                    + c["actions"]
                )
            )
    for line in (WORK / "book_extra.jsonl").read_text().splitlines():
        chars.update("".join(json.loads(line)["clauses"] or []))
    mapping = defaultdict(set)
    for ch in chars:
        if ch.isspace() or ord(ch) in face.coverage:
            mapping[face(ch)].add(" " if ch.isspace() else ch)

    def decode(raw):
        return "".join(
            next(iter(mapping[b])) if len(mapping[b]) == 1 else "□"
            for b in in_cells(raw)
        )

    data = {}
    for arm in arms:
        scores = json.loads((PUBLIC / f"{arm}_evaluation.json").read_text())["groups"][
            "new"
        ]
        for route in ("separate", "direct") if arm == "parent" else ("separate",):
            path = WORK / f"{arm}_new_{route}.npz"
            assert sha(path) == scores[route]["images_sha"]
            data[arm, route] = load_streams(path)
    assert (
        sha(WORK / "qwen.json")
        == json.loads((PUBLIC / "text_control.json").read_text())["sha256"]
    )
    text = {r["key"]: r["reply"] for r in json.loads((WORK / "qwen.json").read_text())}
    blind = []
    keys = []
    for c, w, lang, a, ix, key in samples():
        choices = {}
        for (arm, route), (ys, ends) in data.items():
            if route == "direct":
                choices["parent_direct"] = decode(ys[ix + a]), ends[ix + a]
            else:
                choices[arm] = (
                    decode(ys[ix + a] + ys[ix + 1 - a]),
                    ends[ix + a] and ends[ix + 1 - a],
                )
        choices["qwen"] = (text[key]["text"], text[key]["stop"] == "stop")
        ordered = sorted(
            set(choices.values()),
            key=lambda v: sha256(f"clause/{key}/{v}".encode()).digest(),
        )
        labels = {v: chr(65 + j) for j, v in enumerate(ordered)}
        blind.append(
            dict(
                key=key,
                page=c["pages"][w],
                questions=c["qs"][lang][::-1] if a else c["qs"][lang],
                expected=target(c, w, a),
                choices={labels[v]: dict(text=v[0], stopped=v[1]) for v in ordered},
            )
        )
        keys.append({name: labels[v] for name, v in choices.items()})
    record(WORK / "blind.json", blind)
    record(WORK / "keys.json", keys)
    record(
        PUBLIC / "blind_binding.json",
        dict(
            blind=sha(WORK / "blind.json"),
            keys=sha(WORK / "keys.json"),
            cases=len(blind),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["qwen", "prepare", "score"])
    a = ap.parse_args()
    if a.action == "qwen":
        with gpu_owner():
            qwen()
    elif a.action == "prepare":
        prepare()
    else:
        score_hidden(PUBLIC, WORK)
