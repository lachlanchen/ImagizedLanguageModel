"""Measure stopped-answer likelihoods and read generated replies with arm labels hidden."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json

import torch

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.audit_book_sequence_probability import score_batch
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.local_translation_teacher import local_teacher
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.role_change_corpus import load, target
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_pointing_causal_writer import inputs
from scripts.train_role_change_writer import PARENT, examples
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def sample(data):
    face = Face(FACES[0])
    for i, scene in enumerate(data["fresh"]):
        for offset, (key, w, lang, rev, p, q) in enumerate(examples(scene, face)):
            if rev == bool(i % 2):
                qs = scene["qs"][lang][::-1] if rev else scene["qs"][lang]
                yield (
                    i * 8 + offset,
                    key,
                    scene["pages"][w],
                    qs,
                    target(scene, w, rev),
                )


def qwen():
    rows = []
    model = "qwen3:8b-q8_0"
    with local_teacher(model, WORK) as (ask, digest, size):
        for ix, key, page, qs, _ in sample(load()):
            r = ask(
                "读短文，按问题先后用中文写两个完整、简短的分句。每个分句都写明人物和对应的事情。只写答案。",
                dict(role="user", content=page + "\n" + "\n".join(qs)),
                think=False,
                options=dict(temperature=0, seed=242110, num_ctx=2048, num_predict=96),
            )
            rows.append(dict(key=key, reply=r))
    record(WORK / "qwen.json", rows)
    record(
        PUBLIC / "text_control.json",
        dict(
            model=model,
            manifest=digest,
            size=size,
            rows=len(rows),
            sha256=sha(WORK / "qwen.json"),
        ),
    )


@torch.inference_mode()
def likelihood():
    data = load()
    face = Face(FACES[0])
    report = {}
    for arm, folder in (
        ("parent", PARENT),
        ("independent", WORK / "independent/page"),
        ("aligned", WORK / "aligned/page"),
    ):
        net = PointingCausalWriter.restore(folder, "cuda").eval()
        result = {}
        for group in ("fresh", "duie"):
            rows = []
            for c in data[group]:
                for key, w, lang, rev, p, q in examples(c, face):
                    rows.extend(
                        Row(group, key, p, q, face(target(c, v, rev)), b"")
                        for v in (w, 1 - w)
                    )
            read, alphabet, pairs, ys = inputs(net.writer, rows)
            # Both answers contain precisely the same glyphs. If one needs an
            # unavailable output, both have zero probability and neither wins.
            scores = [None] * len(rows)
            eligible = [
                j
                for j, ((source, _), answer) in enumerate(zip(pairs, ys, strict=True))
                if all(k < len(alphabet) or k in source for k in answer)
            ]
            for i in range(0, len(eligible), 16):
                ids = eligible[i : i + 16]
                values, _ = score_batch(
                    net, read, alphabet, [pairs[j] for j in ids], [ys[j] for j in ids]
                )
                for j, value in zip(ids, values, strict=True):
                    scores[j] = value
            detail = {}
            for r, a, b in zip(rows[::2], scores[::2], scores[1::2], strict=True):
                assert (a is None) == (b is None)
                detail[r.key] = a - b if a is not None else None
            if len(detail) != len(rows) // 2:
                raise ValueError("Repeated assessment key")
            path = WORK / f"margins_{arm}_{group}.json"
            record(path, detail)
            by = defaultdict(list)
            for k, v in detail.items():
                name, w, lang, rev = k.split("/")
                by[name, lang].append(v)
            result[group] = dict(
                requests=len(detail),
                unsupported=sum(v is None for v in detail.values()),
                correct=sum(v is not None and v > 0 for v in detail.values()),
                complete=sum(
                    all(v is not None and v > 0 for v in values)
                    for values in by.values()
                ),
                contrasts=len(by),
                mean=sum(v for v in detail.values() if v is not None)
                / max(1, sum(v is not None for v in detail.values())),
                sha256=sha(path),
            )
        report[arm] = result
        del net
    record(PUBLIC / "likelihood.json", report)


def prepare():
    data = load()
    face = Face(FACES[0])
    chars = set("。？！□，、；：｜")
    for scenes in data.values():
        for c in scenes:
            chars.update(
                "".join(
                    c["pages"]
                    + c["actions"]
                    + c["names"]
                    + c["qs"]["zh"]
                    + c["qs"]["en"]
                )
            )
    mapping = defaultdict(set)
    for ch in chars:
        if ch.isspace() or ord(ch) in face.coverage:
            mapping[face(ch)].add(" " if ch.isspace() else ch)
    scores = json.loads((PUBLIC / "scores.json").read_text())
    streams = {}
    for arm in ("previous", "independent", "aligned"):
        path = WORK / f"fresh_{arm}.npz"
        assert sha(path) == scores[arm]["fresh"]["sha256"]
        streams[arm] = load_streams(path)
    assert (
        sha(WORK / "qwen.json")
        == json.loads((PUBLIC / "text_control.json").read_text())["sha256"]
    )
    texts = {r["key"]: r["reply"] for r in json.loads((WORK / "qwen.json").read_text())}
    blind = []
    keys = []
    for ix, key, page, qs, expected in sample(data):
        choices = {}
        for arm, (ys, ends) in streams.items():
            decoded = "".join(
                next(iter(mapping[x])) if len(mapping[x]) == 1 else "□"
                for x in in_cells(ys[ix])
            )
            choices[arm] = (decoded, ends[ix])
        choices["qwen"] = (texts[key]["text"], texts[key]["stop"] == "stop")
        unique = sorted(
            set(choices.values()),
            key=lambda v: sha256(f"role-change/{key}/{v}".encode()).digest(),
        )
        letters = {v: chr(65 + j) for j, v in enumerate(unique)}
        blind.append(
            dict(
                key=key,
                page=page,
                questions=qs,
                expected=expected,
                choices={letters[v]: dict(text=v[0], stopped=v[1]) for v in unique},
            )
        )
        keys.append({a: letters[v] for a, v in choices.items()})
    record(WORK / "blind.json", blind)
    record(WORK / "keys.json", keys)
    record(
        PUBLIC / "blind_binding.json",
        dict(
            blind=sha(WORK / "blind.json"),
            keys=sha(WORK / "keys.json"),
            cases=len(keys),
        ),
    )


def score():
    binding = json.loads((PUBLIC / "blind_binding.json").read_text())
    assert all(sha(WORK / (k + ".json")) == binding[k] for k in ("blind", "keys"))
    blind, keys = [
        json.loads((WORK / (k + ".json")).read_text()) for k in ("blind", "keys")
    ]
    reading = json.loads((PUBLIC / "reading.json").read_text())
    counts = defaultdict(Counter)
    by = defaultdict(list)
    out = []
    for row, key, letters in zip(blind, keys, reading["credits"], strict=True):
        assert all(
            row["choices"][letter]["stopped"]
            and "□" not in row["choices"][letter]["text"]
            for letter in letters
        )
        answers = {}
        for arm, letter in key.items():
            ok = letter in letters
            counts[arm]["complete"] += ok
            counts[arm]["requests"] += 1
            by[arm, row["key"].split("/")[0]].append(ok)
            answers[arm] = dict(**row["choices"][letter], complete=ok)
        out.append(
            {k: v for k, v in row.items() if k != "choices"} | dict(answers=answers)
        )
    for (arm, _), flags in by.items():
        counts[arm]["four_correct"] += all(flags)
    record(PUBLIC / "reviewed_answers.json", out)
    record(
        PUBLIC / "reading_result.json",
        dict(
            scores=counts, reading=sha(PUBLIC / "reading.json"), blind=binding["blind"]
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["likelihood", "qwen", "prepare", "score"])
    action = parser.parse_args().action
    if action in ("qwen", "likelihood"):
        with gpu_owner():
            torch.set_num_threads(2)
            (qwen if action == "qwen" else likelihood)()
    else:
        (prepare if action == "prepare" else score)()
