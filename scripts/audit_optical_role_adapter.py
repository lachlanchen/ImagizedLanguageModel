"""Recount independent answers, bootstrap by excerpt and replay the new work on CPU."""

from collections import defaultdict

import numpy as np
import torch

from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.optical_role_reader import embeddings, read_many, restore
from scripts.evaluate_optical_role_adapter import cases
from scripts.evaluate_role_question_binding import scores
from scripts.train_optical_role_adapter import BASE, PUBLIC, WORK, read, record, sha
from scripts.train_typeface_writer import FACES, Face


def coord(row):
    return tuple(row[k] for k in ("group", "lang", "key", "world", "h"))


def main():
    face = Face(FACES[0])
    source = {coord(r): r for r in cases()}
    fits = [read(PUBLIC / (a + "_fit.json")) for a in ("aligned", "independent")]
    for key in (
        "base",
        "initial",
        "source",
        "parameters",
        "trained",
        "updates",
        "questions",
    ):
        assert fits[0][key] == fits[1][key]
    bank = InkBank()
    replay, arrays = [], {}
    checked = 0
    for arm in ("aligned", "independent", "parent", "generic", "text"):
        path = WORK / (arm + "_responses.json")
        receipt = read(PUBLIC / (arm + "_scores.json"))
        assert sha(path) == receipt["response_sha"]
        rows = read(path)
        assert scores(rows) == receipt["counts"]
        keys = {k for k in source if arm != "text" or k[0] != "novels"}
        assert len(keys) == len(rows) and keys == {coord(r) for r in rows}
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            key = coord(row)
            r = source[key]
            y, a, b = row["reply"], r["answer"], r["other"]
            if arm != "text":
                y, a, b = bytes.fromhex(y), face(a), face(b)
                page = face(r["page"])
                assert 0 < len(y) <= 1024 and len(y) % 128 == 0
                assert any(page.startswith(y, j) for j in range(0, len(page), 128))
                checked += 1
                if arm == "aligned" and r["group"] == "guoyu":
                    replay.append(
                        ((bank.encode(page), bank.encode(face(r["question"]))), y)
                    )
            assert row["correct"] == (row["ended"] and y == a)
            assert row["rival"] == (row["ended"] and y == b)
            grouped[key[:2]][key[2]].append(row["correct"])
        for k, groups in grouped.items():
            arrays[(arm, *k)] = np.array(
                [groups[s] for s in sorted(groups)], dtype=bool
            )
    model = restore(BASE, "cpu", WORK / "aligned/adapter.safetensors")
    emb = embeddings(model, bank)
    identical = 0
    for start in range(0, len(replay), 12):
        batch = replay[start : start + 12]
        answers = read_many(model, emb, [r[0] for r in batch])
        identical += sum(
            b"".join(bank.cells[j] for j in answer) == expected
            for answer, (_, expected) in zip(answers, batch, strict=True)
        )
    contrasts = {}
    for group in ("guoyu", "zuo", "shishuo", "novels"):
        for lang in ("zh", "en"):
            m = arrays["aligned", group, lang]
            ix = np.random.default_rng(241140).integers(len(m), size=(10000, len(m)))
            for other in ("parent", "independent"):
                ref = arrays[other, group, lang]
                for metric, x, y in (
                    ("individual", m.mean(1), ref.mean(1)),
                    ("complete", m.all(1), ref.all(1)),
                ):
                    d = x.astype(float) - y
                    contrasts["/".join((group, lang, metric, other))] = dict(
                        delta=float(d.mean()),
                        ci95=np.quantile(d[ix].mean(1), [0.025, 0.975]).tolist(),
                    )
    record(
        PUBLIC / "audit.json",
        dict(
            images=checked,
            cpu_replays=len(replay),
            cpu_identical=identical,
            contrasts=contrasts,
        ),
    )
    assert identical == len(replay)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
