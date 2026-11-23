"""Independently recount physical responses, edited-page contrasts and paired uncertainty."""

import gzip
import json
from collections import defaultdict

import numpy as np
import torch

from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.optical_role_reader import embeddings, read_many, restore
from scripts.prepare_fixed_actions import PUBLIC, WORK, read, sha, record
from scripts.train_optical_role_adapter import BASE, WORK as FIT
from scripts.train_typeface_writer import Face, FACES


def coord(row):
    return tuple(row[k] for k in ("scene", "lang", "order", "world", "h"))


def audit():
    items = read(WORK / "inputs.json")
    assert sha(WORK / "inputs.json") == read(PUBLIC / "inputs.json")["inputs"]
    refs = {coord(r): r for r in items}
    assert len(refs) == len(items) == 512
    face = Face(FACES[0])
    counts = {}
    matrices = {}
    ledger = []
    checked = 0
    bank = InkBank()
    replay = []
    for arm in ("parent", "independent", "aligned", "generic", "text"):
        rows = read(WORK / (arm + ".json"))
        assert sha(WORK / (arm + ".json")) == read(PUBLIC / (arm + ".json"))["replies"]
        assert len(rows) == len(refs) and {coord(r) for r in rows} == set(refs)
        groups = defaultdict(dict)
        for row in rows:
            k = coord(row)
            r = refs[k]
            a, b = r["answer"], r["other"]
            if arm == "text":
                answer = row["reply"]
            else:
                answer = bytes.fromhex(row["reply"])
                page = face(r["page"])
                a, b = face(a), face(b)
                checked += 1
                assert (
                    0 < len(answer) <= 1024
                    and len(answer) % 128 == 0
                    and any(
                        page.startswith(answer, j) for j in range(0, len(page), 128)
                    )
                )
                if arm == "aligned":
                    replay.append(
                        ((bank.encode(page), bank.encode(face(r["question"]))), answer)
                    )
            assert (answer == a and row["ended"]) == row["correct"]
            assert (answer == b and row["ended"]) == row["rival"]
            groups[r["group"], r["lang"]][k] = row
            ledger.append(dict(reader=arm, **row))
        for group, data in groups.items():
            names = sorted({k[0] for k in data})

            def array(fn):
                return np.array(
                    [
                        [
                            [
                                [fn(data[s, group[1], o, w, h]) for h in (0, 1)]
                                for w in (0, 1)
                            ]
                            for o in (0, 1)
                        ]
                        for s in names
                    ]
                )

            exact = array(lambda r: r["correct"])
            output = array(lambda r: r["reply"])
            repeats = output[:, :, 0, :] == output[:, :, 1, :]
            measures = dict(
                exact=exact.sum((1, 2, 3)),
                original=exact[:, 0].all((1, 2)),
                reversed=exact[:, 1].all((1, 2)),
                both_orders=exact.all((1, 2, 3)),
                unchanged=repeats.sum((1, 2)),
            )
            if arm != "text":
                margin = array(lambda r: r["source_minus_rival"])
                wins = margin > 0
                measures.update(
                    preferred=wins.sum((1, 2, 3)),
                    restricted_original=wins[:, 0].all((1, 2)),
                    restricted_both=wins.all((1, 2, 3)),
                    prefer_but_wrong=(wins & ~exact).sum((1, 2, 3)),
                    rival=array(lambda r: r["rival"]).sum((1, 2, 3)),
                    ties=(margin == 0).sum((1, 2, 3)),
                )
            matrices[(arm, *group)] = measures
            counts["/".join((arm, *group))] = {
                k: int(v.sum()) for k, v in measures.items()
            }
    differences = {}
    for work in ("three", "water"):
        for lang in ("zh", "en"):
            chosen = matrices["aligned", work, lang]
            for arm in ("parent", "independent"):
                for metric, denom in (
                    ("original", 1),
                    ("both_orders", 1),
                    ("exact", 8),
                ):
                    d = (
                        chosen[metric].astype(float) - matrices[arm, work, lang][metric]
                    ) / denom
                    ix = np.random.default_rng(241172).integers(
                        len(d), size=(10000, len(d))
                    )
                    differences["/".join((work, lang, arm, metric))] = [
                        float(d.mean()),
                        *np.quantile(d[ix].mean(1), [0.025, 0.975]).tolist(),
                    ]
    model = restore(BASE, "cpu", FIT / "aligned/adapter.safetensors")
    emb = embeddings(model, bank)
    identical = 0
    mismatches = []
    for i in range(0, len(replay), 24):
        batch = replay[i : i + 24]
        replies = read_many(model, emb, [p for p, _ in batch])
        for offset, (reply, (_, target)) in enumerate(zip(replies, batch, strict=True)):
            out = b"".join(bank.cells[j] for j in reply)
            if out == target:
                identical += 1
            else:
                r = items[i + offset]
                mismatches.append(
                    dict(
                        key=list(coord(r)),
                        cpu=out.hex(),
                        gpu=target.hex(),
                        cpu_correct=out == face(r["answer"]),
                        gpu_correct=target == face(r["answer"]),
                    )
                )
    record(PUBLIC / "cpu_differences.json", mismatches)
    dest = PUBLIC / "replies.jsonl.gz"
    with (
        dest.open("wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
    ):
        for r in ledger:
            gz.write(
                (
                    json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
    record(
        PUBLIC / "result.json",
        dict(
            counts=counts,
            differences=differences,
            physical=checked,
            cpu_replayed=len(replay),
            cpu_identical=identical,
            ledger_sha=sha(dest),
            eligible={
                l: all(
                    differences["/".join((g, l, "independent", "original"))][1] > 0
                    for g in ("three", "water")
                )
                for l in ("zh", "en")
            },
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    audit()
