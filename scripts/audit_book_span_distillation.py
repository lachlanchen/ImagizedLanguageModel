"""Score original output bitmaps and compare readers by paired passage resampling."""

import json

import numpy as np

from scripts.prepare_book_span_distillation import LAST, PREVIOUS, PUBLIC, WORK
from scripts.read_book_participant import answer, restore
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_typeface_writer import Face, FACES
import torch


def read(path):
    return json.loads(path.read_text())


def verified(path, receipt):
    assert sha(path) == read(receipt)["sha256"], path
    return read(path)


def audit():
    pages = read(LAST / "reserved.json")
    assert (
        sha(LAST / "reserved.json")
        == read(PREVIOUS / "sources.json")["reserved"]["sha256"]
    )
    pairs = verified(WORK / "pairs.json", PUBLIC / "corpus.json")
    assert {r["page"] for r in pairs}.isdisjoint(r["page"] for r in pages)
    fits = [read(PUBLIC / f"{a}_fit.json") for a in ("matched", "exchanged", "human")]
    for key in ("initial", "parameters", "epochs", "updates", "question_presentations"):
        assert all(fit[key] == fits[0][key] for fit in fits[1:])
    with np.load(WORK / "teacher_targets.npz") as cache:
        assert (
            sha(WORK / "teacher_targets.npz")
            == read(PUBLIC / "teacher_learn.json")["sha256"]
        )
        assert np.allclose(cache["prob"].sum(1) + cache["residual"], 1, atol=1e-6)
        assert len(cache["prob"]) == 2 * len(pairs)
        assert np.isfinite(cache["prob"]).all()
    face = Face(FACES[0])
    checked = 0
    results = {}
    flags = {}
    teacher = verified(WORK / "teacher_reserved.json", PUBLIC / "teacher_reserved.json")
    release = ROOT / "models/book_span_distillation/reader.safetensors"
    assert sha(release) == fits[0]["weights"]
    model = restore(release, "cpu")
    cpu_matches = []
    for arm in ("original", "matched", "exchanged", "human", "teacher"):
        if arm == "original":
            detail = verified(
                LAST / "aligned_reserved_responses.json",
                PREVIOUS / "aligned_reserved.json",
            )
        elif arm != "teacher":
            detail = verified(
                WORK / f"{arm}_responses.json", PUBLIC / f"{arm}_reserved.json"
            )
        else:
            assert len(teacher) == 4 * len(pages)
        modes = (
            ["open", "proposed"]
            if arm in ("matched", "exchanged", "human")
            else ["open"]
        )
        for mode in modes:
            scores = []
            for i, row in enumerate(pages):
                if arm != "teacher":
                    assert row["page_sha"] == detail[i]["key"]
                p = face(row["page"])
                current = []
                for j in range(4):
                    if arm == "teacher":
                        image = bytes.fromhex(teacher[i * 4 + j])
                    else:
                        reply = detail[i]["replies"][j]
                        image = bytes.fromhex(
                            reply["image" if mode == "open" else "proposed_image"]
                        )
                    assert (
                        0 < len(image) <= 1024
                        and len(image) % 128 == 0
                        and any(p.startswith(image, k) for k in range(0, len(p), 128))
                    )
                    match = image == face(row["names"][j % 2])
                    if arm != "teacher":
                        assert match == reply[mode]
                    if arm == "matched" and mode == "open" and i % 7 == 0:
                        q = row["questions"][j % 2][("zh", "en")[j // 2]]
                        cpu_matches.append(answer(model, p, face(q)) == image)
                    checked += 1
                    current.append(match)
                scores.append(current)
            flags[arm, mode] = np.array(scores, dtype=float)
            results[arm + "_" + mode] = {
                lang: {"individual": int(m.sum()), "both": int(m.all(1).sum())}
                for h, lang in enumerate(("zh", "en"))
                for m in [flags[arm, mode][:, 2 * h : 2 * h + 2]]
            }
    works = np.array([r["work"] for r in pages])
    rng = np.random.default_rng(241127)
    boot = np.concatenate(
        [
            rng.choice(ids, (10000, len(ids)))
            for w in sorted(set(works))
            for ids in [np.flatnonzero(works == w)]
        ],
        axis=1,
    )
    comparisons = {}
    for h, lang in enumerate(("zh", "en")):
        for comparator in ("exchanged", "original", "human"):
            delta = (flags["matched", "open"] - flags[comparator, "open"])[
                :, 2 * h : 2 * h + 2
            ].mean(1)
            comparisons[f"{lang}/{comparator}"] = dict(
                delta=float(delta.mean()),
                ci95=np.quantile(delta[boot].mean(1), [0.025, 0.975]).tolist(),
            )
    record(
        PUBLIC / "audit.json",
        dict(
            checked_bitmaps=checked,
            cpu_replayed=len(cpu_matches),
            cpu_identical=sum(cpu_matches),
            results=results,
            passage_bootstrap_resamples=10000,
            comparisons=comparisons,
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    audit()
