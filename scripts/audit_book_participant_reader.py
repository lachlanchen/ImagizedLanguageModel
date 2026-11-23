"""Recount the actual emitted ink and estimate passage-paired differences by source work."""

import json

import numpy as np
from scipy.stats import binomtest
from scripts.prepare_book_participant_questions import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES


def read(path):
    return json.loads(path.read_text())


def run():
    source = read(PUBLIC / "sources.json")
    for split, spec in source.items():
        assert sha(WORK / (split + ".json")) == spec["sha256"]
    rows = read(WORK / "reserved.json")
    learn = read(WORK / "learn.json")
    assert {r["work"] for r in rows}.isdisjoint(r["work"] for r in learn)
    assert {r["page_sha"] for r in rows}.isdisjoint(r["page_sha"] for r in learn)
    fits = [read(PUBLIC / f"{a}_fit.json") for a in ("aligned", "independent")]
    for k in ("initial", "updates", "pictured_questions", "parameters"):
        assert fits[0][k] == fits[1][k]
    face = Face(FACES[0])
    work = np.array([r["work"] for r in rows])
    rng = np.random.default_rng(241127)
    samples = []
    for k in sorted(set(work)):
        ix = np.flatnonzero(work == k)
        samples.append(rng.choice(ix, (10000, len(ix)), replace=True))
    boot = np.concatenate(samples, axis=1)
    methods = {}
    flags = {}
    for arm in ("aligned", "independent", "literal"):
        path = WORK / f"{arm}_reserved_responses.json"
        assert sha(path) == read(PUBLIC / f"{arm}_reserved.json")["sha256"]
        out = read(path)
        for m in ("restricted", "open"):
            flags[arm, m] = np.zeros((len(rows), 4), bool)
        for i, (r, y) in enumerate(zip(rows, out, strict=True)):
            assert r["page_sha"] == y["key"]
            p = face(r["page"])
            for j, reply in enumerate(y["replies"]):
                gold = face(r["names"][j % 2])
                image = bytes.fromhex(reply["image"])
                assert len(image) % 128 == 0 and 0 < len(image) <= 1024
                assert any(p.startswith(image, k) for k in range(0, len(p), 128))
                ok = image == gold
                assert (
                    ok == reply["open"] and (reply["margin"] > 0) == reply["restricted"]
                )
                flags[arm, "open"][i, j] = ok
                flags[arm, "restricted"][i, j] = reply["restricted"]
    path = WORK / "person_proposal_replies.json"
    assert sha(path) == read(PUBLIC / "person_reserved.json")["sha256"]
    proposals = read(path)
    for a in ("aligned", "independent"):
        out = []
        for r, case in zip(rows, proposals, strict=True):
            assert r["page_sha"] == case["key"]
            current = []
            for j, reply in enumerate(case["replies"]):
                gold = face(r["names"][j % 2])
                actual = bytes.fromhex(reply["images"][a])
                bit = actual == gold
                assert bit == reply["correct"][a] and (
                    not bit or case["reference_rank"][j % 2]
                )
                p = face(r["page"])
                assert any(p.startswith(actual, k) for k in range(0, len(p), 128))
                current.append(bit)
            out.append(current)
        flags[a, "proposed"] = np.array(out)

    def diff(x, y):
        d = x.astype(float) - y.astype(float)
        return dict(
            difference=float(d.mean()),
            ci95=np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist(),
        )

    for h, lang in enumerate(("zh", "en")):
        sl = slice(2 * h, 2 * h + 2)
        correct = flags["aligned", "restricted"][:, sl]
        switched_right = int(correct.all(1).sum())
        switched_wrong = int((~correct).all(1).sum())
        methods[lang] = dict(
            both_correct=switched_right,
            both_wrong=switched_wrong,
            exact_two_sided_switch_p=binomtest(
                switched_right, switched_right + switched_wrong
            ).pvalue,
            restricted_comparison={
                a: diff(correct.all(1), flags[a, "restricted"][:, sl].all(1))
                for a in ("independent", "literal")
            },
            autonomous_comparison={
                f"{a}_{m}": diff(
                    flags["aligned", "proposed"][:, sl].mean(1),
                    flags[a, m][:, sl].mean(1),
                )
                for a, m in (("aligned", "open"), ("independent", "proposed"))
            },
        )
    record(
        PUBLIC / "paired_audit.json",
        dict(
            resamples=10000,
            unit="passage within work",
            question_pairs=len(rows) * 2,
            verified_free_images=5 * len(rows) * 4,
            results=methods,
        ),
    )


if __name__ == "__main__":
    run()
