"""Read three classical works and the two reserved novels with an independent optical model."""

import argparse
from time import monotonic
import torch

from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.optical_role_reader import read_many, restore, embeddings
from scripts.evaluate_classical_argument_reader import populations as earlier
from scripts.evaluate_role_question_binding import scores, decode
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_classical_argument_reader import (
    WORK as CLASSICAL,
    PUBLIC as PREVIOUS,
)
from scripts.train_optical_role_adapter import BASE, PUBLIC, WORK, read, sha, record
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def cases():
    path = WORK / "guoyu.json"
    assert sha(path) == read(PUBLIC / "corpus.json")["test_sha"]
    for r in read(path):
        for w, p in enumerate(r["pages"]):
            for l in ("zh", "en"):
                for h, q in enumerate(r["qs"]):
                    yield dict(
                        group="guoyu",
                        key=r["key"],
                        world=w,
                        h=h,
                        lang=l,
                        page=p,
                        question=q[l],
                        answer=r["qs"][w ^ h]["answer"],
                        other=r["qs"][1 ^ h ^ w]["answer"],
                    )
    yield from (r for r in earlier() if r["group"] in ("zuo", "shishuo", "novels"))


@torch.inference_mode()
def evaluate(arm):
    items = list(cases())
    if arm == "text":
        items = [r for r in items if r["group"] == "guoyu"]
        detail, cost = decode("text", items, WORK)
        old = CLASSICAL / "text_responses.json"
        assert sha(old) == read(PREVIOUS / "text_scores.json")["response_sha"]
        detail.extend(read(old))
        cost["earlier_reference_sha"] = sha(old)
    else:
        parent = arm == "parent"
        fit = "aligned" if arm == "generic" else arm
        spec = read(PUBLIC / "corpus.json")
        assert all(sha(BASE / n) == h for n, h in spec["base"].items())
        adapter = None if parent else WORK / fit / "adapter.safetensors"
        if adapter:
            assert sha(adapter) == read(PUBLIC / (fit + "_fit.json"))["weights"]
        net = restore(BASE, "cuda", adapter)
        bank = InkBank()
        face = Face(FACES[0])
        encoded = [
            (
                bank.encode(face(r["page"])),
                bank.encode(
                    face(
                        ("誰？" if r["lang"] == "zh" else "Who?")
                        if arm == "generic"
                        else r["question"]
                    )
                ),
            )
            for r in items
        ]
        emb = embeddings(net, bank)
        detail = []
        energy = snapshot()
        t = monotonic()
        for i in range(0, len(encoded), 24):
            answers = read_many(net, emb, encoded[i : i + 24])
            for r, a in zip(items[i : i + 24], answers, strict=True):
                y = b"".join(bank.cells[j] for j in a)
                detail.append(
                    dict(
                        **{k: r[k] for k in ("group", "key", "world", "lang", "h")},
                        reply=y.hex(),
                        ended=True,
                        correct=y == face(r["answer"]),
                        rival=y == face(r["other"]),
                    )
                )
        cost = dict(
            seconds=monotonic() - t,
            energy=consumed(energy, snapshot()),
            base=spec["base"],
            adapter=sha(adapter) if adapter else None,
        )
        if parent:
            # The zero-update implementation must match the earlier teacher on
            # each already recorded classical output, rather than just its score.
            old = CLASSICAL / "optical_responses.json"
            assert sha(old) == read(PREVIOUS / "optical_scores.json")["response_sha"]
            coord = lambda r: (r["group"], r["key"], r["world"], r["lang"], r["h"])
            reference = {coord(r): r["reply"] for r in read(old)}
            compared = [
                r["reply"] == reference[coord(r)]
                for r in detail
                if coord(r) in reference
            ]
            cost["parent_identical"] = sum(compared)
            cost["parent_compared"] = len(compared)
    f = WORK / (arm + "_responses.json")
    record(f, detail)
    record(
        PUBLIC / (arm + "_scores.json"),
        dict(counts=scores(detail), response_sha=sha(f), **cost),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "arm", choices=("parent", "aligned", "independent", "generic", "text")
    )
    args = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        evaluate(args.arm)
