"""Audit saved answer pixels, group denominators and correction duplicates; export Figure 72 samples."""

from hashlib import sha256
import numpy as np
from scripts.evaluate_paired_student import read_streams
from scripts.evaluate_independent_entry_reader import views
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import write_json
from scripts.train_independent_entry_reader import ARMS, MODELS, OUT, corpus, load, sha


def run():
    source = corpus("development")
    metrics = load(OUT / "evaluation.json")["conditions"]
    report = {}
    examples = MODELS / "examples"
    examples.mkdir()
    strips = {}
    for view, qs, ids, k in views():
        targets = [source["answers"][i] for i in ids]
        unique = {}
        for i, (q, t) in enumerate(zip(qs, ids, strict=True)):
            unique.setdefault((t // 8, q, targets[i]), i)
        section = report[view] = {"unique": len(unique)}
        for arm in ("parent", *ARMS):
            stat = metrics[view][arm]
            if stat.get("archived"):
                continue
            assert (
                sha(OUT / "outputs" / (view + "_" + arm + ".npz")) == stat["stream_sha"]
            )
            predicted = [
                x.tobytes() for x in read_streams(view + "_" + arm, OUT / "outputs")
            ]
            good = np.array(
                [
                    s and p == t
                    for p, t, s in zip(predicted, targets, stat["stops"], strict=True)
                ]
            )
            assert (
                good.sum() == stat["exact"]
                and good.reshape(-1, k).all(1).sum() == stat["all_correct"]
            )
            assert np.array_equal(np.flatnonzero(good), stat["rows"])
            section[arm] = dict(
                exact_distinct=int(good[list(unique.values())].sum()), n=len(good)
            )
            if arm == "shared" and view == "independent":
                old = set(metrics[view]["parent"]["rows"])
                eligible = set(range(len(qs))) - set(
                    load(OUT / "review.json")["same_two_requested_items_clear_except"]
                )
                success = next(i for i in sorted(eligible) if good[i] and i not in old)
                failure = next(i for i in sorted(eligible) if not good[i])
                for label, i in (("gained", success), ("missed", failure)):
                    card = source["cards"][ids[i] // 8]
                    page(card, width=24).save(examples / (label + "_card.png"))
                    for key, raw in (
                        ("question", qs[i]),
                        ("answer", predicted[i]),
                        ("reference", targets[i]),
                    ):
                        name = label + "_" + key
                        img = page(raw, width=min(32, len(raw) // 128))
                        img.save(examples / (name + ".png"))
                        if key != "reference" or label == "missed":
                            img.save(OUT / (name + ".png"))
                            strips[name] = dict(
                                cells=len(raw) // 128,
                                cols=min(32, len(raw) // 128),
                                row=i,
                                sha=sha(OUT / (name + ".png")),
                            )
                    section[label] = dict(
                        index=i,
                        row=int(ids[i]),
                        question_cells=len(qs[i]) // 128,
                        card_cells=len(card) // 128,
                        sha=sha256(predicted[i]).hexdigest(),
                    )
    names = ("parent", "shared", "candidate")
    table = [["Question set", "Requests", "Previous", "Shared", "Candidate-specific"]]
    for view, label in (
        ("independent", "Independent writer"),
        ("before", "Cue before field"),
        ("correction", "Correction (distinct)"),
        ("reference", "Former or latter"),
        ("simple", "Original simple"),
    ):
        if view == "correction":
            vals = [report[view][m]["exact_distinct"] for m in names]
            n = report[view]["unique"]
        else:
            vals = [metrics[view][m]["exact"] for m in names]
            n = metrics[view]["shared"]["n"]
        table.append([label, f"{n:,}", *[f"{v:,}" for v in vals]])
    write_json(OUT / "recount.json", report)
    write_json(
        OUT / "figure.json",
        dict(
            table=table,
            series=[
                [round(x, 5) for x in load(OUT / (m + "_fit.json"))["loss"]]
                for m in ARMS
            ],
            strips=strips,
        ),
    )


if __name__ == "__main__":
    run()
