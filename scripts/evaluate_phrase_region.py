"""Measure complete free replies to pairs of pictured questions and factual interventions."""

from collections import defaultdict
import numpy as np
import torch
from scripts.train_phrase_region import MODEL
from scripts.prepare_phrase_confirmation import WORK, PUB
from scripts.prepare_region_writer import MODEL as OLD, WORK as DATA, PUB as RECORD, Ink
from scripts.prepare_open_exam import read, dump, sha
from scripts.evaluate_region_writer import run_group, changed, offpage, e2e_scorer
from scripts.audit_region_writer import recount
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.write_region_writer import model, phrase_model
from scripts.train_paired_evidence import gpu_owner
from scripts.probe_region_writer_transfer import transfer


def measurements(rows, values, ink, score):
    return [
        recount({**r, "task": "catalogue"} if r["task"] == "exam" else r, v, ink, score)
        for r, v in zip(rows, values, strict=True)
    ]


def group_counts(rows, values, edits, altered):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["task"]].append(i)
    result = {}
    for t, indices in groups.items():
        families = defaultdict(list)
        for i in indices:
            families[rows[i]["family"]].append(i)
        pairs = [ii for ii in families.values() if len(ii) == 2]
        group = {
            k: sum(values[i][k] for i in indices)
            for k in (
                "exact",
                "bitmap_exact",
                "novel",
                "novel_exact",
                "fact",
                "stopped",
            )
        }
        e = [(r, a) for r, a in zip(edits, altered, strict=True) if r["task"] == t]
        group.update(
            n=len(indices),
            pairs=len(pairs),
            both_exact=sum(all(values[i]["bitmap_exact"] for i in ii) for ii in pairs),
            both_fact=sum(all(values[i]["fact"] for i in ii) for ii in pairs),
            edits=len(e),
            edit_pair_fact=sum(
                values[r["original"]]["fact"] and a["fact"] for r, a in e
            ),
            edit_pair_exact=sum(
                values[r["original"]]["bitmap_exact"] and a["bitmap_exact"]
                for r, a in e
            ),
        )
        result[t] = group
    return result


def intervals(rows, evaluations):
    groups = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        groups[r["task"]][r["family"]].append(i)
    result = {}
    for t, g in groups.items():
        res = {}
        draws = np.random.default_rng(4261125).integers(0, len(g), (5000, len(g)))
        for key in ("bitmap_exact", "fact", "novel_exact"):
            diff = [
                np.mean(
                    [
                        int(evaluations["phrase"][i][key])
                        - int(evaluations["single"][i][key])
                        for i in family
                    ]
                )
                for family in g.values()
            ]
            res[key] = np.percentile(
                np.array(diff)[draws].mean(1) * 100, [2.5, 97.5]
            ).tolist()
        result[t] = res
    return result


def main():
    ref = read(PUB / "confirmation.json")
    if (
        sha(WORK / "confirmation.json") != ref["sha256"]
        or sha(DATA / "dev-source.json")
        != read(RECORD / "corpus.json")["files"]["dev-source.json"]
    ):
        raise ValueError("Evaluation source changed")
    fresh = read(WORK / "confirmation.json")
    prior = read(DATA / "dev-source.json")
    external, excluded, norm = offpage()
    sets = {"confirmation": fresh, "development": prior}
    score = e2e_scorer()
    chars = read(OLD / "readback.json")
    palette = np.load(OLD / "palette.npy")
    results = {}
    for label, rows in sets.items():
        edits = changed(rows)
        all_measure = {}
        scores = {}
        with gpu_owner(), torch.inference_mode():
            torch.set_float32_matmul_precision("highest")
            selector = model("selector", "cuda")
            for arm in ("single", "phrase", "restricted"):
                ink = Ink(palette)
                if arm == "single" and label == "development":
                    z = unpack(DATA / "selected-predictions.json.gz")
                else:
                    net = (
                        model("selected", "cuda")
                        if arm == "single"
                        else phrase_model(MODEL, "cuda")
                    )
                    net.singleton_only = arm == "restricted"
                    z = {
                        k: run_group(
                            net, selector, "selected", rs, ink, chars, score, norm
                        )
                        for k, rs in (("original", rows), ("edited", edits))
                    }
                    if label == "development" and arm == "phrase":
                        z["coqa"] = run_group(
                            net, selector, "selected", external, ink, chars, score, norm
                        )
                packed(WORK / (label + "-" + arm + ".json.gz"), z)
                m = measurements(rows, z["original"], ink, score)
                e = measurements(edits, z["edited"], ink, score)
                all_measure[arm] = m
                scores[arm] = group_counts(rows, m, edits, e)
                if "coqa" in z:
                    scores[arm]["coqa"] = {
                        "tested": len(external),
                        "outside_geometry": excluded,
                        "exact": sum(r["exact"] for r in z["coqa"]),
                    }
                print(label, arm, scores[arm], flush=True)
        results[label] = dict(
            scores=scores, paired_family_bootstrap_pp=intervals(rows, all_measure)
        )
        dump(PUB / "answers.json", results)
    dump(
        PUB / "transfer.json",
        transfer([("phrase", "selected", lambda: phrase_model(MODEL, "cuda"))]),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
