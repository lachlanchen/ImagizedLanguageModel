"""Compare stopped generation on identical stories with a paired story bootstrap."""

from collections import defaultdict
import gzip
import json
import numpy as np
from scripts.prepare_dialogue_ink import WORK, PUB, dump, sha
from scripts.evaluate_optical_passage_qa import normalized
from scripts.score_dialogue_ink_references import official_index


def compare(cold=False):
    evaluator, lookup, stories, raw, code = official_index()
    paths = [
        WORK / p / "decisions.json.gz"
        for p in ("evaluation", "word_code_evaluation", "subword_evaluation")
    ]
    names = ("direct", "word_code", "subword")
    if cold:
        paths = [WORK / "matched" / p / "decisions.json.gz" for p in ("ink", "byte")]
        names = ("ink", "byte")
    rows = [json.loads(gzip.decompress(p.read_bytes())) for p in paths]
    results = {}
    differences = {}
    seed = 64117
    for group in ("reserved", "dev"):
        samples = [defaultdict(list) for _ in rows]
        paired = defaultdict(list)
        for items in zip(*rows, strict=True):
            r = items[0]
            identity = (
                "group",
                "story",
                "index",
                "turn",
                "answer",
                "question",
                "history",
                "question_control",
                "passage_control",
            )
            if any(
                tuple(v[k] for k in identity) != tuple(r[k] for k in identity)
                for v in items
            ):
                raise ValueError("Nonidentical comparison subjects")
            if r["group"] != group:
                continue
            key = r["story"]
            strict = r["strict"]
            tag = ["all"] + (["strict"] if strict else [])
            official = group == "dev"
            if official:
                marker = lookup[
                    key, r["turn"], r["question"], r["history"], r["answer"]
                ]
                refs = evaluator.gold_data[marker]
                fields = [
                    " " + normalized(s) + " "
                    for s in (stories[key], r["question"], r["history"])
                ]
                if strict and all(
                    len(normalized(a).split()) >= 2
                    and all((" " + normalized(a) + " ") not in s for s in fields)
                    for a in refs
                ):
                    tag += ["all_references_absent"]
            for arm, run in enumerate(items):
                e = run["out"]["writer"]
                for cat in tag:
                    arr = [
                        int(e["exact"]),
                        float(e["f1"]),
                        int(e["stopped"]),
                        int(e["pixels"]),
                        int(run["out"]["exchanged_question"]["exact"]),
                        int(run["out"]["exchanged_passage"]["exact"]),
                    ]
                    if official:
                        stats = evaluator.compute_turn_score(*marker, e["text"])
                        arr += [
                            stats["em"] if e["stopped"] else 0,
                            stats["f1"] if e["stopped"] else 0,
                            int(
                                e["stopped"]
                                and any(
                                    normalized(a) == normalized(e["text"]) for a in refs
                                )
                            ),
                        ]
                    samples[arm][cat].append(arr)
            paired[key].append(
                int(items[0]["out"]["writer"]["exact"])
                - int(items[-1]["out"]["writer"]["exact"])
            )
        by_story = np.array([(sum(v), len(v)) for v in paired.values()])
        draws = np.random.default_rng(seed).integers(
            len(by_story), size=(5000, len(by_story))
        )
        sums = by_story[draws].sum(1)
        differences[group] = dict(
            stories=len(by_story),
            point=float(by_story[:, 0].sum() / by_story[:, 1].sum()),
            interval95=np.quantile(sums[:, 0] / sums[:, 1], [0.025, 0.975]).tolist(),
        )
        results[group] = {}
        for arm, rows_ in zip(names, samples):
            results[group][arm] = {}
            for k, vs in rows_.items():
                z = np.array(vs)
                s = z.sum(0)
                n = len(z)
                result = dict(
                    n=n,
                    exact=int(s[0]),
                    f1=s[1] / n,
                    stopped=int(s[2]),
                    verbatim=int(s[3]),
                    exchanged_question_exact=int(s[4]),
                    exchanged_passage_exact=int(s[5]),
                )
                if group == "dev":
                    result.update(
                        official_em=s[6] / n, official_f1=s[7] / n, any_exact=int(s[8])
                    )
                results[group][arm][k] = result
    name = "cold_comparison.json" if cold else "token_comparison.json"
    result = dict(
        scores=results,
        paired=differences,
        paired_arms=[names[0], names[-1]],
        bootstrap=dict(seed=seed, replicates=5000, unit="whole story"),
        decisions=list(map(sha, paths)),
        official_data=sha(raw),
        official_scorer=sha(code),
    )
    if cold:
        efficiency = {}
        for arm in names:
            curves = json.loads((PUB / f"{arm}_curve.json").read_text())
            meta = json.loads(
                (
                    WORK.parents[1] / "models/dialogue_surfaces" / arm / "weights.json"
                ).read_text()
            )
            efficiency[arm] = {}
            for cutoff in (10, 20, 25):
                first = next(
                    (
                        s
                        for s in ("100", "800", "2400", "8000")
                        if all(
                            100 * group[s]["writer_exact"] >= cutoff * group[s]["turns"]
                            for group in curves.values()
                        )
                    ),
                    None,
                )
                efficiency[arm][cutoff] = (
                    None
                    if first is None
                    else dict(
                        step=int(first),
                        joules=meta["preparation"]["energy"]["joules"]
                        + meta["checkpoints"][first]["energy"]["joules"],
                    )
                )
        result["first_measured_energy_at_quality"] = efficiency
    dump(PUB / name, result)
    print(differences)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold", action="store_true")
    compare(p.parse_args().cold)
