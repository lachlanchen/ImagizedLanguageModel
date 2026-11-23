"""Count exact stopped pictures, compare within passage, and export unedited examples."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import binomtest

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_interval_teacher import trials
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_page_evidence_writer import tensor
from scripts.interval_teacher_corpus import WORK, PUBLIC, PARENT
from scripts.train_causal_glyph_writer import sha, record

ARMS = ("parent", "teacher", "replay", "aligned", "distilled")


def compare(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    gains = int((a & ~b).sum())
    losses = int((~a & b).sum())
    delta = a.astype(int) - b
    rng = np.random.default_rng(241179)
    draws = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
    return dict(
        n=len(delta),
        gains=gains,
        losses=losses,
        difference=float(delta.mean()),
        paired_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
        paired_sign_p=binomtest(gains, gains + losses).pvalue
        if gains + losses
        else 1.0,
    )


def save_image(path, value):
    write_strip(tensor(value).numpy(), path)


def audit():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    groups = trials()
    passed = {}
    complete = {}
    streams = {}
    summary = {}
    for arm in ARMS:
        root = WORK / "evaluation" / arm
        result = json.loads((root / "scores.json").read_text())
        summary[arm] = result
        passed[arm] = {}
        complete[arm] = {}
        for name, score in result["scores"].items():
            archive = root / (name + ".npz")
            if sha(archive) != score["sha256"]:
                raise ValueError("Physical replies changed")
            outputs = list(zip(*load_streams(archive), strict=True))
            goods = []
            boards = defaultdict(list)
            for (y, s), r in zip(outputs, groups[name], strict=True):
                good = bool(s and y in r.refs)
                goods.append(good)
                boards[r.board].append(good)
            passes = list(map(all, boards.values()))
            assert (len(goods), sum(goods), len(passes), sum(passes)) == (
                score["n"],
                score["correct"],
                score["boards"],
                score["complete"],
            )
            passed[arm][name] = goods
            complete[arm][name] = passes
            if name in ("fresh", "prior", "yan"):
                streams[arm, name] = outputs
    pairs = {}
    for a, b in (
        ("aligned", "replay"),
        ("distilled", "replay"),
        ("distilled", "aligned"),
        ("distilled", "teacher"),
        ("replay", "parent"),
    ):
        pairs[a + "_minus_" + b] = {
            name: compare(complete[a][name], complete[b][name])
            for name in ("fresh", "prior")
        }
    partitions = defaultdict(Counter)
    for i, (p, t) in enumerate(
        zip(passed["parent"]["prior"], passed["teacher"]["prior"], strict=True)
    ):
        group = ("writer" if p else "") + ("reader" if t else "") or "neither"
        partitions[group]["n"] += 1
        for a in ARMS:
            partitions[group][a] += int(passed[a]["prior"][i])
    assert partitions["reader"]["n"] == 98 and partitions["writer"]["n"] == 21
    markers = {}
    for a in ("parent", "replay", "aligned", "distilled"):
        bins = Counter()
        for (y, s), r in zip(streams[a, "fresh"], groups["fresh"], strict=True):
            if s and y in r.refs:
                continue
            bins["wrong_initial"] += int(all(y[:128] != ref[:128] for ref in r.refs))
            bins["right_initial"] += int(any(y[:128] == ref[:128] for ref in r.refs))
            bins["extended_reference"] += int(
                any(y.startswith(ref) and len(y) > len(ref) for ref in r.refs)
            )
            bins["short_prefix"] += int(
                any(ref.startswith(y) and len(y) < len(ref) for ref in r.refs)
            )
        markers[a] = dict(bins)
    e = PUBLIC / "examples"
    e.mkdir(exist_ok=True)
    fresh = groups["fresh"]
    good = passed["distilled"]["fresh"]
    other = passed["replay"]["fresh"]
    exemplar = {}
    for label, idx in [
        (
            "gain",
            next((i for i in range(len(fresh)) if good[i] and not other[i]), None),
        ),
        ("error", next((i for i in range(len(fresh)) if not good[i]), None)),
    ]:
        if idx is None:
            continue
        r = fresh[idx]
        for name, raw in (
            ("page", r.p),
            ("question", r.q),
            ("reference", r.refs[0]),
            ("replay", streams["replay", "fresh"][idx][0]),
            ("distilled", streams["distilled", "fresh"][idx][0]),
        ):
            save_image(e / (label + "_" + name + ".png"), raw)
        exemplar[label] = dict(
            index=idx, board=r.board, exact={a: passed[a]["fresh"][idx] for a in ARMS}
        )
    fresh_rows = list(
        map(json.loads, (WORK / "pairs_dev.jsonl").read_text().splitlines())
    )
    for key, v in exemplar.items():
        row = fresh_rows[v["index"]]
        v.update(q=row["q"], refs=row["refs"])

    def broad(arm):
        return sum(summary[arm]["scores"][g]["correct"] for g in ("cmrc", "drcd"))

    eligible = [
        a
        for a in ("aligned", "distilled")
        if sum(complete[a]["fresh"]) > sum(complete["replay"]["fresh"])
        and broad(a) > broad("replay")
        and all(passed[a]["yan"][-2:])
    ]
    record(
        PUBLIC / "result.json",
        dict(
            arms=summary,
            comparisons=pairs,
            prior_groups=partitions,
            errors=markers,
            examples=exemplar,
            eligible=eligible,
        ),
    )
    for n in ("corpus", "teacher"):
        shutil.copyfile(WORK / (n + ".json"), PUBLIC / (n + ".json"))
    for a in ("replay", "aligned", "distilled"):
        shutil.copyfile(WORK / a / "page_fit.json", PUBLIC / (a + "_fit.json"))
    print("eligible", eligible, "paired", pairs["distilled_minus_replay"], flush=True)


if __name__ == "__main__":
    audit()
