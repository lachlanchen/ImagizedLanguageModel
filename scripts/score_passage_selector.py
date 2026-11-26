"""Recount complete words directly from emitted ink, including ordinary self-feedback."""

from collections import Counter, defaultdict
from functools import lru_cache
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq

from scripts.evaluate_passage_selector import ARCHIVE, SPLITS, read_ink, read_array
from scripts.evaluate_book_page_memory import RAW, normalize
from scripts.prepare_passage_selector import WORK, PUB, MODEL, ROOT, RULE, base, WORD
from ilm.visual_lm.passage_selector import Print


def bounds(ok, other, groups):
    _, inverse = np.unique(groups, return_inverse=True)
    n = np.bincount(inverse)
    gain = np.bincount(inverse, weights=ok.astype(float) - other)
    rng = np.random.default_rng(862173)
    samples = []
    for _ in range(100):
        ids = rng.integers(len(n), size=(100, len(n)))
        samples.extend(gain[ids].sum(1) / n[ids].sum(1))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def backoff():
    """Supplementary word-count baseline with last-three, two, one and zero-word contexts."""
    with np.load(WORK / "train.npz") as data:
        s = data["source"][:, -3:].tolist()
        t = data["task"].tolist()
        ys = data["target"][:, 0].tolist()
    counts = defaultdict(Counter)
    for row, task, y in zip(s, t, ys, strict=True):
        for k in range(4):
            counts[(task, *row[3 - k :])][y] += 1
    choices = {k: max(v, key=v.get) for k, v in counts.items() if sum(v.values()) >= 3}

    def choose(row, task):
        for k in range(3, -1, -1):
            key = (task, *row[3 - k :])
            if key in choices:
                return choices[key]
        raise AssertionError("Missing frequency prior")

    return choose


def exact(seq, gold):
    return np.array([np.array_equal(x, y) for x, y in zip(seq, gold, strict=True)])


def main():
    face = Print()
    forms = base.read(WORK / "forms.json")
    choose = backoff()
    result = {}
    historical = pq.read_table(RAW / "lambada-validation.parquet", columns=["text"])[
        "text"
    ].to_pylist()
    with ZipFile(ARCHIVE) as z:
        for split in SPLITS:
            with np.load(WORK / (split + ".npz")) as inputs:
                src, targets, group, task, marks, ends = [
                    inputs[k]
                    for k in ("source", "target", "group", "task", "marks", "ends")
                ]
            assert np.array_equal(
                group, read_array(z, split + "/group")
            ) and np.array_equal(task, read_array(z, split + "/task"))
            within = (src == targets[:, 0, None]).any(1)
            assert np.array_equal(within, read_array(z, split + "/present"))
            gold = read_ink(z, split + "/gold0")
            raw = base.Ink()

            @lru_cache(maxsize=None)
            def target(i):
                code = raw.encode(forms[i])
                return raw.array()[code]

            assert all(
                np.array_equal(target(y), blob)
                for y, blob in zip(targets[:, 0], gold, strict=True)
            )
            answers = {
                arm: exact(read_ink(z, split + "/" + arm), gold)
                for arm in (
                    "intact",
                    "shuffled",
                    "donor",
                    "salience",
                    "pointer",
                    "spell",
                )
            }
            ids = [
                choose(row, t)
                for row, t in zip(src[:, -3:].tolist(), task, strict=True)
            ]
            answers["local_counts"] = exact([target(i) for i in ids], gold)
            if split == "lambada":
                full = []
                for gid in group:
                    prefix, y = normalize(historical[gid]).rsplit(" ", 1)
                    words = [w for w in WORD.findall(prefix) if len(w) <= 23]
                    count = Counter(reversed(words))
                    winner = (
                        max(
                            count,
                            key=lambda w: count[w]
                            * np.log(97 / (1 + face.df.get(w.lower(), 0))),
                        )
                        if count
                        else ""
                    )
                    full.append(winner == y)
                assert (
                    sum(full)
                    == base.read(
                        ROOT / "references/evidence/book_page_memory_v1/audit.json"
                    )["salience"]["page"]
                )
                answers["full_page_salience"] = np.array(full)
            ix = read_array(z, split + "/donor")
            assert np.all(group[ix] != group) and np.all(task[ix] == task)
            from scripts.evaluate_passage_selector import observations

            swapped, _ = observations(src, marks, ix)
            donor_has = (swapped == targets[:, 0, None]).any(1)
            slices = {
                "all": np.ones(len(src), bool),
                "present": within,
                "absent": ~within,
                "present_both": within & donor_has,
                "absent_both": ~within & ~donor_has,
                "ordinary": task == 0,
                "final": task == 1,
            }
            strata = {
                k: dict(
                    n=int(v.sum()),
                    correct={a: int(hit[v].sum()) for a, hit in answers.items()},
                )
                for k, v in slices.items()
                if v.any()
            }
            ci = {
                arm: bounds(answers["intact"], other, group)
                for arm, other in answers.items()
                if arm != "intact"
            }
            stream = {}
            if split != "lambada":
                selection = read_array(z, split + "/stream-rows")
                stop = read_array(z, split + "/stream-stop_step")
                reached = np.ones(len(selection), bool)
                gold_end = np.zeros(len(selection), int)
                first = np.argmax(ends[selection], 1) + 1
                gold_end = np.where(ends[selection].any(1), first, 0)
                for j in range(3):
                    gen = read_ink(z, split + "/stream" + str(j))
                    ref = read_ink(z, split + "/gold" + str(j))
                    eligible = (gold_end == 0) | (gold_end >= j + 1)
                    active = (stop == 0) | (stop >= j + 1)
                    ok = exact(gen, [ref[i] for i in selection])
                    assert np.array_equal(active, np.array([len(x) > 0 for x in gen]))
                    reached &= ok & active
                    stream[str(j + 1)] = dict(
                        eligible=int(eligible.sum()),
                        emitted=int(active.sum()),
                        prefix_exact=int((reached & eligible).sum()),
                    )
                stream["stop_agreement"] = int((stop == gold_end).sum())
                stream["both_stopped_at_reference"] = int(
                    ((stop == gold_end) & (stop > 0)).sum()
                )
                stream["sample"] = len(selection)
            result[split] = dict(
                strata=strata,
                intervals=ci,
                stream=stream,
                pass_cohort=bool(ci["salience"][0] > 0 and ci["donor"][0] > 0),
            )
            print(split, strata["all"], ci, stream, flush=True)
    base.dump(
        PUB / "scores.json",
        dict(
            archive=base.sha(ARCHIVE),
            protocol=base.sha(RULE),
            weights={
                a: base.sha(MODEL / a / "weights.safetensors")
                for a in ("intact", "shuffled")
            },
            results=result,
            passed=all(result[k]["pass_cohort"] for k in ("fiction", "gutenberg")),
        ),
    )


if __name__ == "__main__":
    main()
