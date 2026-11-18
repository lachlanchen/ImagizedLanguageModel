"""Find monotone classical–modern sentence correspondences in a bounded archival chapter."""

import hashlib
import json
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scripts.prepare_visual_edit_translation import BASE, OUT, pairs
from scripts.prepare_sentence_choice import WORK as PRIOR
from scripts.train_local_ink import (
    ROOT,
    calibration_group,
    load,
    require,
    sha,
    tiles,
    write_new,
)

WORK = ROOT / "artifacts/aligned_visual_translation_v1"
WIDTH = 16


def features(lines):
    vocab = {}
    ind, offset, val = [], [0], []
    for line in lines:
        symbols = [c for c in line if c.isalnum()]
        counts = Counter(symbols)
        counts.update(
            {k: 1.5 * n for k, n in Counter(zip(symbols, symbols[1:])).items()}
        )
        norm = sum(x * x for x in counts.values()) ** 0.5 or 1.0
        for token, n in counts.items():
            ind.append(vocab.setdefault(token, len(vocab)))
            val.append(n / norm)
        offset.append(len(ind))
    return csr_matrix(
        (val, ind, offset), shape=(len(lines), len(vocab)), dtype=np.float32
    )


def path(rows):
    """Banded dynamic program. Vertical and horizontal jumps skip a sentence."""
    n = len(rows)
    if not n:
        return {}
    texts = [s for s, t in rows] + [t for s, t in rows]
    matrix = features(texts)
    score = np.zeros((n, 2 * WIDTH + 1), np.float32)
    for delta in range(-WIDTH, WIDTH + 1):
        a = max(0, -delta)
        b = min(n, n - delta)
        if b > a:
            score[a:b, WIDTH + delta] = np.asarray(
                matrix[a:b].multiply(matrix[n + a + delta : n + b + delta]).sum(1)
            ).ravel()
    dp = np.full((n + 1, 2 * WIDTH + 3), -np.inf)
    trace = np.zeros(dp.shape, np.int8)
    mid = WIDTH + 1
    dp[0, mid] = 0.0
    for i in range(n + 1):
        for j in range(max(0, i - WIDTH), min(n, i + WIDTH) + 1):
            if i == j == 0:
                continue
            d = mid + j - i
            diagonal = dp[i - 1, d] + score[i - 1, d - 1] - 0.16 if i and j else -np.inf
            drop = dp[i - 1, d + 1] - 0.045 if i else -np.inf
            skip = dp[i, d - 1] - 0.045 if j else -np.inf
            best = max(diagonal, drop, skip)
            dp[i, d] = best
            trace[i, d] = 1 if best == diagonal else 2 if best == drop else 3
    i = j = n
    found = {}
    while i or j:
        action = trace[i, mid + j - i]
        if action == 1:
            i -= 1
            j -= 1
            own = float(score[i, WIDTH])
            new = float(score[i, WIDTH + j - i])
            if i == j or (new >= 0.25 and new >= own + 0.12):
                found[i] = (j, new, own)
        elif action == 2:
            i -= 1
        else:
            j -= 1
    return found


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    require(
        sha(BASE / "archive.tar.gz")
        == load(OUT / "preparation.json")["archive_sha256"],
        "Archive changed",
    )
    require(
        sha(PRIOR / "rows.jsonl") == load(PRIOR / "preparation.json")["rows_sha256"],
        "Source rows changed",
    )
    prior_review = (
        ROOT / "references/evidence/block_translation_20261118/calibration_mapping.json"
    )
    require(
        sha(prior_review)
        == "db5851d71ee9932d7f781679ab8ae4f991bc54651762982656fd74890901a568",
        "Changed exclusion roster",
    )
    reviewed = {r["chapter"] for r in load(prior_review)}
    rng = np.random.default_rng(24931)
    char_ids = {}
    chosen = defaultdict(dict)
    eval_groups = set()
    with (PRIOR / "rows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            for field, ids in [("source", "x"), ("target", "y")]:
                char_ids.update(zip(r[field], r[ids]))
            group = r["g"]
            if group in reviewed or not (
                3 <= len(r["x"]) <= 60 and 3 <= len(r["y"]) <= 100
            ):
                continue
            selected = calibration_group(group)
            if selected:
                eval_groups.add(group)
            chosen[group][r["i"]] = (r["source"], r["target"])
    train_groups = sorted(set(chosen) - eval_groups)
    groups = set(rng.choice(train_groups, min(len(train_groups), 1200), replace=False))
    groups.update(
        rng.choice(sorted(eval_groups), min(len(eval_groups), 120), replace=False)
    )
    rows = {"train": [], "heldout": []}
    with tarfile.open(BASE / "archive.tar.gz", "r|gz") as archive:
        for item in archive:
            p = Path(item.name)
            group = "/".join(p.parts[2:-1])
            if (
                group not in groups
                or p.name != "bitext.txt"
                or p.parts[1] != "双语数据"
            ):
                continue
            data = list(pairs(archive.extractfile(item).read()))
            # Independent windows cap memory and retain exact original ordinal provenance.
            part = rows["heldout" if group in eval_groups else "train"]
            for start in range(0, len(data), 384):
                block = data[start : start + 384]
                mapping = path(block)
                for i, (s, t) in enumerate(block):
                    if chosen[group].get(i + start) != (s, t):
                        continue
                    j, new, old = mapping.get(i, (i, None, None))
                    alternative = block[j][1]
                    if not 3 <= len(alternative) <= 100 or any(
                        c not in char_ids for c in alternative
                    ):
                        j, alternative = i, t
                    part.append(
                        dict(
                            group=group,
                            ordinal=i + start,
                            aligned=j + start,
                            source=s,
                            old=t,
                            new=alternative,
                            x=[char_ids[c] for c in s],
                            y=[char_ids[c] for c in t],
                            z=[char_ids[c] for c in alternative],
                        )
                    )
    report = {}
    for split, data in rows.items():
        rng.shuffle(data)
        dest = WORK / (split + ".jsonl")
        with dest.open("x") as f:
            for row in data:
                f.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        report[split] = dict(
            rows=len(data),
            chapters=len({r["group"] for r in data}),
            changed=sum(r["old"] != r["new"] for r in data),
            sha256=sha(dest),
        )
    require(
        not (
            {r["group"] for r in rows["train"]} & {r["group"] for r in rows["heldout"]}
        ),
        "Chapter leakage",
    )
    sample = []
    seen = set()
    for changed in (True, False):
        picked = 0
        for i, r in enumerate(rows["heldout"]):
            if r["group"] in seen or (r["old"] != r["new"]) != changed:
                continue
            choices = [r["old"], r["new"]] if changed else [r["old"]]
            flip = int(rng.integers(2)) if changed else 0
            if flip:
                choices.reverse()
            sample.append(
                (
                    dict(source=r["source"], choices=choices),
                    dict(index=i, changed=changed, old=flip),
                )
            )
            seen.add(r["group"])
            picked += 1
            if picked == 24:
                break
    rng.shuffle(sample)
    write_new(WORK / "blind.json", [x[0] for x in sample])
    write_new(WORK / "key.json", [x[1] for x in sample])
    write_new(
        WORK / "preparation.json",
        dict(
            splits=report,
            bank_sha256=hashlib.sha256(tiles().tobytes()).hexdigest(),
            packet=sha(WORK / "blind.json"),
            key=sha(WORK / "key.json"),
            seconds=time.monotonic() - started,
        ),
    )
    print(report, flush=True)


if __name__ == "__main__":
    run()
