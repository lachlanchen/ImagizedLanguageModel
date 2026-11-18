"""Measure the reachable language of a frozen visual phrase translator on fresh chapters."""

import hashlib
import json
import random
import resource
import time
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein
from ilm.visual_lm.phrase_lattice import admits, best_streams, graph, nearest_stream
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.align_visual_translation import (
    PRIOR,
    ROOT,
    WORK,
    calibration_group,
    load,
    require,
    sha,
    tiles,
    write_new,
)
from scripts.train_aligned_visual_translation import read_rows

OUT = ROOT / "artifacts/pictured_clause_support_v1"
MARKS = [
    ("negation", "不未无非勿莫毋弗罔"),
    ("speech", "曰言谓云"),
    ("event", "攻杀伐赐拜徙任取击救立还"),
]
NAMES = ("copy", "default", "rank1", "six", "unrestricted")


def classify(source):
    return next(
        (label for label, chars in MARKS if any(v in source for v in chars)), "other"
    )


def select():
    train, previous = read_rows("train"), read_rows("heldout")
    excluded = {r["group"] for r in train + previous}
    excluded.update(
        r["chapter"]
        for r in load(
            ROOT
            / "references/evidence/block_translation_20261118/calibration_mapping.json"
        )
    )
    require(
        sha(PRIOR / "rows.jsonl") == load(PRIOR / "preparation.json")["rows_sha256"],
        "Changed source inventory",
    )
    candidates = []
    with (PRIOR / "rows.jsonl").open() as data:
        for line in data:
            r = json.loads(line)
            if r["g"] in excluded or not calibration_group(r["g"]):
                continue
            if 5 <= len(r["x"]) <= 24 and 3 <= len(r["y"]) <= 48:
                rank = hashlib.sha256(f"support1/{r['g']}/{r['i']}".encode()).digest()
                candidates.append((rank, r))
    rows, counts = [], Counter()
    for _, r in sorted(candidates, key=lambda p: p[0]):
        if counts[r["g"]] < 8:
            rows.append(r)
            counts[r["g"]] += 1
        if len(rows) == 1024:
            break
    require(len(rows) == 1024, "Insufficient fresh records")
    return rows, train + previous


def experiment():
    started = time.monotonic()
    OUT.mkdir()
    rows, prior = select()
    weights = ROOT / "models/visual_edit_zh/composer_55k.msgpack.gz"
    require(
        sha(weights) == load(WORK / "scale/summary.json")["checkpoint_sha256"],
        "Wrong composer",
    )
    model = VisualPhraseComposer.restore(weights)
    bank = tiles()
    require(
        hashlib.sha256(bank.tobytes()).hexdigest()
        == load(WORK / "preparation.json")["bank_sha256"],
        "Changed bank",
    )
    codes = {bytes(p): model.indices.get(bytes(p), -2 - i) for i, p in enumerate(bank)}
    by_index = {v: k for k, v in codes.items()}

    def sequence(row, field):
        return tuple(codes[bytes(bank[i])] for i in row[field])

    training_sources = {sequence(r, "x") for r in prior}
    sources = [sequence(r, "x") for r in rows]
    require(
        len(set(sources)) == len(rows) and not training_sources.intersection(sources),
        "Duplicate source ink",
    )
    trans = defaultdict(Counter)
    for r in rows:
        for field, val in [("x", "source"), ("y", "target")]:
            for idx, char in zip(r[field], r[val]):
                trans[codes[bytes(bank[idx])]][char] += 1
    for r in prior:
        for idx, char in zip(r["z"], r["new"]):
            trans[codes[bytes(bank[idx])]][char] += 1

    def readable(seq):
        return "".join(
            trans[i].most_common(1)[0][0] if i in trans else "▧" for i in seq
        )

    review_ids = set()
    occupied = set()
    for name, _ in MARKS:
        eligible = [i for i, r in enumerate(rows) if classify(r["source"]) == name]
        n = 0
        for i in eligible:
            if rows[i]["g"] in occupied:
                continue
            occupied.add(rows[i]["g"])
            review_ids.add(i)
            n += 1
            if n == 6:
                break
        require(n == 6, "Missing review stratum")
    all_stats = []
    rng = random.Random(1987)
    blind, mapping = [], []
    offsets, raw = defaultdict(lambda: [0]), defaultdict(bytearray)
    for i, (r, src) in enumerate(zip(rows, sources)):
        ref = sequence(r, "y")
        edges = graph(model, src)
        actual = model.decode(src)
        expected_image = model.generate(bank[r["x"]]).tobytes()
        require(
            expected_image == b"".join(by_index[k] for k in actual),
            "Public runtime disagreement",
        )
        candidates = [seq for seq, _ in best_streams(model, edges)]
        errors = [Levenshtein.distance(s, ref) for s in candidates]
        closest = candidates[errors.index(min(errors))]
        lower, witness = nearest_stream(edges, ref)
        require(
            admits(edges, ref) == (lower == 0) and admits(edges, witness),
            "Witness or exact membership failed",
        )
        require(
            Levenshtein.distance(witness, ref) == lower <= min(errors)
            and lower <= Levenshtein.distance(actual, ref),
            "Bad edit bound",
        )
        outputs = {
            "copy": src,
            "default": actual,
            "rank1": candidates[0],
            "six": closest,
            "unrestricted": witness,
            "reference": ref,
        }
        # Keep all six independent proposals plus the bound and actual operating point.
        for name, s in {
            **outputs,
            **{f"candidate{j}": s for j, s in enumerate(candidates)},
        }.items():
            raw[name].extend(b"".join(by_index[k] for k in s))
        for name in set(outputs) | {f"candidate{j}" for j in range(6)}:
            offsets[name].append(len(raw[name]) // 128)
        distances = {n: Levenshtein.distance(outputs[n], ref) for n in NAMES}
        all_stats.append(
            dict(
                chapter=r["g"],
                ordinal=r["i"],
                ref_len=len(ref),
                source_len=len(src),
                dist=distances,
                candidates=len(candidates),
                default_in_six=actual in candidates,
                default_is_best=actual == candidates[0],
            )
        )
        if i in review_ids:
            opts = {**outputs, **{f"candidate{j}": s for j, s in enumerate(candidates)}}
            unique = list(dict.fromkeys(opts.values()))
            rng.shuffle(unique)
            blind.append(
                dict(source=r["source"], choices=[readable(s) for s in unique])
            )
            mapping.append(
                dict(
                    row=i,
                    stratum=classify(r["source"]),
                    options={n: unique.index(s) for n, s in opts.items()},
                )
            )
        if i % 64 == 0:
            require(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 6 * 2**20,
                "Process memory ceiling",
            )
            with open("/proc/meminfo") as f:
                require(
                    int(next(s for s in f if s.startswith("MemAvailable:")).split()[1])
                    >= 24 * 2**20,
                    "Host reserve",
                )
    for name in raw:
        np.savez_compressed(
            OUT / (name + ".npz"),
            packed=np.frombuffer(raw[name], np.uint8).reshape(-1, 128),
            offsets=offsets[name],
        )
    write_new(OUT / "review.json", blind)
    write_new(OUT / "mapping.json", mapping)
    write_new(OUT / "rows.json", all_stats)
    write_new(
        OUT / "summary.json",
        dict(
            count=len(rows),
            chapters=len({r["g"] for r in rows}),
            denominator=sum(len(r["y"]) for r in rows),
            edits={n: sum(r["dist"][n] for r in all_stats) for n in NAMES},
            exact={n: sum(r["dist"][n] == 0 for r in all_stats) for n in NAMES},
            changed_exact={
                n: sum(r["dist"]["copy"] > 0 and r["dist"][n] == 0 for r in all_stats)
                for n in NAMES
            },
            default_is_best=sum(r["default_is_best"] for r in all_stats),
            default_in_six=sum(r["default_in_six"] for r in all_stats),
            review_sha256=sha(OUT / "review.json"),
            model_sha256=sha(weights),
            seconds=time.monotonic() - started,
            private_hashes={p.name: sha(p) for p in OUT.iterdir()},
        ),
    )


if __name__ == "__main__":
    experiment()
