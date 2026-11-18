"""Generate all revision arms from untouched passages before reading method assignments."""

from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import random
import time

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.pictured_revision import PicturedReviser
from scripts.align_visual_translation import (
    ROOT,
    PRIOR,
    load,
    sha,
    tiles,
    write_new,
    require,
)
from scripts.measure_pictured_clause_support import MARKS, classify
from scripts.train_pictured_revision import (
    OUT,
    apply_proposals,
    composer,
    proposals,
    read_rows,
)


def fresh(bank, *, earlier=(), seed="reviseread1"):
    prior = read_rows("train") + read_rows("heldout")
    chapters = {r["group"] for r in prior}
    chapters.update(
        r["chapter"]
        for r in load(
            ROOT
            / "references/evidence/block_translation_20261118/calibration_mapping.json"
        )
    )

    def identity(packed):
        return hashlib.sha256(packed.tobytes()).digest()

    used = {identity(bank[r["x"]]) for r in prior}
    for name, sub, source_file, *roster_file in [
        ("pictured_clause_support_v1", "", "copy"),
        ("causal_glyph_translation_v1", "evaluation", "copy"),
        ("visual_feedback_recovery_v1", "evaluation", "copy"),
        *earlier,
    ]:
        d = ROOT / "artifacts" / name / sub
        chapters.update(r["chapter"] for r in load(d / (roster_file[0] if roster_file else "rows.json")))
        with np.load(d / (source_file + ".npz")) as words:
            cells = words["packed"]
            offs = words["offsets"]
            used.update(identity(cells[a:b]) for a, b in zip(offs, offs[1:]))
    require(
        sha(PRIOR / "rows.jsonl") == load(PRIOR / "preparation.json")["rows_sha256"],
        "Raw corpus changed",
    )
    candidates = []
    with (PRIOR / "rows.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["g"] in chapters or not (
                5 <= len(r["x"]) <= 24 and 3 <= len(r["y"]) <= 48
            ):
                continue
            score = hashlib.sha256(f"{seed}/{r['g']}/{r['i']}".encode()).digest()
            candidates.append((score, r))
    rows = []
    counts = Counter()
    for _, r in sorted(candidates, key=lambda p: p[0]):
        digest = identity(bank[r["x"]])
        if counts[r["g"]] >= 8 or digest in used:
            continue
        rows.append(r)
        counts[r["g"]] += 1
        used.add(digest)
        if len(rows) == 1024:
            break
    require(len(rows) == 1024, "Test roster too small")
    return rows, prior


def run():
    dest = OUT / "evaluation"
    dest.mkdir()
    start = time.monotonic()
    fit = load(OUT / "fit.json")
    weights = OUT / "reviser.safetensors"
    require(sha(weights) == fit["checkpoint"], "Reviser changed")
    model = PicturedReviser.restore(weights).cuda()
    raw = tiles()
    rows, prior = fresh(raw)
    engine = composer()
    sources = [raw[r["x"]] for r in rows]
    drafts = [engine.generate(s) for s in sources]
    groups = defaultdict(list)
    for i, s in enumerate(sources):
        groups[len(s)].append(i)
    exchange = [None] * len(rows)
    for indices in groups.values():
        for j, i in enumerate(indices):
            options = indices[j + 1 :] + indices[:j]
            exchange[i] = next(k for k in options if rows[i]["g"] != rows[k]["g"])
    require(
        all(
            i != j and len(sources[i]) == len(sources[j])
            for i, j in enumerate(exchange)
        ),
        "Exchange",
    )
    outputs = {
        "source": sources,
        "draft": drafts,
        "reference": [raw[r["y"]] for r in rows],
    }
    for arm, view in [
        ("revised", sources),
        ("exchanged", [sources[j] for j in exchange]),
    ]:
        edits = proposals(model, model.bank.cpu().numpy(), view, drafts)
        outputs[arm] = [
            apply_proposals(d, e, fit["margin"]) for d, e in zip(drafts, edits)
        ]
    records = []
    names = list(outputs)
    for i, r in enumerate(rows):
        ref = tuple(map(bytes, outputs["reference"][i]))
        dist = {
            k: Levenshtein.distance(tuple(map(bytes, v[i])), ref)
            for k, v in outputs.items()
        }
        records.append(
            dict(
                chapter=r["g"],
                ordinal=r["i"],
                exchange_row=exchange[i],
                ref_len=len(ref),
                dist=dist,
            )
        )
    changed = [not np.array_equal(a, b) for a, b in zip(drafts, outputs["revised"])]
    total = sum(r["ref_len"] for r in records)
    gstats = defaultdict(lambda: np.zeros(3))
    arms = {}
    for k, v in outputs.items():
        d = np.array([r["dist"][k] for r in records])
        copied = np.array([r["dist"]["source"] == 0 for r in records])
        arms[k] = dict(
            edits=int(d.sum()),
            cer=float(d.sum() / total),
            exact=int((d == 0).sum()),
            changed_exact=int(((d == 0) & ~copied).sum()),
            emitted=sum(map(len, v)),
        )
        np.savez_compressed(
            dest / (k + ".npz"),
            packed=np.concatenate(v),
            offsets=np.cumsum([0] + list(map(len, v))),
        )
    for r in records:
        d = r["dist"]
        gstats[r["chapter"]] += (
            d["revised"] - d["draft"],
            d["revised"] - d["exchanged"],
            r["ref_len"],
        )
    grouped = np.stack(list(gstats.values()))
    gen = np.random.default_rng(48135)
    boot = grouped[gen.integers(len(grouped), size=(2000, len(grouped)))].sum(axis=1)
    packets = {"changed": [], "all": []}
    seen = set()
    for i, r in enumerate(rows):
        if changed[i] and r["g"] not in seen:
            packets["changed"].append(i)
            seen.add(r["g"])
            if len(seen) == 18:
                break
    seen = set()
    for topic, _ in MARKS:
        subset = []
        for i, r in enumerate(rows):
            if r["g"] not in seen and classify(r["source"]) == topic:
                subset.append(i)
                seen.add(r["g"])
                if len(subset) == 6:
                    break
        require(len(subset) == 6, "Insufficient review stratum")
        packets["all"] += subset
    lex = defaultdict(Counter)
    for r in prior:
        for idx, c in zip(r["z"], r["new"]):
            lex[bytes(raw[idx])][c] += 1
    for r in rows:
        for ids, chars in [(r["x"], r["source"]), (r["y"], r["target"])]:
            for idx, c in zip(ids, chars):
                lex[bytes(raw[idx])][c] += 1
    reader = random.Random(64132)
    review = []
    mapping = []
    for i in sorted(set(packets["changed"]) | set(packets["all"])):
        alternatives = {
            k: tuple(map(bytes, outputs[k][i])) for k in names if k != "source"
        }
        options = list(dict.fromkeys(alternatives.values()))
        reader.shuffle(options)
        review.append(
            dict(
                source=rows[i]["source"],
                choices=[
                    "".join(lex[v].most_common(1)[0][0] for v in o) for o in options
                ],
            )
        )
        mapping.append(
            dict(row=i, options={k: options.index(v) for k, v in alternatives.items()})
        )
    write_new(dest / "review.json", review)
    write_new(dest / "mapping.json", mapping)
    write_new(dest / "packets.json", packets)
    write_new(dest / "rows.json", records)
    d = np.array([r["dist"]["revised"] - r["dist"]["draft"] for r in records])
    write_new(
        dest / "summary.json",
        dict(
            sentences=len(rows),
            chapters=len(gstats),
            target_cells=total,
            arms=arms,
            changed=sum(changed),
            better=int((d < 0).sum()),
            worse=int((d > 0).sum()),
            tied_changed=int(sum(changed) - (d != 0).sum()),
            exact_lost=sum(
                r["dist"]["draft"] == 0 and r["dist"]["revised"] != 0 for r in records
            ),
            exact_gained=sum(
                r["dist"]["draft"] != 0 and r["dist"]["revised"] == 0 for r in records
            ),
            source_sensitive=sum(
                not np.array_equal(a, b)
                for a, b in zip(outputs["revised"], outputs["exchanged"])
            ),
            ci95=np.quantile(
                boot[:, :2] / boot[:, 2:], [0.025, 0.975], axis=0
            ).T.tolist(),
            seconds=time.monotonic() - start,
            files={p.name: sha(p) for p in dest.iterdir()},
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
