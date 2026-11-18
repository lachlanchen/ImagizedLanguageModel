"""Evaluate the causal visual translator by generating full answer images."""

from collections import Counter, defaultdict
import argparse
import fcntl
import hashlib
import json
import random
import time

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file
from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.measure_pictured_clause_support import MARKS, classify, OUT as SUPPORT
from scripts.align_visual_translation import (
    PRIOR,
    ROOT,
    WORK,
    load,
    sha,
    tiles,
    require,
    write_new,
)
from scripts.train_causal_glyph_translator import OUT
from scripts.train_aligned_visual_translation import read_rows


def select(extra_groups=(), seed="causal-glyph1"):
    previous = read_rows("train") + read_rows("heldout")
    groups = {r["group"] for r in previous} | set(extra_groups)
    groups.update(r["chapter"] for r in load(SUPPORT / "rows.json"))
    groups.update(
        r["chapter"]
        for r in load(
            ROOT
            / "references/evidence/block_translation_20261118/calibration_mapping.json"
        )
    )
    require(
        sha(PRIOR / "rows.jsonl") == load(PRIOR / "preparation.json")["rows_sha256"],
        "Source roster changed",
    )
    candidates = []
    with (PRIOR / "rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["g"] in groups or not (
                5 <= len(row["x"]) <= 24 and 3 <= len(row["y"]) <= 48
            ):
                continue
            rank = hashlib.sha256(f"{seed}/{row['g']}/{row['i']}".encode()).digest()
            candidates.append((rank, row))
    rows, counts = [], Counter()
    for _, row in sorted(candidates, key=lambda v: v[0]):
        if counts[row["g"]] < 8:
            rows.append(row)
            counts[row["g"]] += 1
        if len(rows) == 1024:
            break
    assert len(rows) == 1024
    return rows, previous


def run(recovery=False):
    began = time.monotonic()
    extra_groups = set()
    path = OUT / "translator.safetensors"
    require(sha(path) == load(OUT / "fit.json")["checkpoint_sha256"], "Weight hash")
    paths = {"writer": path}
    directory = OUT
    seed = "causal-glyph1"
    if recovery:
        from scripts.train_visual_feedback_recovery import OUT as directory

        extra_groups = {r["chapter"] for r in load(OUT / "evaluation/rows.json")}
        paths = {"parent": path}
        for name, receipt in load(directory / "fit.json")["fits"].items():
            trained = directory / (name + ".safetensors")
            require(sha(trained) == receipt["sha256"], "Recovery weights changed")
            paths[name] = trained
        seed = "recovery-test1"
    dest = directory / "evaluation"
    dest.mkdir()
    rows, old = select(extra_groups, seed)
    bank = tiles()
    glyphs = torch.tensor(np.unpackbits(bank, axis=1), device="cuda").bool()

    def mask(seq):
        return tuple(map(bytes, bank[seq]))

    require(
        not {mask(r["x"]) for r in rows}.intersection(mask(r["x"]) for r in old),
        "Training or earlier physical source overlap",
    )
    models = {}
    for name, checkpoint in paths.items():
        weights = load_file(str(checkpoint))
        model = CausalGlyphTranslator(weights["landmarks"], weights["exemplars"])
        model.load_state_dict(weights)
        models[name] = model.cuda().eval()
    if not recovery:
        models["reversed"] = models["writer"]
    composer_path = ROOT / "models/visual_edit_zh/composer_55k.msgpack.gz"
    require(
        sha(composer_path) == load(WORK / "scale/summary.json")["checkpoint_sha256"],
        "Composer hash",
    )
    composer = VisualPhraseComposer.restore(composer_path)
    streams = defaultdict(list)
    route = defaultdict(Counter)
    endings = Counter()
    # Only classical images and their own lengths enter these generation calls.
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        lens = torch.tensor([len(r["x"]) for r in batch], device="cuda")
        for name, model in models.items():
            rev = name == "reversed"
            indices = np.zeros((len(batch), int(lens.max())), np.int32)
            for i, r in enumerate(batch):
                ids = r["x"][::-1] if rev else r["x"]
                indices[i, : len(ids)] = ids
            pixels, lengths, stopped, modes = model.generate(
                glyphs[torch.as_tensor(indices, device="cuda").long()], lens
            )
            endings[name] += int(stopped.sum())
            for v, n, actions in zip(
                pixels.cpu().numpy(), lengths.cpu().tolist(), modes.cpu().numpy()
            ):
                streams[name].append(np.packbits(v[:n], axis=1))
                route[name].update(actions[:n].tolist())
        for r in batch:
            streams["copy"].append(bank[r["x"]])
            streams["reference"].append(bank[r["y"]])
            streams["composer"].append(composer.generate(bank[r["x"]]))
    records = []
    for i, r in enumerate(rows):
        ref = mask(r["y"])
        records.append(
            dict(
                chapter=r["g"],
                ordinal=r["i"],
                ref_len=len(ref),
                dist={
                    k: Levenshtein.distance(tuple(map(bytes, s[i])), ref)
                    for k, s in streams.items()
                },
            )
        )
    target_total = sum(r["ref_len"] for r in records)
    summary = {}
    candidate, control = ("recovery", "clean") if recovery else ("writer", "composer")
    pooled = tuple(
        zip(
            *[
                (r["chapter"], r["dist"][candidate] - r["dist"][control], r["ref_len"])
                for r in records
            ]
        )
    )
    groups = defaultdict(lambda: np.zeros(2))
    for g, v, n in zip(*pooled):
        groups[g] += (v, n)
    totals = np.stack(list(groups.values()))
    rng = np.random.default_rng(64391)
    boot = totals[rng.integers(len(totals), size=(2000, len(totals)))].sum(axis=1)
    for k, vals in streams.items():
        delta = np.array([r["dist"][k] for r in records])
        changed = np.array([r["dist"]["copy"] > 0 for r in records])
        summary[k] = dict(
            edits=int(delta.sum()),
            cer=float(delta.sum() / target_total),
            exact=int((delta == 0).sum()),
            changed_exact=int(((delta == 0) & changed).sum()),
            output_cells=sum(len(v) for v in vals),
            empty=sum(not len(v) for v in vals),
        )
        np.savez_compressed(
            dest / (k + ".npz"),
            packed=np.concatenate(vals),
            offsets=np.cumsum([0] + list(map(len, vals))),
        )
    lex = defaultdict(Counter)
    for r in old:
        for char, idx in zip(r["new"], r["z"]):
            lex[bytes(bank[idx])][char] += 1
    for r in rows:
        for s, k in [("source", "x"), ("target", "y")]:
            for char, idx in zip(r[s], r[k]):
                lex[bytes(bank[idx])][char] += 1
    rng = random.Random(7023)
    occupied = set()
    selected = set()
    for topic, _ in MARKS:
        n = 0
        for i, r in enumerate(rows):
            if classify(r["source"]) == topic and r["g"] not in occupied:
                occupied.add(r["g"])
                selected.add(i)
                n += 1
                if n == 6:
                    break
        require(n == 6, "Review stratum")
    blind = []
    key = []
    for i in sorted(selected):
        options = {k: tuple(map(bytes, v[i])) for k, v in streams.items()}
        unique = list(dict.fromkeys(options.values()))
        rng.shuffle(unique)
        blind.append(
            dict(
                source=rows[i]["source"],
                choices=[
                    "".join(
                        lex[b].most_common(1)[0][0] if b in lex else "▧" for b in seq
                    )
                    for seq in unique
                ],
            )
        )
        key.append(
            dict(row=i, options={k: unique.index(s) for k, s in options.items()})
        )
    write_new(dest / "review.json", blind)
    write_new(dest / "mapping.json", key)
    write_new(dest / "rows.json", records)
    write_new(
        dest / "summary.json",
        dict(
            arms=summary,
            chapter_count=len(groups),
            sentences=len(rows),
            target_cells=target_total,
            route_counts=route,
            stops=endings,
            **{
                candidate + "_minus_" + control + "_ci95": np.quantile(
                    boot[:, 0] / boot[:, 1], [0.025, 0.975]
                ).tolist()
            },
            seconds=time.monotonic() - began,
            checkpoint=sha(path),
            checkpoints={k: sha(v) for k, v in paths.items()},
            files_sha256={p.name: sha(p) for p in dest.iterdir()},
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.recovery)
