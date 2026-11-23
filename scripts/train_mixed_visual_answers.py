"""Continue the page writer on physical replies from Chinese reading and historical cards."""

from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.paragraph_evidence_corpus import WORK as SEGMENTS
from scripts.pointing_glyph_corpus import Row, digest, plates
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import ROOT, Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_page_writer_requests import VARIED, load_records
from scripts.train_paragraph_evidence import records
from scripts.train_pointing_causal_writer import inputs, objective
from scripts.train_span_choice import gpu_owner

WORK = ROOT / "artifacts/mixed_visual_answers_v1"
PUBLIC = ROOT / "references/evidence/mixed_visual_answers_20261124"
SEED = 241154


def materialize(face):
    draw = lru_cache(maxsize=16000)(face)
    held = {draw(p) for v in records(SEGMENTS, "held") for p in v["parts"]}
    held.update(draw(v["p"]) for v in split("fresh"))
    held.update(
        draw(v["p"]) for group in ("held", "fresh") for v in load_records(VARIED, group)
    )
    held.update(r.p for r in plates("development", face) if r.task != "yan")
    replay = load_records(VARIED, "training")
    items = []
    for i in np.random.default_rng(SEED).permutation(len(replay))[:6000]:
        v = replay[i]
        items.append(
            Row(
                "replay_" + v["task"],
                v["key"],
                draw(v["p"]),
                draw(v["q"]),
                draw(v["y"]),
                b"",
            )
        )
    for v in split("train"):
        if v["group"] in ("cmrc", "drcd"):
            items.append(
                Row(
                    v["group"],
                    v["key"],
                    draw(v["p"]),
                    draw(v["q"]),
                    draw(v["refs"][0]),
                    b"",
                )
            )
    for i, v in enumerate(records(SEGMENTS, "train")):
        if v["type"] in ("cmrc", "drcd"):
            p = v["parts"][v["right"]]
            target = next(r for r in v["refs"] if r in p)
            items.append(
                Row(
                    v["type"] + "_part",
                    str(i),
                    draw(p),
                    draw(v["q"]),
                    draw(target),
                    b"",
                )
            )
    items.extend(plates("train", face))
    return [r for r in items if r.p not in held]


def fit(folder):
    folder.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    base = VARIED / "page"
    original = json.loads((VARIED / "page_fit.json").read_text())["weights"]
    rows = materialize(Face((FONTS[0], 2)))
    fit_rows(folder, base, original, rows, SEED)


def fit_rows(folder, base, original, rows, seed, supervision=None, typeface_map=None):
    torch.manual_seed(seed)
    if any(sha(base / n) != h for n, h in original.items()):
        raise ValueError("Parent writer changed")
    model = PointingCausalWriter.restore(base, "cuda")
    if typeface_map is not None:
        model.writer.attach_typeface_map(typeface_map)
    model.writer.requires_grad_(False)
    model.writer.body.h[-4:].requires_grad_(True)
    model.writer.body.ln_f.requires_grad_(True)
    report = dict(
        seed=seed,
        groups=dict(Counter(r.task for r in rows)),
        rows=len(rows),
        image_digest=digest(rows),
        parent=original,
    )
    record(folder / "corpus.json", report)
    print(report["groups"], flush=True)
    read, alphabet, pairs, answers = inputs(model.writer, rows)
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=[p for p in model.writer.parameters() if p.requires_grad],
                lr=5e-5,
            ),
            dict(params=model.pointer.parameters(), lr=4e-4),
        ],
        fused=True,
    )
    total = count = steps = 0
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)
    for pool in np.array_split(rng.permutation(len(rows)), max(1, len(rows) // 512)):
        pool = sorted(pool, key=lambda i: len(rows[i].p) + 2 * len(rows[i].q))
        for ids in np.array_split(pool, int(np.ceil(len(pool) / 20))):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, nll, n = objective(
                    model,
                    read,
                    alphabet,
                    [pairs[i] for i in ids],
                    [answers[i] for i in ids],
                    False,
                    supervision(ids) if supervision is not None else None,
                )
            if not torch.isfinite(loss) or loss.item() > 100:
                raise ValueError("Unreachable or non-finite physical answer")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            count += n
            total += float(nll)
            steps += 1
            if steps % 200 == 0:
                print(steps, round(total / count, 3), flush=True)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 20 * 2**30
            ):
                raise RuntimeError("Resource ceiling reached")
    stage = folder / "page.partial"
    stage.mkdir()
    for f in ("config.json", "interface.safetensors", "metric.safetensors"):
        os.link(base / f, stage / f)
    if typeface_map is not None:
        os.link(typeface_map, stage / "typeface.safetensors")
    for name, part in (("body", model.writer.body), ("pointer", model.pointer)):
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in part.state_dict().items()},
            str(stage / (name + ".safetensors")),
        )
    stage.rename(folder / "page")
    record(
        folder / "page_fit.json",
        dict(
            rows=len(rows),
            updates=steps,
            exposures=count,
            nll=total / count,
            optimization_seconds=monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            weights={p.name: sha(p) for p in (folder / "page").iterdir()},
        ),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=WORK)
    with gpu_owner():
        fit(ap.parse_args().run_dir)
