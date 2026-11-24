"""Fit both cold word writers sequentially and measure quality through training."""

import argparse
from collections import defaultdict
from functools import partial
import gzip
import json
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file, load_file

from ilm.visual_lm.dialogue_surface import SurfaceDialogue
from scripts.answer_dialogue_ink import generate
from scripts.dialogue_ink_data import Corpus, WordImages, read_data
from scripts.evaluate_dialogue_ink import exchanges
from scripts.evaluate_optical_passage_qa import metrics
from scripts.prepare_dialogue_ink import ROOT, WORK, PUB, dump, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.train_dialogue_ink import SEED, dialogue_loss
from scripts.train_dialogue_subword import timed
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot

STEPS = (100, 800, 2400, 8000)
ROOT_MODEL = ROOT / "models/dialogue_surfaces"


def representation(mode):
    if mode == "ink":
        return Face(FACES[0]), 128, bytes(128)
    return str.encode, 1, b" "


def corpus(data, group, net, face, unit):
    bank = partial(WordImages, unit=unit, encode_words=net.encode_words)
    return Corpus(
        data, group, net.reader, None, face=face, bank_factory=bank, unit=unit
    )


def fit(data, mode, dest, work):
    if dest.exists() or work.exists():
        raise FileExistsError("Choose a fresh fit location")
    face, unit, _ = representation(mode)
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    # Enumerating observed answer cells defines the writable palette. Word matches use physical bytes.
    glyphs = sorted(
        {
            face(c)
            for r in data["rows"]
            if r[0] == "train"
            for c in set(r[5])
            if c != " "
        }
    )
    palette = torch.from_numpy(
        np.frombuffer(bytes(unit) + b"".join(glyphs), np.uint8).copy().reshape(-1, unit)
    )
    torch.manual_seed(SEED)
    net = SurfaceDialogue(palette, mode).cuda()
    ds = corpus(data, "train", net, face, unit)
    observed, labels = ds.glyph_targets()
    if not torch.equal(palette, observed):
        raise ValueError("Different training palette")
    prep = timed(start, meter)
    optim = torch.optim.AdamW(net.parameters(), lr=0.00075)
    rng = np.random.default_rng(SEED)
    dest.mkdir(parents=True)
    work.mkdir(parents=True)
    costs = {}
    seen = set()
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, max(STEPS) + 1):
        ix = rng.integers(len(ds.rows), size=32)
        seen.update(ix)
        loss, *_ = dialogue_loss(net, ds, ix, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        optim.step()
        if step % 250 == 0:
            print(mode, step, round(monotonic() - start, 1), flush=True)
        if step % 250 == 0 or step in STEPS:
            if (
                monotonic() - start > 3000
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Exceeded registered fit ceiling")
        if step in STEPS:
            cost = timed(start, meter)
            file = (
                dest / "writer.safetensors"
                if step == max(STEPS)
                else work / f"{step}.safetensors"
            )
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
                str(file),
            )
            costs[str(step)] = dict(
                **cost, examples=32 * step, distinct=len(seen), sha256=sha(file)
            )
    dump(
        dest / "weights.json",
        dict(
            mode=mode,
            parameters=sum(v.numel() for v in net.parameters()),
            seed=SEED,
            glyphs=len(glyphs),
            preparation=prep,
            checkpoints=costs,
            split=sha(PUB / "split.json"),
        ),
    )
    return net


@torch.inference_mode()
def evaluate(data, net, mode, dest, work):
    face, unit, sep = representation(mode)
    symbols = {face(chr(i)): chr(i) for i in range(32, 127)}
    held = json.loads(
        gzip.decompress((WORK / "evaluation/decisions.json.gz").read_bytes())
    )
    scores = defaultdict(dict)
    records = []
    for group in ("reserved", "dev"):
        rows = [r for r in held if r["group"] == group]
        ds = corpus(data, group, net, face, unit)
        q, p = exchanges(ds.rows)
        for step in STEPS:
            file = (
                dest / "writer.safetensors"
                if step == max(STEPS)
                else work / f"{step}.safetensors"
            )
            net.load_state_dict(load_file(str(file)))
            totals = defaultdict(int)
            totals["turns"] = len(rows)
            for arm in (
                ("writer", "exchanged_question", "exchanged_passage")
                if step == max(STEPS)
                else ("writer",)
            ):
                for off in range(0, len(rows), 48):
                    idx = np.arange(off, min(off + 48, len(rows)))
                    x, variants = ds.inputs(
                        idx,
                        q[idx] if arm == "exchanged_question" else None,
                        p[idx] if arm == "exchanged_passage" else None,
                    )
                    ink, end, _ = generate(
                        net,
                        None,
                        ds.bank,
                        x,
                        variants,
                        separator=sep,
                        feedback=net.surface,
                    )
                    for i, raw, stop in zip(idx, ink, end, strict=True):
                        text = "".join(
                            symbols[raw[j : j + unit]] for j in range(0, len(raw), unit)
                        )
                        ref = rows[i]["answer"]
                        hit, f1 = metrics(text, [ref])
                        e = dict(
                            text=text,
                            stopped=stop,
                            exact=bool(hit and stop),
                            f1=f1 if stop else 0.0,
                            pixels=bool(stop and raw == face(ref)),
                        )
                        if step == max(STEPS):
                            rows[i].setdefault("surface", {})[arm] = e
                        totals[arm + "_exact"] += e["exact"]
                        totals[arm + "_verbatim"] += e["pixels"]
            scores[group][step] = dict(totals)
        for r in rows:
            r["out"] = r.pop("surface")
        records.extend(rows)
    (work / "decisions.json.gz").write_bytes(
        gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    )
    dump(work / "curve.json", scores)
    if work.resolve() == (WORK / "matched" / mode).resolve():
        dump(PUB / f"{mode}_curve.json", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT_MODEL)
    parser.add_argument("--work", type=Path, default=WORK / "matched")
    args = parser.parse_args()
    with gpu_owner():
        data = read_data()
        for mode in ("ink", "byte"):
            dest = args.output / mode
            work = args.work / mode
            model = fit(data, mode, dest, work)
            evaluate(data, model.eval(), mode, dest, work)
