"""Check image-mass aggregation and compare its stopped replies with the saved control."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from scripts.answer_dialogue_surface import load
from ilm.visual_lm.image_copy_search import copy_groups, modal_copy_position
from scripts.evaluate_compose_dialogue import (
    assignment,
    paths_for,
    score,
    summarize,
    confidence,
)
from scripts.evaluate_dialogue_evidence import predict
from scripts.dialogue_teacher_stories import absent
from scripts.prepare_compose_dialogue import MODEL, prepared
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_span_choice import gpu_owner


def check(device):
    rng = torch.Generator().manual_seed(6417)
    ids = torch.randint(0, 16, (63, 72), generator=rng)
    x = torch.randn(ids.shape, generator=rng) * 4
    ids[0, :] = 0
    ids[0, :3] = torch.tensor([8, 9, 8])
    x[0, :3] = torch.tensor([0.0, 0.5, 0.0])
    ids[1, :] = 0
    ids[1, :4] = torch.tensor([8, 9, 9, 8])
    x[1, :4] = 0.0
    ids[2, :] = 0
    ids[2, 1] = 4
    x[2, 1] = -20000.0
    out = modal_copy_position(
        x.to(device), copy_groups(ids.to(device)), ids.to(device) != 0
    ).tolist()
    for i, (row, s) in enumerate(zip(ids.tolist(), x.double(), strict=True)):
        probs = s.masked_fill(torch.tensor(row) == 0, -torch.inf).softmax(-1).tolist()
        masses = {k: sum(p for j, p in zip(row, probs) if j == k) for k in row if k}
        if out[i] != row.index(max(masses, key=masses.get)):
            raise ValueError("Word mass differs from enumeration")
    return len(ids)


def evaluate(root, control, out):
    out.mkdir(parents=True, exist_ok=False)
    data = prepared()
    result = {"checks": dict(cpu=check("cpu"), cuda=check("cuda")), "groups": {}}
    binding = json.loads((control / "results.json").read_text())
    for group in ("fresh", "lists"):
        ds = data[group]
        rows = ds["rows"]
        refs = [[r[5]] for r in rows]
        subsets = {
            "all": np.arange(len(rows)),
            "difficult": np.array(
                [
                    i
                    for i, r in enumerate(rows)
                    if absent(refs[i], (ds["pages"][r[1]], r[3], r[4]))
                ]
            ),
        }
        swaps = assignment(rows)
        stats = result["groups"][group] = {}
        for mode in ("ink", "byte"):
            face, unit, sep = representation(mode)
            decode = {face(chr(i)): chr(i) for i in range(32, 127)}
            paths = paths_for(mode, root)
            bank = corpus(ds, "fresh", load(paths["plain"], "cuda"), face, unit)
            for arm, folder in paths.items():
                n = load(folder, "cuda")
                n.marginal_copy = True
                rec = predict(folder, bank, n, sep, unit, decode, rows, swaps)
                name = f"{mode}/{arm}"
                source = control / f"{group}-{mode}-{arm}.json.gz"
                b = binding[group]["models"][name]
                if (
                    sha(source) != b["outputs"]
                    or sha(folder / "writer.safetensors") != b["weights"]
                ):
                    raise ValueError("Changed control")
                previous = json.loads(gzip.decompress(source.read_bytes()))
                new = score(rec, refs, None, None)
                old = score(previous, refs, None, None)
                summary, g = summarize(new, subsets)
                _, g0 = summarize(old, subsets)
                changes = {}
                for k, ix in subsets.items():
                    changes[k] = dict(
                        new=int((g[ix] & ~g0[ix]).sum()),
                        lost=int((g0[ix] & ~g[ix]).sum()),
                        f1=confidence(new[:, 0, 1] - old[:, 0, 1], rows, ix),
                    )
                path = out / source.name
                path.write_bytes(
                    gzip.compress(
                        json.dumps(rec, separators=(",", ":")).encode(), mtime=0
                    )
                )
                stats[name] = dict(
                    scores=summary,
                    changes=changes,
                    outputs=sha(path),
                    control=sha(source),
                )
        print(
            group,
            [(k, v["scores"]["difficult"]["grounded"]) for k, v in stats.items()],
            flush=True,
        )
    dump(out / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        evaluate(a.models, a.control, a.output)
