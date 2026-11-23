"""Train wide, depth-one and randomly supervised readers against the same visual features."""

import argparse
from pathlib import Path
from collections import defaultdict
from time import monotonic

from safetensors.torch import save_file
import torch

from ilm.visual_lm.span_list import SpanListReader, SpanRanker, image_choice
from scripts.prepare_screened_span_list import CACHE, PUBLIC, SEED, split
from scripts.prepare_visual_facts import Face, FONTS, ROOT, write_json
from scripts.train_pointing_causal_writer import sha
from scripts.train_span_choice import gpu_owner, model

WEIGHTS = ROOT / "models/span_list_screened"


def collect(reader, face, items):
    blocks = []
    for start in range(0, len(items), 24):
        chunk = items[start : start + 24]
        x, valid, classes, fields = reader.proposals(
            [(face(v["p"]), face(v["q"])) for v in chunk]
        )
        labels = []
        for v, (_, _, options) in zip(chunk, fields, strict=True):
            refs = {face(r) for r in v["refs"]}
            labels.append([o in refs for o in options])
        blocks.append((x.cpu(), valid.cpu(), classes.cpu(), torch.tensor(labels)))
        if start % 2400 == 0:
            print("pictured", start, len(items), flush=True)
    return tuple(torch.cat(parts).to("cuda") for parts in zip(*blocks))


def fit(train, val, items, arm, destination):
    torch.manual_seed(SEED)
    head = SpanRanker().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.00065, weight_decay=0.02)
    x, valid, _, gold = train
    vx, vm, classes, vg = val
    if arm == "first":
        col = torch.arange(x.size(1), device=x.device) < 4
        valid, vm = valid & col, vm & col
    y = gold & valid
    if arm == "shuffled":
        # Preserve coverage and positive counts; randomize which valid positions receive credit.
        noise = torch.rand(y.shape, device=x.device).masked_fill(~valid, 2.0)
        y = noise.argsort(1).argsort(1) < y.sum(1)[:, None]
    useful = y.any(1).nonzero().flatten()
    natural = [i for i, v in enumerate(items) if v["group"] != "paired"]
    boards = defaultdict(list)
    for i, v in enumerate(items):
        if v["group"] == "paired":
            boards[v["key"]].append(i)
    assert len(boards) == 96 and all(len(b) == 4 for b in boards.values())
    board_indices = torch.tensor(list(boards.values()), device=x.device)
    history = []
    best = -1.0
    for epoch in range(16):
        total = 0.0
        indices = useful[torch.randperm(len(useful), device=x.device)]
        for take in indices.split(96):
            logits = head(x[take], valid[take])
            loss = (
                logits.logsumexp(1)
                - logits.masked_fill(~y[take], -torch.inf).logsumexp(1)
            ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(take)
        with torch.no_grad():
            sel = image_choice(head(vx, vm), classes)
            ok = vg.gather(1, sel[:, None]).flatten()
            natural_hits = int(ok[natural].sum())
            board_hits = int(ok[board_indices].all(1).sum())
            metric = natural_hits / len(natural) + board_hits / len(boards)
            history.append(
                dict(
                    epoch=epoch + 1,
                    loss=total / len(useful),
                    natural=natural_hits,
                    boards=board_hits,
                )
            )
            if metric > best:
                best = metric
                selected = epoch + 1
                state = {
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                }
    path = destination / (arm + ".safetensors")
    save_file(state, str(path))
    return dict(
        weights=sha(path),
        selected=selected,
        history=history,
        covered=len(useful),
        parameters=sum(v.numel() for v in head.parameters()),
    )


def main(run_dir=None):
    receipt = (run_dir or PUBLIC) / "fit.json"
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=False)
    if receipt.exists():
        raise FileExistsError("Completed fit")
    start = monotonic()
    WEIGHTS.mkdir(exist_ok=True)
    reader = SpanListReader(model())
    face = Face((FONTS[0], 2))
    val_rows = split("validation")
    train, val = [collect(reader, face, items) for items in (split("train"), val_rows)]
    prepared = monotonic()
    arms = {
        arm: fit(
            train, val, val_rows, arm, run_dir or (WEIGHTS if arm == "wide" else CACHE)
        )
        for arm in ("wide", "first", "shuffled")
    }
    write_json(
        receipt,
        dict(
            arms=arms,
            feature_seconds=prepared - start,
            fit_seconds=monotonic() - prepared,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(),
        ),
    )
    print("complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Write a new independent fit and controls to a new directory",
    )
    args = parser.parse_args()
    with gpu_owner():
        main(args.run_dir)
