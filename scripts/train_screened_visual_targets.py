"""Compare identical one-pass visual writers after changing targets on reviewed source rows."""

import csv
import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.ordered_copy_anchors import forced_matches
from scripts.prepare_screened_visual_targets import (
    EVIDENCE,
    OUT,
    SEED,
    ROOT,
    load,
    write_new,
)
from scripts.prepare_phrase_boundary_trial import Rasterizer, sha, order
from scripts.train_anchor_gap_writer import prepare, targets, DATA
from scripts.train_aligned_visual_translation import MODELS

PARENT = MODELS / "anchor_gap_writer.safetensors"
ACCEPTED = EVIDENCE / "accepted.json"


def schedule(n, reviewed):
    rng = np.random.default_rng(SEED)
    replay = rng.permutation(n).reshape(-1, 48)
    need = len(replay) * 16
    stream = np.concatenate(
        [
            rng.permutation(reviewed)
            for _ in range((need + len(reviewed) - 1) // len(reviewed))
        ]
    )[:need]
    return np.concatenate([replay, stream.reshape(-1, 16)], axis=1)


def register():
    test = load(EVIDENCE / "holdout.json")
    rows = test["rows"].copy()
    taken = set()
    reader = []
    for band in range(4):
        choices = sorted(
            (i for i, r in enumerate(rows) if r["length_band"] == band),
            key=lambda i: order("screen-read-1/" + rows[i]["source_sha"]),
        )
        ids = []
        for i in choices:
            if rows[i]["chapter"] not in taken:
                ids.append(i)
                taken.add(rows[i]["chapter"])
                if len(ids) == 4:
                    break
        assert ids
        reader.extend(ids)
    with (EVIDENCE / "contrast_pairs.tsv").open() as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    assert len(pairs) == 12
    render = Rasterizer()
    with np.load(DATA / "training.npz") as z:
        bank = z["cells"]
        src = z["source"]
        xs = z["x_offsets"]
        training = {
            hashlib.sha256(bank[src[a:b]].tobytes()).digest()
            for a, b in zip(xs[:-1], xs[1:])
        }
    for pair_id, pair in enumerate(pairs):
        for side in ("first", "second"):
            x = pair[side]
            key = hashlib.sha256(b"".join(map(render.tile, x))).digest()
            assert key not in training
            reader.append(len(rows))
            rows.append(
                dict(
                    source=x,
                    target=pair[side + "_meaning"],
                    category=pair["category"],
                    pair=pair_id,
                    side=side,
                    source_sha=key.hex(),
                )
            )
    assert len({r["source_sha"] for r in rows}) == len(rows)
    selected = load(ACCEPTED)
    batches = schedule(len(xs) - 1, [r["row"] for r in selected])
    assert batches.shape == (1164, 64) and np.array_equal(
        np.sort(batches[:, :48].flatten()), np.arange(55872)
    )
    write_new(
        EVIDENCE / "registration.json",
        dict(
            rows=rows,
            archive=test["selected"],
            reader=reader,
            batches_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),
            parent_sha=sha(PARENT),
            training_sha=sha(DATA / "training.npz"),
            approved_sha=sha(ACCEPTED),
            test_sha=sha(EVIDENCE / "holdout.json"),
            reviewed_rows=len(selected),
            teacher_sha=sha(EVIDENCE / "teacher_drafts.json"),
            contrast_sha=sha(EVIDENCE / "contrast_pairs.tsv"),
        ),
    )


def data_sets():
    packed, data = prepare(DATA / "training.npz")
    raw = list(map(bytes, packed))
    ix = {v: i for i, v in enumerate(raw)}
    render = Rasterizer()
    accepted = load(ACCEPTED)
    max_target = max(data[1].shape[1], max(len(r["target"]) for r in accepted))
    x, y, n, m, keep, left, right, off, close = data

    def widen(a, size):
        return np.pad(a, ((0, 0), (0, size - a.shape[1]))) if size > a.shape[1] else a

    original = (
        x,
        widen(y, max_target),
        n,
        m,
        keep,
        *[widen(a, max_target + 1) for a in (left, right, off, close)],
    )
    changed = [a.copy() if k not in (0, 2) else a for k, a in enumerate(original)]
    _, new_y, _, new_m, new_keep, new_l, new_r, new_off, new_c = changed
    same = 0
    for row in accepted:
        i = row["row"]
        observed = x[i, : n[i]].tolist()
        assert b"".join(raw[v] for v in observed) == b"".join(
            map(render.tile, row["source"])
        )
        desired = []
        for c in row["target"]:
            b = render.tile(c)
            if b not in ix:
                ix[b] = len(raw)
                raw.append(b)
            desired.append(ix[b])
        size = len(desired)
        same += desired == y[i, : m[i]].tolist()
        new_y[i].fill(0)
        new_y[i, :size] = desired
        new_m[i] = size
        _, matched = forced_matches(observed, desired)
        k, l, r, o, c = targets(len(observed), size, matched)
        new_keep[i, : len(k)] = k
        for ar, v in ((new_l, l), (new_r, r), (new_off, o), (new_c, c)):
            ar[i].fill(0)
            ar[i, : size + 1] = v
    return (
        np.frombuffer(b"".join(raw), np.uint8).reshape(-1, 128),
        original,
        tuple(changed),
        same,
    )


def fit(arm, packed, data, maximum, batches, manifest):
    start = time.monotonic()
    model_path = OUT / (arm + ".safetensors")
    if model_path.exists() or (EVIDENCE / (arm + "_fit.json")).exists():
        raise FileExistsError(model_path)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    state = load_file(str(PARENT))
    model = AnchorGapWriter(state["landmarks"], state["exemplars"])
    model.load_state_dict(state)
    model.cuda()
    indexes = {bytes(v): i for i, v in enumerate(packed)}
    lookup = torch.full((len(packed),), -1, dtype=torch.long)
    for j, b in enumerate(np.packbits(state["exemplars"].numpy(), axis=1)):
        lookup[indexes[bytes(b)]] = j
    lookup = lookup.cuda()
    bank = torch.from_numpy(np.unpackbits(packed, axis=1)).cuda().bool()
    features = visual_input(bank, model.landmarks)
    x, y, n, m, keep, left, right, off, close = (
        torch.as_tensor(a, device="cuda") for a in data
    )
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.0002
    )
    records = []
    loss_sum = 0.0
    for step, b in enumerate(batches, 1):
        idx = torch.as_tensor(b, device="cuda")
        nx, ny = data[2][b].max(), maximum[b].max()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(
                features,
                x[idx, :nx].long(),
                y[idx, :ny].long(),
                n[idx],
                m[idx],
                keep[idx, :nx],
                left[idx, : ny + 1].long(),
                right[idx, : ny + 1].long(),
                off[idx, : ny + 1].long(),
                close[idx, : ny + 1],
                lookup,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        loss_sum += loss.item()
        if step % 194 == 0:
            mem = next(
                a
                for a in Path("/proc/meminfo").read_text().splitlines()
                if a.startswith("MemAvailable:")
            )
            if (
                int(mem.split()[1]) < 24 * 2**20
                or torch.cuda.max_memory_allocated() > 10 * 2**30
                or time.monotonic() - start > 1800
            ):
                raise RuntimeError("Resource ceiling")
            records.append(loss_sum / 194)
            loss_sum = 0.0
            print(arm, step, records[-1], flush=True)
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
        str(model_path),
    )
    write_new(
        EVIDENCE / (arm + "_fit.json"),
        dict(
            arm=arm,
            parent_sha=manifest["parent_sha"],
            schedule_sha=manifest["batches_sha256"],
            checkpoint_sha=sha(model_path),
            updates=len(batches),
            source_exposures=batches.size,
            source_cells=int(data[2][batches].sum()),
            target_cells=int(data[3][batches].sum()),
            reviewed_oversamples=batches[:, 48:].size,
            loss_by_194_steps=records,
            seconds=time.monotonic() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
        ),
    )


def train():
    reg = load(EVIDENCE / "registration.json")
    assert (
        sha(PARENT)
        == reg["parent_sha"]
        == "71053d75fd2c323c305b7a36c7f49d2fd575d3c99373ac3c5d184b43e70575da"
    )
    assert (
        sha(DATA / "training.npz") == reg["training_sha"]
        and sha(ACCEPTED) == reg["approved_sha"]
    )
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    packed, old, new, same = data_sets()
    ids = [r["row"] for r in load(ACCEPTED)]
    batches = schedule(len(old[0]), ids)
    assert hashlib.sha256(batches.tobytes()).hexdigest() == reg["batches_sha256"]
    maximum = np.maximum(old[3], new[3])
    write_new(
        EVIDENCE / "training_account.json",
        dict(
            same_targets=same,
            changed_targets=len(ids) - same,
            rows=len(old[0]),
            images=len(packed),
        ),
    )
    for name, data in (("book", old), ("screened", new)):
        fit(name, packed, data, maximum, batches, reg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "train"))
    if parser.parse_args().action == "register":
        register()
    else:
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train()
