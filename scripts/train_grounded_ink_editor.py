"""Fit the visual writing policy with assigned or independently shuffled evidence."""

import argparse
import random
import numpy as np
from time import monotonic

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.book_participant_reader import InkBank, padded
from ilm.visual_lm.composed_answer_writer import COPY, WRITE, STOP
from ilm.visual_lm.grounded_ink_editor import GroundedInkEditor, pooled
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_grounded_ink_editor import PUBLIC, WORK, STYLES, read, sha, record
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

STEPS = 2400
BATCH = 48
SEED = 241175


def pool():
    path = WORK / "clauses.json"
    assert sha(path) == read(PUBLIC / "corpus.json")["pool"]
    bank = InkBank()
    face = Face(FACES[0])

    def cells(s):
        return bank.encode(face(s))

    rows = []
    for r in read(path):
        pos = r["text"].index(r["name"])
        rows.append((cells(r["text"]), pos, pos + len(r["name"]) - 1))
    glue = [cells("與。"), cells("，而。")]
    separators = cells("，。；：")
    styles = [cells(s) for s in STYLES]
    return rows, bank, glue, separators, styles


def examples(rng, rows, glue, separators, independent):
    pages = []
    regions = []
    modes = []
    targets = []
    for _ in range(BATCH):
        # Irrelevant neighbouring clauses prevent a fixed page offset from solving the writing task.
        parts = rng.sample(rows, 2 + rng.randrange(3))
        first, second = rng.sample(range(len(parts)), 2)
        page = []
        names = []
        clauses = []
        for glyphs, l, r in parts:
            if page:
                page.append(rng.choice(separators))
            start = len(page)
            page.extend(glyphs)
            clauses.append([start, len(page) - 1])
            names.append([start + l, start + r])
        mode = rng.randrange(2)
        random_order = rng.randrange(2)
        anchors = [names[first], names[second]]
        regions.append(anchors[::-1] if independent and random_order else anchors)
        subjects = (names, clauses)[mode]
        actions = [(COPY, *subjects[first], 0)]
        for cell in glue[mode][:-1]:
            actions.append((WRITE, 0, 0, cell))
        actions.extend(
            [
                (COPY, *subjects[second], 0),
                (WRITE, 0, 0, glue[mode][-1]),
                (STOP, 0, 0, 0),
            ]
        )
        pages.append(page)
        modes.append(mode)
        targets.append(actions)
    return pages, regions, modes, targets


def fit(arm):
    device = "cuda"
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    rows, bank, glue, separators, styles = pool()
    packed = np.frombuffer(b"".join(bank.cells), dtype=np.uint8).reshape(-1, 128)
    bits = torch.from_numpy(np.unpackbits(packed, axis=1)).to(device)
    shapes = pooled(bits)
    s = padded(styles, device)
    model = GroundedInkEditor().to(device)
    out = WORK / arm
    out.mkdir(exist_ok=False)

    def save(path):
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(path),
        )

    save(out / "initial.safetensors")
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=0.00015)
    start = monotonic()
    energy = snapshot()
    history = []
    total = 0.0
    for step in range(1, STEPS + 1):
        pages, regions, modes, targets = examples(
            rng, rows, glue, separators, arm == "independent"
        )
        ids = padded(pages, device)
        valid = ids != 0
        inst = s[modes]
        pix = shapes[ids]
        x, state = model.encode(
            pix, valid, torch.tensor(regions, device=device), shapes[inst], inst != 0
        )
        seq = torch.tensor(
            [t + [(STOP, 0, 0, 0)] * (6 - len(t)) for t in targets], device=device
        )
        active = torch.ones(BATCH, dtype=torch.bool, device=device)
        pos = torch.arange(ids.shape[1], device=device)[None]
        loss = state.new_zeros(())
        for actions in seq.unbind(1):
            act, a, b, w = actions.unbind(1)
            action, bounds, raster = model.decide(x, valid, state)
            copy = act == COPY
            write = act == WRITE
            loss += (F.cross_entropy(action, act, reduction="none") * active).mean()
            end = bounds[:, 1].masked_fill(pos < a[:, None], -1e4)
            location = F.cross_entropy(
                bounds[:, 0], a, reduction="none"
            ) + F.cross_entropy(end, b, reduction="none")
            loss += (location * copy).mean()
            raster_loss = F.binary_cross_entropy_with_logits(
                raster,
                bits[w].float(),
                pos_weight=raster.new_tensor(3.0),
                reduction="none",
            ).mean(1)
            loss += (raster_loss * write).mean() * 2
            span = (pos >= a[:, None]) & (pos <= b[:, None])
            feedback = (pix * span[:, :, None]).sum(1) / (b - a + 1)[:, None]
            state = model.advance(
                state, act, torch.where(write[:, None], shapes[w], feedback)
            )
            active = active & (act != STOP)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        total += loss.item()
        if step % 200 == 0:
            history.append(total / 200)
            total = 0.0
            print(arm, step, history[-1], flush=True)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 8 * 1024**3
            ):
                raise RuntimeError("Editor ceiling exceeded")
    elapsed = monotonic() - start
    cost = consumed(energy, snapshot())
    save(out / "editor.safetensors")
    record(
        PUBLIC / (arm + "_fit.json"),
        dict(
            initial=sha(out / "initial.safetensors"),
            weights=sha(out / "editor.safetensors"),
            pool=sha(WORK / "clauses.json"),
            steps=STEPS,
            examples=STEPS * BATCH,
            parameters=sum(p.numel() for p in model.parameters()),
            seconds=elapsed,
            energy=cost,
            peak_bytes=torch.cuda.max_memory_allocated(),
            loss=history,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("aligned", "independent"))
    with gpu_owner():
        torch.set_num_threads(2)
        fit(ap.parse_args().arm)
