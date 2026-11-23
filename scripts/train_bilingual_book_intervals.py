"""Adapt a visual interval reader on matched and exchanged bilingual book requests."""

import argparse
from collections import Counter
import json
import os
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from scripts.mine_stopped_book_replies import BOOK, PUBLIC as PRIOR
from scripts.prepare_bilingual_book_intervals import WORK, PUBLIC
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES
from scripts.train_visual_facts import READER

WIDTH = 48
SEED = 241129


def data():
    receipt = json.loads((PUBLIC / "preparation.json").read_text())
    source, binding = BOOK / "train.jsonl", WORK / "training.json"
    assert (sha(source), sha(binding)) == (
        receipt["source_sha"],
        receipt["training_sha"],
    )
    specs = {v["id"]: v for v in json.loads(binding.read_text())}
    # Preserve whole original exchanges so the English input multiset is identical.
    keys = set(specs)
    while True:
        complete = {k for k in keys if specs[k]["swapped"] in keys}
        if complete == keys:
            break
        keys = complete
    assert Counter(specs[k]["swapped"] for k in keys) == Counter(keys)
    face = Face(FACES[0])
    # 0 and 1 are the classifier and separator images' learned boundary vectors.
    tiles = {}

    def indices(text):
        return tuple(
            tiles.setdefault(cell, len(tiles) + 2) for cell in in_cells(face(text))
        )

    rows = []
    for line in source.read_text().splitlines():
        v = json.loads(line)
        page = indices(v["page"])
        questions = {q["id"]: q for q in v["questions"]}
        for k, q in questions.items():
            if k not in keys:
                continue
            spec = specs[k]
            zh, en, other = [
                indices(s) for s in (q["zh"], q["en"], questions[spec["swapped"]]["en"])
            ]
            rows.extend(
                [
                    (page, zh, zh, spec["intervals"]),
                    (page, en, other, spec["intervals"]),
                ]
            )
    assert len(rows) == 2 * len(keys)
    return rows, tiles, len(specs) - len(keys)


def interval_loss(start, end, mask, targets):
    """Marginal likelihood of any accepted occurrence among valid bounded spans."""
    ends = end.masked_fill(~mask, -torch.inf)
    values = start[:, :, None] + F.pad(ends, (0, WIDTH - 1), value=-torch.inf).unfold(
        1, WIDTH, 1
    )
    values = values.masked_fill(~mask[:, :, None], -torch.inf)
    batch, starts, offsets = zip(
        *[(i, a, b - a) for i, options in enumerate(targets) for a, b in options],
        strict=True,
    )
    allowed = torch.zeros_like(values, dtype=torch.bool)
    allowed[batch, starts, offsets] = True
    z = values.flatten(1).logsumexp(1)
    gold = values.masked_fill(~allowed, -torch.inf).flatten(1).logsumexp(1)
    return (z - gold).mean()


def step(reader, bank, batch, arm):
    device = bank.device
    q_index = 1 if arm == "aligned" else 2
    qlens = [len(row[q_index]) for row in batch]
    passages = [r[0] for r in batch]
    ids = [torch.tensor((0, *r[q_index], 1, *r[0], 1), device=device) for r in batch]
    lengths = torch.tensor([len(s) for s in ids], device=device)
    ids = pad_sequence(ids, batch_first=True)
    pos = torch.arange(ids.shape[1], device=device)[None, :]
    begin = torch.tensor(qlens, device=device)[:, None] + 2
    mask = pos < lengths[:, None]
    logits = reader.encoder(
        inputs_embeds=bank[ids],
        attention_mask=mask,
        token_type_ids=((pos >= begin) & mask).long(),
    )
    size = max(map(len, passages))
    offsets = torch.arange(size, device=device)[None, :]
    page_mask = offsets < torch.tensor(list(map(len, passages)), device=device)[:, None]
    at = (begin + offsets).clamp_max(ids.shape[1] - 1)
    return interval_loss(
        logits.start_logits.gather(1, at).float(),
        logits.end_logits.gather(1, at).float(),
        page_mask,
        [r[3] for r in batch],
    )


def fit(arm, rows, tiles, removed):
    dest = WORK / arm
    if dest.exists():
        raise FileExistsError(dest)
    before = json.loads((PRIOR / "span_baseline.json").read_text())["weights"]
    assert all(sha(READER / n) == h for n, h in before.items())
    torch.manual_seed(SEED)
    reader = FontInvariantReader.restore(READER, "cuda").eval()
    reader.requires_grad_(False)
    # Retain the first eight contextual layers and the learned optical interface.
    reader.encoder.bert.encoder.layer[-4:].requires_grad_(True)
    reader.encoder.qa_outputs.requires_grad_(True)
    with torch.no_grad():
        bank = torch.cat((reader.markers[:2], reader.embed(b"".join(tiles))))
    opt = torch.optim.AdamW(
        [p for p in reader.parameters() if p.requires_grad], lr=2e-5, fused=True
    )
    rng = np.random.default_rng(SEED)
    batches = []
    for group in np.array_split(
        rng.permutation(len(rows)), int(np.ceil(len(rows) / 512))
    ):
        group = sorted(
            group, key=lambda i: len(rows[i][0]) + max(len(rows[i][1]), len(rows[i][2]))
        )
        batches.extend([group[j : j + 24] for j in range(0, len(group), 24)])
    started = monotonic()
    torch.cuda.reset_peak_memory_stats()
    total = 0.0
    for n, ids in enumerate(batches, 1):
        batch = [rows[i] for i in ids]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = step(reader, bank, batch, arm)
        if not torch.isfinite(loss):
            raise ValueError("Invalid interval mass")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            reader.parameters(), 1.0, error_if_nonfinite=True
        )
        opt.step()
        total += loss.item() * len(batch)
        if n % 200 == 0:
            print(arm, n, len(batches), loss.item(), flush=True)
        if (
            monotonic() - started > 1200
            or torch.cuda.max_memory_allocated() > 20 * 2**30
        ):
            raise RuntimeError("Fit exceeded its resource bound")
    temp = WORK / (arm + ".partial")
    temp.mkdir()
    for name in ("vision.safetensors", "metric.safetensors", "config.json"):
        os.link(READER / name, temp / name)
    save_file(
        {
            k: v.detach().cpu().contiguous()
            for k, v in reader.encoder.state_dict().items()
        },
        str(temp / "context.safetensors"),
    )
    report = dict(
        arm=arm,
        seed=SEED,
        rows=len(rows),
        updates=len(batches),
        pair_cycle_exclusions=removed,
        distinct_input_glyphs=len(tiles),
        trainable=sum(p.numel() for p in reader.parameters() if p.requires_grad),
        parameters=sum(p.numel() for p in reader.parameters()),
        mean_train_nll=total / len(rows),
        training_seconds=monotonic() - started,
        peak_bytes=torch.cuda.max_memory_allocated(),
        weights={p.name: sha(p) for p in temp.iterdir()},
        parent=before,
    )
    record(temp / "fit.json", report)
    temp.rename(dest)
    record(PUBLIC / (arm + "_fit.json"), report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", choices=("aligned", "exchanged"), nargs="+")
    with gpu_owner():
        rows, tiles, removed = data()
        for arm in ap.parse_args().arms:
            fit(arm, rows, tiles, removed)
