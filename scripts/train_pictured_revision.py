"""Fit the pictured source/draft editor and calibrate its edit threshold."""

from collections import Counter, defaultdict
import fcntl
import hashlib
import time

import numpy as np
from rapidfuzz.distance import Levenshtein
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from ilm.visual_lm.bitmap_alignment import blocks
from ilm.visual_lm.pictured_revision import PicturedReviser
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.train_aligned_visual_translation import (
    ROOT,
    MODELS,
    read_rows,
    tiles,
    sha,
    write_new,
)
from scripts.align_visual_translation import load, require

DRAFTS = ROOT / "artifacts/pictured_revision_drafts_v1"
OUT = ROOT / "artifacts/source_grounded_revision_v1"
COMPOSER = MODELS / "composer_55k.msgpack.gz"
MARGINS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 1e6)


def pad(rows):
    lens = np.asarray(list(map(len, rows)))
    ids = np.zeros((len(rows), max(lens)), np.int32)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
    return ids, lens


def training_data():
    protocol = load(ROOT / "references/evidence/pictured_revision_drafts_20261118.json")
    require(
        sha(DRAFTS / "drafts.npz") == protocol["output_sha256"], "Changed draft archive"
    )
    rows = read_rows("train")
    with np.load(DRAFTS / "drafts.npz") as d:
        bank, canonical = np.unique(tiles(), axis=0, return_inverse=True)
        require(np.array_equal(bank, d["bank"]), "Physical training bank differs")
        offset, images = d["offsets"], d["images"]
        require(
            len(rows) + 1 == len(offset) and offset[-1] == len(images),
            "Draft alignment",
        )
        entries = [
            (canonical[r["x"]], images[a:b], canonical[r["z"]])
            for r, a, b in zip(rows, offset, offset[1:])
        ]
    used = np.unique(np.concatenate([v for record in entries for v in record]))
    local = np.zeros(len(bank), np.int32)
    local[used] = np.arange(1, len(used) + 1)
    # Row zero denotes unrecognized or padded ink. Only fitted physical cells are shipped.
    bank = np.vstack((np.zeros((1, 128), np.uint8), bank[used]))
    counts = defaultdict(Counter)
    all_x, all_d, targets = [], [], []
    for x, d, y in entries:
        if not len(d):
            continue
        x, d, y = local[x].tolist(), local[d].tolist(), local[y].tolist()
        aligned = [tuple(part) for part in blocks(d, y)]
        for k, rep in zip(d, aligned):
            counts[k][rep] += 1
        all_x.append(x)
        all_d.append(d)
        targets.append(aligned)
    catalogue = {(): 0}
    options = np.full((len(bank), 12), -1, np.int32)
    prior = np.zeros(options.shape, np.float32)
    lookups = {}
    for k in range(len(bank)):
        alternatives = [
            (p, n)
            for p, n in counts[k].most_common()
            if len(p) <= 4 and p != (k,) and n >= 2
        ][:11]
        choices = [((k,), counts[k][(k,)] + 1)] + alternatives
        for j, (part, n) in enumerate(choices):
            options[k, j] = catalogue.setdefault(part, len(catalogue))
            prior[k, j] = np.log(n / choices[0][1])
        lookups[k] = {part: j for j, (part, _) in enumerate(choices)}
    recipes = np.zeros((len(catalogue), 4), np.int32)
    sizes = np.zeros(len(recipes), np.int32)
    for part, i in catalogue.items():
        recipes[i, : len(part)] = part
        sizes[i] = len(part)
    labels = []
    for draft, pieces in zip(all_d, targets):
        # Supervise actual complete edits. Keep the original for unsupported spans.
        labels.append([lookups[k].get(p, 0) for k, p in zip(draft, pieces)])
    x, n = pad(all_x)
    d, m = pad(all_d)
    y, _ = pad(labels)
    return (
        (bank, options, recipes, sizes, prior),
        (x, n, d, m, y),
        dict(
            total_rows=len(entries),
            nonempty_drafts=len(labels),
            fitted_images=len(bank) - 1,
            physical_actions=len(recipes),
            supported_positive_positions=int((y > 0).sum()),
            positions=int(m.sum()),
            available_revisions=int((options[:, 1:] >= 0).sum()),
        ),
    )


def composer():
    require(
        sha(COMPOSER)
        == load(ROOT / "references/evidence/clause_support_20261118/summary.json")[
            "model_sha256"
        ],
        "Composer changed",
    )
    return VisualPhraseComposer.restore(COMPOSER)


@torch.inference_mode()
def proposals(model, bank, sources, drafts):
    # All streams are represented as original packed pixels. Unknown source and
    # draft cells pass through the encoder even when they have no learned edit.
    from ilm.visual_lm.idiom_ink import visual_input

    device = model.bank.device
    answer = [None] * len(drafts)
    for start in range(0, len(drafts), 48):
        ids = [i for i in range(start, min(start + 48, len(drafts))) if len(drafts[i])]
        if not ids:
            continue

        def view(seq):
            lengths = torch.tensor([len(seq[i]) for i in ids], device=device)
            arr = np.zeros((len(ids), int(lengths.max()), 128), np.uint8)
            for j, i in enumerate(ids):
                arr[j, : len(seq[i])] = seq[i]
            pixels = torch.from_numpy(np.unpackbits(arr, axis=2)).to(device)
            return visual_input(pixels, model.landmarks), lengths

        a, n = view(sources)
        b, m = view(drafts)
        codes, _ = pad(
            [[model.indices.get(bytes(v), 0) for v in drafts[i]] for i in ids]
        )
        code = torch.from_numpy(codes).long().to(device)
        score = model(a, n, b, m, code)
        best, idx = score[:, :, 1:].max(-1)
        delta = (best - score[:, :, 0]).cpu().numpy()
        piece = model.candidates[code, idx + 1].long().clamp_min(0)
        recipe = model.recipes[piece].cpu().numpy()
        size = model.sizes[piece].cpu().numpy()
        for j, i in enumerate(ids):
            answer[i] = [
                (z, bank[k[:l]])
                for z, k, l in zip(delta[j, : len(drafts[i])], recipe[j], size[j])
            ]
    return answer


def apply_proposals(draft, choices, margin):
    if not len(draft):
        return draft
    return np.concatenate(
        [
            part if z > margin else original[None]
            for original, (z, part) in zip(draft, choices)
        ]
    )


def fit():
    OUT.mkdir()
    began = time.monotonic()
    torch.manual_seed(19478)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    (bank, opt, recipe, sizes, prior), data, info = training_data()
    landmarks = load_file(str(MODELS / "editor.safetensors"))["landmarks"]
    model = PicturedReviser(
        torch.from_numpy(bank),
        landmarks,
        *(torch.from_numpy(t) for t in (opt, recipe, sizes, prior)),
    ).cuda()
    x, n, d, m, y = [torch.as_tensor(v, device="cuda") for v in data]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
    rng = np.random.default_rng(489)
    losses = []
    steps = 0
    started = time.monotonic()
    for epoch in range(8):
        batches = []
        for chunk in np.array_split(rng.permutation(len(x)), 16):
            chunk = chunk[np.argsort(data[3][chunk], kind="stable")]
            batches.extend(chunk[i : i + 96] for i in range(0, len(chunk), 96))
        rng.shuffle(batches)
        loss_sum = 0.0
        for rows in batches:
            r = torch.as_tensor(rows, device="cuda")
            a, b = int(n[r].max()), int(m[r].max())
            xx = x[r, :a].long()
            dd = d[r, :b].long()
            valid = torch.arange(b, device="cuda") < m[r, None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = model(model.features[xx], n[r], model.features[dd], m[r], dd)
                loss = F.cross_entropy(scores[valid], y[r, :b][valid].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += loss.item() * len(rows)
            steps += 1
            if steps % 100 == 0:
                require(
                    torch.cuda.max_memory_allocated() < 10 * 2**30,
                    "GPU training reserve",
                )
                with open("/proc/meminfo") as f:
                    require(
                        int(
                            next(l for l in f if l.startswith("MemAvailable:")).split()[
                                1
                            ]
                        )
                        > 24 * 2**20,
                        "Host reserve",
                    )
        losses.append(loss_sum / len(x))
        print(epoch + 1, losses[-1], flush=True)
    train_time = time.monotonic() - started
    dev = sorted(
        read_rows("heldout"),
        key=lambda r: hashlib.sha256(
            f"revdev/{r['group']}/{r['ordinal']}".encode()
        ).digest(),
    )[:512]
    raw = tiles()
    baseline = composer()
    sources = [raw[r["x"]] for r in dev]
    drafts = [baseline.generate(v) for v in sources]
    gold = [tuple(map(bytes, raw[r["z"]])) for r in dev]
    choices = proposals(model.eval(), bank, sources, drafts)
    counts = []
    for margin in MARGINS:
        errors, edits = 0, 0
        for a, b, t in zip(drafts, choices, gold):
            output = apply_proposals(a, b, margin)
            ids = tuple(map(bytes, output))
            errors += Levenshtein.distance(ids, t)
            edits += int(not np.array_equal(output, a))
        counts.append((errors, edits, -margin))
    selected = -min(counts)[2]
    model.margin.fill_(selected)
    path = OUT / "reviser.safetensors"
    save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(path)
    )
    write_new(
        OUT / "fit.json",
        dict(
            **info,
            steps=steps,
            epochs=8,
            loss=losses,
            params=sum(p.numel() for p in model.parameters()),
            train_seconds=train_time,
            total_seconds=time.monotonic() - began,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            margin=selected,
            development=dict(
                count=len(dev),
                chapters=len({r["group"] for r in dev}),
                target_cells=sum(map(len, gold)),
                scores=[
                    dict(margin=-item[2], edits=item[0], changed=item[1])
                    for item in counts
                ],
            ),
            checkpoint=sha(path),
            composer=sha(COMPOSER),
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit()
