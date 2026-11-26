"""Fit matched image writers after independent certification of the bilingual training family."""

from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from scripts.prepare_opus_paired_supervision import (
    ROOT,
    PUB,
    WORK,
    SEED,
    base,
    gpu_owner,
    configure_cuda_cli,
)
from scripts.reserve_paired_branch_books import RULE
from scripts.evaluate_proposition_translation import QWEN, JUDGE, MODELS
from scripts.train_visual_training_breadth import (
    restore as broad_writer,
    corpus as rehearsal,
    COHORT,
    PASSES,
    MODEL as BROAD,
)
from scripts.train_glyph_draft_editor import atomic_save
from scripts.write_glyph_draft_editor import pictured_input, pack
from scripts.write_covered_phrases import CoveredWriter

MODEL = ROOT / "models/paired_branch_writer"
RATES = (0.00010, 0.00008, 0.00006, 0.00005, 0.00004, 0.00003)


def qualified_ids():
    if not (PUB / "confirmation.json").exists():
        raise RuntimeError("Independent corpus confirmation is still pending")
    primary = base.read(PUB / "primary.json")
    final = base.read(PUB / "confirmation.json")
    ids = primary["train"]
    primary_votes = np.load(PUB / "primary.npz")
    final_votes = np.load(PUB / "confirmation.npz")
    assert primary["model"] == MODELS[QWEN] and final["model"] == MODELS[JUDGE]
    native = base.read(PUB / "native.json")
    assert primary_votes["ids"].tolist() == native["queue"]
    assert native["selection"] == base.sha(PUB / "selection.json")
    assert primary_votes["votes"].shape == (len(native["queue"]), 2, 2, 2)
    assert final_votes["votes"].shape == (len(primary["reserved"]), 2, 2, 2)
    align = np.array([1, 2])[None, None, :, None]
    eligible = {
        int(k)
        for k, v in zip(
            primary_votes["ids"],
            (primary_votes["votes"] == align).all(axis=(1, 2, 3)),
            strict=True,
        )
        if v
    }
    assert final["parent"] == base.sha(PUB / "primary.json") and primary[
        "parent"
    ] == base.sha(PUB / "native.json")
    assert final["votes"] == base.sha(PUB / "confirmation.npz") and primary[
        "votes"
    ] == base.sha(PUB / "primary.npz")
    assert (
        len(set(ids)) == len(ids)
        and set(ids).isdisjoint(primary["reserved"])
        and eligible == set(ids) | set(primary["reserved"])
    )
    assert (
        final_votes["ids"].tolist() == primary["reserved"]
        and len(primary["reserved"]) == 128
    )
    confirmed = int((final_votes["votes"] == align).all(axis=(1, 2, 3)).sum())
    groups = base.read(PUB / "selection.json")["groups"]
    edit_types = len({tuple(sorted(groups[i]["words"])) for i in ids})
    if not (
        final["passed"]
        and confirmed == final["approved"]
        and confirmed >= 112
        and len(ids) >= 512
        and edit_types >= 100
    ):
        raise ValueError(
            f"Teacher qualification failed: {confirmed}/128 confirmations, {len(ids)} training families, {edit_types} changes"
        )
    return ids


def bindings():
    files = (
        Path(__file__),
        RULE,
        PUB / "native.json",
        PUB / "primary.json",
        PUB / "confirmation.json",
        PUB / "books.json",
        BROAD / "broad/fit.json",
        ROOT / "scripts/train_visual_training_breadth.py",
    )
    return {str(p.relative_to(ROOT)): base.sha(p) for p in files}


def restore(arm, device="cpu"):
    writer = broad_writer("broad", device)
    if arm == "parent":
        return writer
    if arm not in ("positive", "paired"):
        raise ValueError(arm)
    folder = MODEL / arm
    record = base.read(folder / "fit.json")
    assert record["bindings"] == bindings() and record["weights"] == base.sha(
        folder / "weights.safetensors"
    )
    writer.model.load_state_dict(load_file(str(folder / "weights.safetensors")))
    return writer


def prepare_families(ids, writer):
    face = base.Ink(writer.palette)
    rows = base.read(WORK / "families.json")
    assert base.sha(WORK / "families.json") == base.read(PUB / "selection.json")["text"]
    corpus = np.zeros((len(ids), 2, 2, 96), dtype=np.int32)
    drafts = [[], []]
    branch = np.zeros((len(ids), 2), dtype=np.int64)
    for t in (0, 1):
        engine = CoveredWriter(t)
        for j, i in enumerate(ids):
            target = []
            for side, pair in enumerate(rows[i]):
                source = face.encode(pair[1 - t])
                cells = face.encode(pair[t])
                target.append(cells + [1])
                corpus[j, side, t, : len(cells)] = cells
                image = engine.palette[source]
                optical = engine.translate(image)[0]
                drafts[t].append(writer.addresses(optical) or [0])
            branch[j, t] = next(k for k, (a, b) in enumerate(zip(*target)) if a != b)
    assert not face.bad
    return (
        torch.as_tensor(corpus, device="cuda"),
        [pack(ds, "cuda") for ds in drafts],
        torch.as_tensor(branch, device="cuda"),
    )


def merge_width(first, second):
    width = max(first.size(1), second.size(1))
    return torch.cat([F.pad(x, (0, width - x.size(1))) for x in (first, second)])


def stopped(ids):
    n = (ids != 0).sum(1)
    return F.pad(ids[:, : n.max()], (0, 1)).scatter_(1, n[:, None], 1)


def branch_losses(law, desired, column):
    """Score paired alternatives after the same printed prefix on each source."""
    row = torch.arange(len(column), device=column.device)
    branch_law = law[row, column]
    true_glyph = desired[row, column]
    aligned = branch_law[row, true_glyph]
    crossed = branch_law[row, true_glyph[row ^ 1]]
    interaction = (aligned - crossed).reshape(-1, 2).mean(1)
    return -aligned.mean(), F.softplus(2 * (1 - interaction)).mean() / 2


def fit(arm, fresh, old):
    folder = MODEL / arm
    if (folder / "fit.json").exists():
        restore(arm)
        return
    writer = restore("parent", "cuda")
    torch.manual_seed(SEED)
    model = writer.model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=RATES[0], fused=True)
    pairs, ds, branch = fresh
    historical, _, prior_ds, _ = old
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    history = []
    real = COHORT * PASSES
    for ep, rate in enumerate(RATES):
        rng = np.random.default_rng(SEED + ep + 31)
        order = rng.permutation(len(pairs))
        opt.param_groups[0]["lr"] = rate
        schedule = sha256()
        positions = decisions = updates = 0
        likelihood = branch_nll = rank_sum = 0.0
        for offset in range(0, len(order), 16):
            group = order[offset : offset + 16]
            replay = np.concatenate(
                (
                    rng.choice(real, 32, replace=False),
                    real + rng.choice(len(historical) - real, 16, replace=False),
                )
            )
            flat = (group[:, None] * 2 + np.arange(2)).ravel()
            for seq in (group, replay):
                schedule.update(seq.astype("<i4").tobytes())
            n = len(group)
            logloss = 0
            positive = 0
            contrast = 0
            total = 0
            for t in (0, 1):
                fresh_ids = pairs[group].reshape(2 * n, 2, 96)
                seq = torch.cat((fresh_ids, historical[replay])).long()
                source = seq[:, 1 - t]
                source = source[:, : (source != 0).sum(1).max()]
                with torch.no_grad():
                    marker = (
                        (
                            255
                            * writer.marker(writer.features, source).sigmoid()
                            * source.ne(0).unsqueeze(-1)
                        )
                        .round()
                        .byte()
                    )
                    seen = pictured_input(source, marker)
                target = stopped(seq[:, t])
                draft = merge_width(ds[t][flat], prior_ds[t][replay]).long()
                draft = draft[:, : max(1, int(draft.ne(0).sum(1).max()))]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    law = model(
                        writer.features,
                        seen,
                        target,
                        writer.inv[t],
                        t,
                        draft,
                        writer.local[t],
                    )
                desired = writer.local[t][target]
                logloss = logloss + F.nll_loss(
                    law.transpose(1, 2), desired, ignore_index=-100, reduction="sum"
                )
                total += int((target != 0).sum())
                col = branch[group, t].repeat_interleave(2)
                positive_t, contrast_t = branch_losses(law, desired, col)
                positive = positive + positive_t / 2
                contrast = contrast + contrast_t / 2
            loss = logloss / total + 0.25 * positive
            if arm == "paired":
                loss = loss + 0.25 * contrast
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            likelihood += logloss.detach().item()
            branch_nll += positive.detach().item() * n
            rank_sum += contrast.detach().item() * n
            positions += total
            decisions += 4 * n
            updates += 1
            if (
                monotonic() - start > 1800
                or torch.cuda.max_memory_allocated() > 16 * 2**30
            ):
                raise RuntimeError("Training resource ceiling exceeded")
        history.append(
            dict(
                epoch=ep + 1,
                schedule=schedule.hexdigest(),
                updates=updates,
                positions=positions,
                branch_decisions=decisions,
                nll=likelihood / positions,
                branch_nll=branch_nll / len(order),
                contrast_loss=rank_sum / len(order),
            )
        )
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / "weights.safetensors"
    atomic_save(
        lambda name: save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            name,
        ),
        path,
    )
    base.dump(
        folder / "fit.json",
        dict(
            arm=arm,
            weights=base.sha(path),
            epochs=history,
            bindings=bindings(),
            seconds=monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),
        ),
    )
    print(arm, history[-1], flush=True)


def main():
    ids = qualified_ids()
    assert base.read(PUB / "books.json")["protocol"] == base.sha(RULE)
    configure_cuda_cli()
    with gpu_owner():
        writer = restore("parent", "cpu")
        fresh = prepare_families(ids, writer)
        old = rehearsal()
        for arm in ("positive", "paired"):
            fit(arm, fresh, old)
        a, b = [
            base.read(MODEL / m / "fit.json")["epochs"] for m in ("positive", "paired")
        ]
        assert all(
            x[k] == y[k]
            for x, y in zip(a, b, strict=True)
            for k in ("updates", "positions", "branch_decisions", "schedule")
        )


if __name__ == "__main__":
    main()
