"""Train the name-conditioned interval reader and its matched controls."""

from collections import Counter
import hashlib
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scripts.train_prose_fields import prepare, train_rows, load_model
from scripts.prepare_prose_fields import (
    PUBLIC,
    WEIGHTS,
    LIMIT,
    read,
    dump,
    sha,
    FIELDS,
    DOMAINS,
)
from scripts.prose_entity_data import pairs, page
from scripts.train_semantic_page import gpu_owner, snapshot, consumed
from ilm.visual_lm.prose_fields import ProseFields, interval_loss, name_position
from ilm.visual_lm.book_prefix import observations


def model(arm, device, folder=WEIGHTS, receipts=PUBLIC):
    net = ProseFields(len(FIELDS), len(DOMAINS), entity=True).to(device)
    path = folder / (arm + ".safetensors")
    if sha(path) != read(receipts / (arm + "-fit.json"))["weights"]:
        raise ValueError("Changed referent weights")
    net.load_state_dict(load_file(str(path), device=device))
    return net.eval()


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path)
    args = ap.parse_args()
    folder, receipts = (
        (args.run / "weights", args.run / "scores") if args.run else (WEIGHTS, PUBLIC)
    )
    rows, n = train_rows()
    examples = pairs(rows[:n], 8701)
    # Each source combination occurs with both target identities; page order is counterbalanced.
    comp = [
        page(a, b, t, k)
        for i, (x, y, k) in enumerate(examples)
        for a, b in [(x, y) if i % 2 else (y, x)]
        for t in (a, b)
    ]
    combined = rows + comp
    anchors = np.zeros((len(combined), LIMIT, 3), np.float32)
    for i, r in enumerate(combined):
        s = r["text"]
        name = r["name"]
        start = s.index(name) if s.count(name) == 1 else None
        anchors[i, : len(s)] = name_position(len(s), start, len(name))
    with gpu_owner():
        bank, (x, y, lens, domains) = prepare(combined)
        loc = torch.as_tensor(anchors, device="cuda")
        for arm in ("entity-ink", "entity-erased", "entity-byte"):
            if (receipts / (arm + "-fit.json")).exists():
                model(arm, "cpu", folder, receipts)
                continue
            mode = "byte" if arm.endswith("byte") else "ink"
            torch.manual_seed(7491)
            rng = np.random.default_rng(88031)
            net = ProseFields(len(FIELDS), len(DOMAINS), entity=True).cuda()
            state = net.load_state_dict(
                load_model(mode, "cuda", folder, receipts).state_dict(), strict=False
            )
            if state.missing_keys != ["entity.weight"] or state.unexpected_keys:
                raise ValueError(state)
            feats = torch.as_tensor(observations(bank, mode), device="cuda")
            opt = torch.optim.AdamW(
                net.parameters(), lr=0.0008, weight_decay=0.01, fused=True
            )
            dg = hashlib.sha256()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            e = snapshot()
            start = monotonic()
            for t in range(1600):
                ix = np.concatenate(
                    (
                        rng.integers(0, len(rows), 24),
                        rng.integers(len(rows), len(combined), 24),
                    )
                )
                dg.update(ix.tobytes())
                length = lens[ix]
                m = int(length.max())
                pos = loc[ix, :m]
                if arm == "entity-erased":
                    pos = torch.zeros_like(pos)
                o = net(feats[x[ix, :m].long()], length, domains[ix], pos)
                loss = interval_loss(o, length, y[ix].long())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if t == 1000:
                    opt.param_groups[0]["lr"] = 0.0002
                if t % 200 == 0:
                    print(arm, t, float(loss), flush=True)
                    if (
                        not torch.isfinite(loss)
                        or monotonic() - start > 480
                        or torch.cuda.max_memory_allocated() > 2 * 2**30
                    ):
                        raise RuntimeError("Fit exceeded resource limit")
            torch.cuda.synchronize()
            elapsed = monotonic() - start
            cost = consumed(e, snapshot())
            path = folder / (arm + ".safetensors")
            temp = path.with_suffix(".tmp")
            save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(temp))
            temp.replace(path)
            dump(
                receipts / (arm + "-fit.json"),
                dict(
                    weights=sha(path),
                    schedule=dg.hexdigest(),
                    parameters=sum(p.numel() for p in net.parameters()),
                    updates=1600,
                    seconds=elapsed,
                    energy=cost,
                    peak=torch.cuda.max_memory_allocated(),
                    distinct_pairs=len(examples),
                    pairs_by_field=Counter(k for a, b, k in examples),
                ),
            )
    if (
        len(
            {
                read(receipts / (a + "-fit.json"))["schedule"]
                for a in ("entity-ink", "entity-erased", "entity-byte")
            }
        )
        != 1
    ):
        raise ValueError("Unmatched sampling")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
