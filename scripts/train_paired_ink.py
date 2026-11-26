"""Learn complete pictured translations with a binary assignment loss on human siblings."""

from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_paired_ink import (
    BASE,
    FIRST,
    MODELS,
    PUB,
    ROOT,
    WORK,
    read,
    dump,
    sha,
)
from scripts.train_fact_ranking import atomic_save
from scripts.train_ordered_translation import pack
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore


def material(device):
    features = torch.tensor(
        observations({"tiles": np.load(BASE / "palette.npy")}, "ink"), device=device
    )
    inventory = [
        torch.tensor(z, device=device) for z in read(BASE / "inventories.json")
    ]
    local = [torch.full((len(features),), -100, device=device) for _ in inventory]
    for mapping, values in zip(local, inventory, strict=True):
        mapping[values] = torch.arange(len(values), device=device)
    return features, inventory, local


def conditional(net, features, source, target, vocabulary, mapping, t):
    scores = net(features, source, target, vocabulary, t)
    labels = mapping[target]
    nll = F.cross_entropy(
        scores.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    return -nll.sum(dim=1)


def scores_for_pairs(net, features, inventory, local, pairs, t, device):
    source = [z[1 - t] for pair in pairs for z in pair]
    target = [z[t] + [1] for pair in pairs for z in pair]
    y = pack(target, device)
    # Flatten in family order. The second half exchanges the targets within each pair.
    other = y.reshape(-1, 2, y.shape[1]).flip(1).reshape_as(y)
    x = pack(source, device).repeat(2, 1)
    logp = conditional(
        net, features, x, torch.cat([y, other]), inventory[t], local[t], t
    )
    right, crossed = [a.reshape(-1, 2) for a in logp.chunk(2)]
    return right, crossed, y.count_nonzero()


def main():
    configure_cuda_cli()
    split = read(PUB / "split.json")
    for tag, v in split["files"].items():
        if sha(WORK / (tag + ".json")) != v:
            raise ValueError(tag)
    replayfile = FIRST / "replay.json"
    if (
        sha(replayfile)
        != read(ROOT / "references/evidence/proposition_translation_v1/design.json")[
            "files"
        ]["replay"]
    ):
        raise ValueError("Replay")
    inputs = {
        p: sha(ROOT / p)
        for p in (
            "scripts/train_paired_ink.py",
            "scripts/prepare_paired_ink.py",
            "references/paired_ink_protocol.md",
            "references/evidence/paired_ink_v1/split.json",
            "models/translation_paraphrases/varied/weights.safetensors",
            "ilm/visual_lm/ordered_translation.py",
        )
    }
    pairs = [r["cells"] for r in read(WORK / "train.json")]
    replay = read(replayfile)
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
            raise RuntimeError("Insufficient free GPU memory for this fit")
        features, inventory, local = material("cuda")
        for mode in ("likelihood", "assignment"):
            target = MODELS / mode
            target.mkdir(parents=True, exist_ok=True)
            weights = target / "weights.safetensors"
            info = target / "fit.json"
            if info.exists():
                if (
                    read(info)["binding"] != inputs
                    or sha(weights) != read(info)["weights"]
                ):
                    raise ValueError("Saved fit differs")
                continue
            torch.manual_seed(92837)
            net = restore("paraphrase-varied", "cuda").train()
            opt = torch.optim.AdamW(net.parameters(), lr=0.00022, fused=True)
            ck = WORK / (mode + "-resume.pt")
            history = []
            if ck.exists():
                cp = torch.load(ck, map_location="cpu", weights_only=True)
                if cp["binding"] != inputs:
                    raise ValueError("Incompatible resume")
                net.load_state_dict(cp["model"])
                opt.load_state_dict(cp["optimizer"])
                history = cp["history"]
            for epoch in range(len(history), 4):
                start = monotonic()
                torch.cuda.reset_peak_memory_stats()
                rng = np.random.default_rng(95731 + epoch)
                batches = []
                for t in (0, 1):
                    order = rng.permutation(len(pairs))
                    batches.extend(
                        (t, order[i : i + 24]) for i in range(0, len(order), 24)
                    )
                stats = dict(
                    positive_glyphs=0,
                    positive_requests=0,
                    families_scored=0,
                    assignment_nats=0.0,
                    translation_nats=0.0,
                )
                for b in rng.permutation(len(batches)):
                    t, ids = batches[b]
                    rs = [
                        pool[k]
                        for pool in replay
                        for k in rng.choice(len(pool), len(ids), replace=False)
                    ]
                    x = pack([r[1 - t] for r in rs], "cuda")
                    y = pack([r[t] + [1] for r in rs], "cuda")
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        good, foil, count = scores_for_pairs(
                            net,
                            features,
                            inventory,
                            local,
                            [pairs[i] for i in ids],
                            t,
                            "cuda",
                        )
                        extra = conditional(
                            net, features, x, y, inventory[t], local[t], t
                        )
                        total_nll = -good.sum() - extra.sum()
                        nt = count + y.count_nonzero()
                        binary = F.softplus(-(good - foil).sum(1)).mean()
                        loss = total_nll / nt
                        if mode == "assignment":
                            loss = loss + 0.04 * binary
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), 1.0, error_if_nonfinite=True
                    )
                    opt.step()
                    stats["positive_glyphs"] += int(nt)
                    stats["positive_requests"] += 4 * len(ids)
                    stats["families_scored"] += len(ids)
                    stats["assignment_nats"] += float(binary.detach()) * len(ids)
                    stats["translation_nats"] += float(total_nll.detach())
                history.append(
                    dict(
                        epoch=epoch + 1,
                        seconds=monotonic() - start,
                        peak_bytes=torch.cuda.max_memory_allocated(),
                        **stats,
                    )
                )
                atomic_save(
                    lambda p: torch.save(
                        dict(
                            binding=inputs,
                            model=net.state_dict(),
                            optimizer=opt.state_dict(),
                            history=history,
                        ),
                        p,
                    ),
                    ck,
                )
                print(mode, history[-1], flush=True)
            atomic_save(
                lambda p: save_file(
                    {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
                ),
                weights,
            )
            dump(
                info,
                dict(
                    binding=inputs,
                    parameters=sum(p.numel() for p in net.parameters()),
                    epochs=history,
                    weights=sha(weights),
                ),
            )


if __name__ == "__main__":
    main()
