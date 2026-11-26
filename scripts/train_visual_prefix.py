"""Matched language fits with aligned human suffixes after predicted or reference ink."""

from collections import defaultdict
from time import monotonic
import numpy as np
from rapidfuzz.distance import Levenshtein
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from scripts.prepare_visual_rollins import (
    ROOT,
    BASE,
    BOOK,
    PREVIOUS,
    WORK,
    PUB,
    read,
    dump,
    sha,
    training_rows,
)
from scripts.train_book_breadth import schedule
from scripts.train_fact_ranking import atomic_save
from scripts.train_ordered_translation import pack
from scripts.train_paired_ink import material
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore, stopped

MODEL = ROOT / "models/visual_prefix"
ARMS = ("reference", "recovery")


def feedback(gold, cut, draft, consumed, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    bridge = draft if arm == "recovery" else gold[cut : cut + consumed]
    suffix = [*gold[cut + consumed :], 1]
    return [*gold[:cut], *bridge, *suffix], [*gold[:cut], *[0] * len(bridge), *suffix]


def streams(cells, z, arm):
    width = cells.shape[2] + 9
    words = np.zeros((len(cells), 2, width), np.uint16)
    labels = np.zeros_like(words)
    for i, (row, cuts, drafts, ends) in enumerate(
        zip(cells, z["cuts"], z["drafts"], z["ends"], strict=True)
    ):
        for t, side in enumerate(row):
            a, b = feedback(
                side[: np.count_nonzero(side)].tolist(),
                int(cuts[t]),
                drafts[t, : np.count_nonzero(drafts[t])].tolist(),
                int(ends[t]),
                arm,
            )
            words[i, t, : len(a)], labels[i, t, : len(b)] = a, b
    return words, labels


@torch.inference_mode()
def development(net, dev, optical):
    f, inv, _ = optical
    net.eval()
    groups = defaultdict(list)
    for t in (0, 1):
        for j in range(0, len(dev), 32):
            rows = dev[j : j + 32]
            seq = net.generate(
                f, pack([r["cells"][1 - t] for r in rows], "cuda"), inv[t], t
            ).cpu()
            for r, y in zip(rows, seq, strict=True):
                written, stop = stopped(y)
                groups[r["domain"]].append(
                    float(stop)
                    * Levenshtein.normalized_similarity(written, r["cells"][t])
                )
    return {k: float(np.mean(s)) for k, s in groups.items()}


def fit(arm, cells, z, binding, dev, optical):
    folder = MODEL / arm
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / "weights.safetensors"
    if (folder / "fit.json").exists():
        done = read(folder / "fit.json")
        if done["binding"] != binding or done["weights"] != sha(path):
            raise ValueError("Completed fit changed")
        return done
    feed, labels = streams(cells, z, arm)
    lengths = np.count_nonzero(cells, 2)
    feedback_lengths = np.count_nonzero(feed, 2)
    label_count = np.count_nonzero(labels, 2)
    if not np.array_equal(label_count, lengths + 1 - z["ends"]):
        raise ValueError("Supervised positions differ")
    gpu = lambda a: torch.from_numpy(a.astype(np.int32)).to("cuda")
    data, inputs, target = gpu(cells), gpu(feed), gpu(labels)
    del feed, labels
    f, inv, local = optical
    torch.manual_seed(90231)
    net = restore("book-rehearsal", "cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=0.00016, fused=True)
    clean = development(net, dev, optical)
    best_score = -1.0
    best_epoch = 0
    history = []
    ck = WORK / (arm + "-resume.pt")
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        if state["binding"] != binding or state["arm"] != arm:
            raise ValueError("Resume mismatch")
        net.load_state_dict(state["current"])
        opt.load_state_dict(state["opt"])
        best, best_epoch, best_score, history = (
            state[k] for k in ("best", "selected", "score", "history")
        )
        del state
    for ep in range(len(history), 3):
        net.train()
        start = monotonic()
        torch.cuda.reset_peak_memory_stats()
        opt.param_groups[0]["lr"] = (0.00016, 0.00010, 0.00006)[ep]
        total = 0.0
        tally = 0
        order = schedule(lengths, 8 + ep)
        for idx in order:
            m = feedback_lengths[idx].max(0)
            s = lengths[idx].max(0)
            ii = torch.as_tensor(idx, device="cuda")
            opt.zero_grad(set_to_none=True)
            count = int(label_count[idx].sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = []
                for t in (0, 1):
                    y = inputs[ii, t, : m[t]].long()
                    truth = local[t][target[ii, t, : m[t]].long()]
                    logits = net(f, data[ii, 1 - t, : s[1 - t]].long(), y, inv[t], t)
                    losses.append(
                        F.cross_entropy(
                            logits.float().transpose(1, 2),
                            truth,
                            ignore_index=-100,
                            reduction="sum",
                        )
                    )
                loss = sum(losses) / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * count
            tally += count
            if (
                torch.cuda.max_memory_allocated() > 12 * 2**30
                or monotonic() - start + sum(h["seconds"] for h in history) > 3600
            ):
                raise RuntimeError("Fit resource limit")
        assert tally == int(label_count.sum())
        metrics = development(net, dev, optical)
        score = float(np.mean(list(metrics.values())))
        if score > best_score:
            best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_epoch, best_score = ep + 1, score
        history.append(
            dict(
                epoch=ep + 1,
                updates=len(order),
                target_positions=tally,
                feedback_positions=int(feedback_lengths.sum()),
                nll=total / tally,
                development=metrics,
                seconds=monotonic() - start,
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        atomic_save(
            lambda p: torch.save(
                dict(
                    arm=arm,
                    binding=binding,
                    current=net.state_dict(),
                    opt=opt.state_dict(),
                    best=best,
                    selected=best_epoch,
                    score=best_score,
                    history=history,
                ),
                p,
            ),
            ck,
        )
        print(arm, history[-1], flush=True)
    atomic_save(lambda p: save_file(best, p), path)
    result = dict(
        binding=binding,
        weights=sha(path),
        initial_development=clean,
        selected_pass=best_epoch,
        epochs=history,
    )
    dump(folder / "fit.json", result)
    return result


def main():
    configure_cuda_cli()
    files = [
        PUB / "reservation.json",
        PUB / "drafts.json",
        ROOT / "references/visual_prefix_recovery_protocol.md",
        ROOT / "scripts/prepare_visual_rollins.py",
        ROOT / "scripts/train_visual_prefix.py",
        ROOT / "scripts/reserve_visual_prefix.py",
        ROOT / "models/book_breadth/rehearsal/weights.safetensors",
        ROOT / "ilm/visual_lm/ordered_translation.py",
        BASE / "palette.npy",
        BASE / "inventories.json",
    ]
    binding = {str(p.relative_to(ROOT)): sha(p) for p in files}
    if (
        sha(WORK / "test.json") != read(PUB / "reservation.json")["sha256"]
        or sha(WORK / "drafts.npz") != read(PUB / "drafts.json")["archive"]
    ):
        raise ValueError("Preparation changed")
    cells = training_rows()
    with gpu_owner(), np.load(WORK / "drafts.npz") as z:
        if torch.cuda.mem_get_info()[0] < 7 * 2**30:
            raise RuntimeError("Insufficient free memory")
        reports = [
            fit(arm, cells, z, binding, read(BOOK / "dev.json"), material("cuda"))
            for arm in ARMS
        ]
    for k in ("updates", "target_positions"):
        assert [r[k] for r in reports[0]["epochs"]] == [
            r[k] for r in reports[1]["epochs"]
        ]


if __name__ == "__main__":
    main()
