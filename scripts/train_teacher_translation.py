"""Train two image writers on identical sources, matching full target exposure per domain."""

from collections import defaultdict
from time import monotonic
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import torch
from safetensors.torch import save_file
from scripts.prepare_teacher_translation import (
    ROOT,
    BOOK,
    PREVIOUS,
    WORK,
    PUB,
    MODEL,
    read,
    dump,
    sha,
    training_rows,
)
from scripts.train_visual_prefix import (
    development,
    pack,
    gpu_owner,
    atomic_save,
    material,
)
from scripts.train_paired_ink import conditional
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli

ARMS = ("corpus", "distilled")


def balanced_indices(deltas):
    """Largest subset whose complete paired lengths sum to the same total."""
    deltas = np.asarray(deltas, dtype=np.int64)
    values, counts = np.unique(deltas, return_counts=True)
    if not len(values):
        return []
    sol = milp(
        -np.ones(len(values)),
        integrality=np.ones(len(values)),
        bounds=Bounds(0, counts),
        constraints=LinearConstraint(values[None], 0, 0),
        options={"mip_rel_gap": 0.0},
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    capacities = np.rint(sol.x).astype(int)
    if (
        np.any(capacities < 0)
        or np.any(capacities > counts)
        or values @ capacities
        or abs(sol.fun + capacities.sum()) > 1e-5
    ):
        raise ValueError("Integer length balance failed")
    left = dict(zip(values, capacities, strict=True))
    chosen = []
    for j, v in enumerate(deltas):
        if left[v]:
            chosen.append(j)
            left[v] -= 1
    return chosen


def examples():
    rows = read(WORK / "screened.json")
    if sha(WORK / "screened.json") != read(PUB / "teacher.json")["screened"]:
        raise ValueError("Teacher outputs changed")
    groups = defaultdict(list)
    for r in rows:
        if r["blind"] and r["informed"]:
            groups[r["domain"], r["t"]].append(r)
    selected = []
    counts = {}
    for (domain, t), rr in sorted(groups.items()):
        keep = [
            rr[j]
            for j in balanced_indices(
                [len(r["teacher_cells"]) - len(r["target"]) for r in rr]
            )
        ]
        if len(keep) < 64:
            raise ValueError(("Insufficient matched pairs", domain, t, len(keep)))
        target = sum(len(r["target"]) + 1 for r in keep)
        assert target == sum(len(r["teacher_cells"]) + 1 for r in keep)
        selected += keep
        counts[domain + str(t)] = dict(
            eligible=len(rr), selected=len(keep), scored=target
        )
    if len(counts) != 4:
        raise ValueError("Incomplete strata")
    rng = np.random.default_rng(91173)
    rng.shuffle(selected)
    return selected, counts


def replay():
    bank = training_rows()
    roles = np.load(PREVIOUS / "replacement.npz")["indices"]
    skip = {r["id"] for r in read(WORK / "sources.json")}
    skip.update(
        r["training_index"]
        for r in read(
            ROOT / "references/evidence/wide_translation_v1/teacher-targets.json"
        )["records"]
    )
    skip.update(
        r["index"]
        for r in read(
            ROOT / "references/evidence/prefix_recovery_v1/alignment-meaning.json"
        )["records"]
        if r["group"] in ("book", "talk")
    )
    rng = np.random.default_rng(116332)
    natural = np.setdiff1d(
        np.arange(len(bank)), np.asarray(sorted(skip | set(map(int, roles))))
    )
    chosen = np.concatenate(
        [
            rng.choice(natural, 4096, replace=False),
            rng.choice(roles, 768, replace=False),
        ]
    )
    result = []
    for i in chosen:
        pair = [side[: np.count_nonzero(side)].tolist() for side in bank[i]]
        for t in (0, 1):
            result.append(dict(input=pair[1 - t], target=pair[t], t=t))
    return result


def batches(rr, epoch):
    rng = np.random.default_rng(77129 + epoch)
    batches = []
    for t in (0, 1):
        ix = rng.permutation([i for i, r in enumerate(rr) if r["t"] == t])
        # Length bucket is determined solely by the input and is identical in both arms.
        b = np.array([len(rr[i]["input"]) // 12 for i in ix])
        ix = ix[np.argsort(b, kind="stable")]
        batches.extend((t, ix[j : j + 96]) for j in range(0, len(ix), 96))
    return [batches[i] for i in rng.permutation(len(batches))]


def fit(arm, selection, rehearsal, binding):
    folder = MODEL / arm
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "weights.safetensors"
    finished = folder / "fit.json"
    if finished.exists():
        receipt = read(finished)
        if receipt["binding"] != binding or sha(path) != receipt["weights"]:
            raise ValueError("Fit changed")
        return receipt
    # Three occurrences per selected source, with identical replay in both arms.
    rr = selection * 3 + rehearsal
    targets = [
        r.get("teacher_cells", r["target"]) if arm == "distilled" else r["target"]
        for r in rr
    ]
    expected = sum(len(x) + 1 for x in targets)
    gpu = "cuda"
    f, inv, local = optical = material(gpu)
    torch.manual_seed(99241)
    net = restore("prefix-recovery", gpu)
    opt = torch.optim.AdamW(net.parameters(), lr=0.00016, fused=True)
    dev = read(BOOK / "dev.json")
    initial = development(net, dev, optical)
    history = []
    best_score = -1
    ck = WORK / (arm + "-resume.pt")
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        if state["binding"] != binding or state["arm"] != arm:
            raise ValueError("Resume mismatch")
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        history, best, best_score, selected_pass = (
            state[k] for k in ("history", "best", "score", "selected")
        )
        del state
    src = pack([r["input"] for r in rr], gpu)
    tgt = pack([s + [1] for s in targets], gpu)
    ls = np.array([len(r["input"]) for r in rr])
    lt = np.array([len(y) + 1 for y in targets])
    for ep in range(len(history), 12):
        start = monotonic()
        torch.cuda.reset_peak_memory_stats()
        net.train()
        opt.param_groups[0]["lr"] = 0.00016 - (0.00012 / 11) * ep
        total = 0.0
        tally = 0
        order = batches(rr, ep)
        for t, ix in order:
            ii = torch.as_tensor(ix, device=gpu)
            x = src[ii, : ls[ix].max()]
            y = tgt[ii, : lt[ix].max()]
            n = int(lt[ix].sum())
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = -conditional(net, f, x, y, inv[t], local[t], t).sum() / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * n
            tally += n
        scores = development(net, dev, optical)
        score = float(np.mean(list(scores.values())))
        if score > best_score:
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            selected_pass, best_score = ep + 1, score
        history.append(
            dict(
                pass_number=ep + 1,
                updates=len(order),
                supervised_glyphs_and_stops=tally,
                stops=len(rr),
                nll=total / tally,
                development=scores,
                peak_bytes=torch.cuda.max_memory_allocated(),
                seconds=monotonic() - start,
            )
        )
        if (
            tally != expected
            or history[-1]["peak_bytes"] > 12 * 2**30
            or sum(h["seconds"] for h in history) > 3600
        ):
            raise RuntimeError("Fit accounting or resource error")
        atomic_save(
            lambda p: torch.save(
                dict(
                    binding=binding,
                    arm=arm,
                    net=net.state_dict(),
                    opt=opt.state_dict(),
                    history=history,
                    best=best,
                    selected=selected_pass,
                    score=best_score,
                ),
                p,
            ),
            ck,
        )
        print(arm, history[-1], flush=True)
    atomic_save(lambda p: save_file(best, p), path)
    receipt = dict(
        binding=binding,
        weights=sha(path),
        selected_pass=selected_pass,
        initial_development=initial,
        history=history,
    )
    dump(finished, receipt)
    return receipt


def main():
    configure_cuda_cli()
    selection, counts = examples()
    helpers = [
        "scripts/train_teacher_translation.py",
        "references/complete_teacher_translation_protocol.md",
        "references/evidence/teacher_translation_v1/design.json",
        "references/evidence/teacher_translation_v1/teacher.json",
        "models/visual_prefix/recovery/fit.json",
        "models/visual_prefix/recovery/weights.safetensors",
    ]
    binding = {s: sha(ROOT / s) for s in helpers}
    dump(WORK / "selected.json", selection)
    binding[str((WORK / "selected.json").relative_to(ROOT))] = sha(
        WORK / "selected.json"
    )
    dump(
        PUB / "selection.json",
        dict(
            groups=counts,
            selected=len(selection),
            distinct_surfaces=sum(r["target"] != r["teacher_cells"] for r in selection),
            sha256=sha(WORK / "selected.json"),
        ),
    )
    rehearsal = replay()
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 7 * 2**30:
            raise RuntimeError("Insufficient GPU memory")
        arms = [fit(arm, selection, rehearsal, binding) for arm in ARMS]
    for a, b in zip(arms[0]["history"], arms[1]["history"], strict=True):
        for k in ("updates", "supervised_glyphs_and_stops", "stops"):
            assert a[k] == b[k]


if __name__ == "__main__":
    main()
