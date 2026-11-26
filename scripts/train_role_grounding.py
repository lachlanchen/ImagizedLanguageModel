"""Train the registered matched role-location intervention on bilingual book images."""

from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.role_attention import source_role_mass, window_roles
from scripts.prepare_library_roles import PUB, RUN
from scripts.prepare_teacher_translation import dump
from scripts.sample_visual_library import RECEIPT, ROOT, library, read, sha
from scripts.train_paired_ink import conditional
from scripts.train_teacher_translation import batches, replay
from scripts.train_visual_prefix import atomic_save, development, gpu_owner, material, pack
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore

MODELS = ROOT / "models/library_roles"
ARMS = ("likelihood", "shuffled", "grounded")


def shuffled_tags(source):
    """Permute each sentence independently in a fixed per-direction row order."""
    result = source.copy()
    rng = np.random.default_rng(608513)
    for row in result:
        for side in row:
            indices = np.flatnonzero(side)
            side[indices] = rng.permutation(side[indices])
    if not np.array_equal(np.sort(source, axis=-1), np.sort(result, axis=-1)):
        raise ValueError("Changed a within-sentence role count")
    if not np.array_equal(source != 0, result != 0):
        raise ValueError("Changed tagged locations")
    return result


def role_cost(mass, eligible, target_roles):
    matched = mass.gather(-1, target_roles.long().unsqueeze(-1)).squeeze(-1)
    safe = torch.where(eligible, matched, 1.0)
    return -safe.clamp_min(torch.finfo(safe.dtype).tiny).log().sum()


def book_rows():
    pairs = library("train")
    rows = [dict(input=r["cells"][1-t], target=r["cells"][t], t=t)
            for t in (0, 1) for r in pairs]
    return rows + replay(), len(pairs)


def role_grids(roles, source_roles, rows, pairs, x, y):
    src = np.zeros(tuple(x.shape), dtype=np.uint8)
    tgt = np.zeros(tuple(y.shape), dtype=np.uint8)
    for t in (0, 1):
        where = slice(t*pairs, (t+1)*pairs)
        src[where, :96] = source_roles[:, 1-t]
        tgt[where, :96] = roles[:, t]
    mask = np.zeros_like(tgt, dtype=bool)
    for role in (1, 2, 3):
        mask |= (tgt == role) & (src == role).any(1)[:, None]
    return (torch.as_tensor(src, device=x.device), torch.as_tensor(tgt, device=y.device),
            torch.as_tensor(mask, device=y.device), int(mask.sum()))


def fit(arm, rows, pair_count, plans, roles, permuted, optical, dev, binding):
    dest = MODELS / arm
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / "fit.json"
    weights = dest / "weights.safetensors"
    if receipt.exists():
        saved = read(receipt)
        if saved["binding"] != binding or sha(weights) != saved["weights"]:
            raise ValueError("Completed fit changed")
        return saved
    x = pack([r["input"] for r in rows], "cuda")
    y = pack([r["target"] + [1] for r in rows], "cuda")
    lengths = np.array([(len(r["input"]), len(r["target"])+1) for r in rows])
    src_role, tgt_role, eligible, count = role_grids(
        roles, permuted if arm == "shuffled" else roles, rows, pair_count, x, y)
    registered = read(PUB / "alignment.json")["splits"]["train"]["eligible_target"]
    if count != sum(v["positions"] for v in registered.values()):
        raise ValueError("Training annotations do not reproduce the expected exposure")
    f, inv, maps = optical
    torch.manual_seed(34105)
    net = restore("library-broad", "cuda")
    optim = torch.optim.AdamW(net.parameters(), lr=.00016, fused=True)
    history = []
    checkpoint = RUN / f"{arm}-resume.pt"
    if checkpoint.exists():
        cp = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if cp["binding"] != binding:
            raise ValueError("Saved training state belongs to another experiment")
        net.load_state_dict(cp["model"])
        optim.load_state_dict(cp["optimizer"])
        history = cp["history"]
        del cp
    for epoch in range(len(history), 4):
        start = monotonic()
        net.train()
        torch.cuda.reset_peak_memory_stats()
        optim.param_groups[0]["lr"] = .00016 - epoch*.00004
        totals = torch.zeros(3, device="cuda", dtype=torch.float64)
        positions = 0
        for t, ids in plans[epoch]:
            ix = torch.as_tensor(ids, device="cuda")
            nx, ny = lengths[ids].max(axis=0)
            n = int(lengths[ids, 1].sum())
            source, target = x[ix, :nx], y[ix, :ny]
            mask = eligible[ix, :ny]
            costs = []
            handle = None
            if arm != "likelihood":
                fractions = window_roles(source, src_role[ix, :nx], net.stride)
                labels = tgt_role[ix, :ny]

                def observe(module, args):
                    query, (keys, _), valid = args
                    with torch.autocast("cuda", enabled=False):
                        mass = source_role_mass(module, query.float(), keys, valid, fractions)
                        costs.append(role_cost(mass, mask, labels))

                handle = net.dec[-1].cross.register_forward_pre_hook(observe)
            optim.zero_grad(set_to_none=True)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    nll = -conditional(net, f, source, target, inv[t], maps[t], t).sum()
            finally:
                if handle is not None:
                    handle.remove()
            cost, = costs if arm != "likelihood" else (nll.new_zeros(()),)
            objective = (nll + .1 * cost) / n
            objective.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optim.step()
            totals += torch.stack((nll.detach(), cost.detach(), mask.sum()))
            positions += n
        scores = development(net, dev, optical)
        glyph_loss, role_loss, tags_seen = totals.tolist()
        entry = dict(epoch=epoch+1, positions=positions, eligible=int(tags_seen),
                     role_supervised=int(tags_seen) if arm != "likelihood" else 0,
                     updates=len(plans[epoch]), stops=len(rows), nll=glyph_loss/positions,
                     role_nll=role_loss/tags_seen if arm != "likelihood" else None,
                     seconds=monotonic()-start, peak_bytes=torch.cuda.max_memory_allocated(),
                     development=scores)
        history.append(entry)
        if positions != lengths[:, 1].sum() or tags_seen != count:
            raise ValueError("Incomplete exposure")
        if entry["peak_bytes"] > 12*2**30 or sum(h["seconds"] for h in history) > 3600:
            raise RuntimeError("Registered resource ceiling exceeded")
        atomic_save(lambda p: torch.save(dict(model=net.state_dict(), optimizer=optim.state_dict(),
                                              history=history, binding=binding), p), checkpoint)
        print(arm, entry, flush=True)
    atomic_save(lambda p: save_file({k: v.detach().cpu() for k,v in net.state_dict().items()}, p), weights)
    result = dict(arm=arm, weights=sha(weights), history=history, binding=binding)
    dump(receipt, result)
    return result


def main():
    configure_cuda_cli()
    rows, pairs = book_rows()
    roles_path = RUN / "train.npy"
    if sha(roles_path) != read(PUB / "alignment.json")["files"][roles_path.name]:
        raise ValueError("Changed training roles")
    roles = np.load(roles_path)
    permuted = shuffled_tags(roles)
    plans = [batches(rows, epoch) for epoch in range(4)]
    dev = [dict(r, domain=r["work"]) for r in library("dev")]
    paths = [Path(__file__), RECEIPT, PUB / "alignment.json", PUB / "unseen-books.json",
             ROOT / "references/library_role_grounding_protocol.md",
             ROOT / "models/library_diversity/broad/weights.safetensors",
             ROOT / "ilm/visual_lm/role_attention.py", ROOT / "ilm/visual_lm/ordered_translation.py"]
    binding = {str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}
    digest = sha256()
    for row in rows:
        for side in ("input", "target"):
            seq = row[side]
            digest.update(np.asarray([row["t"], len(seq), *seq], dtype="<u4").tobytes())
    binding.update(ordered_training_ink=digest.hexdigest(), shuffled_labels=sha256(permuted.tobytes()).hexdigest())
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 7*2**30:
            raise RuntimeError("Insufficient free GPU memory")
        optical = material("cuda")
        for arm in ARMS:
            fit(arm, rows, pairs, plans, roles, permuted, optical, dev, binding)
    dump(PUB / "training.json", dict(binding=binding, changed_source_tags=int((roles!=permuted).sum()),
         total_tagged_source=int((roles!=0).sum()), fits={m:sha(MODELS/m/"fit.json") for m in ARMS}))


if __name__ == "__main__":
    main()
