"""Fit contextual image retrieval and its matched QA-only control."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from ilm.visual_lm.context_cloze import ContextCloze
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.measure_writer_energy import snapshot
from scripts.prepare_compose_dialogue import prepared, PUB as TRAIN_SOURCE
from scripts.prepare_context_cloze import MODEL, PUB
from scripts.prepare_dialogue_ink import sha, dump
from scripts.prepare_word_memory import MODEL as PARENT
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_dialogue_subword import timed
from scripts.train_span_choice import gpu_owner

STEPS = 4000
SEED = 936413


def inventory(net, ds):
    which = net.lookup(ds.bank)
    eligible = []
    for start in range(0, len(ds.rows), 96):
        ix = torch.arange(start, min(start + 96, len(ds.rows)), device=ds.device)
        ids = torch.cat((ds.page[ds.pindex[ix]], ds.q[ix], ds.hist[ix]), 1).long()
        source = ds.bank.edits[ids].flatten(1)
        target = ds.ans[ix].long()
        stored = which[target] >= 0
        absent = ~(target[:, :, None] == source[:, None]).any(-1)
        b, t = (stored & absent & (ds.al[ix, None] > 1)).nonzero(as_tuple=True)
        eligible.append(torch.stack((ix[b], t), 1))
    slots = torch.cat(eligible)
    tags = which[ds.ans[slots[:, 0], slots[:, 1]].long()]
    counts = torch.bincount(tags, minlength=len(net.prototype_words)).tolist()
    centres = (which[ds.page.long()] >= 0).nonzero()
    return slots.cpu().numpy(), counts, centres


def page_state(net, ds, centres):
    page, t = centres.unbind(1)
    idx = t[:, None] + t.new_tensor([-2, -1, 1, 2])
    inside = (idx >= 0) & (idx < ds.pl[page, None])
    ids = ds.page[page[:, None], idx.clamp(0, ds.page.size(1) - 1)].long()
    ink = ds.bank.features[ids, :50] * inside[:, :, None]
    local = torch.cat((ink * 6, inside[:, :, None], (~inside)[:, :, None]), -1).flatten(
        1
    )
    return torch.cat((local.new_zeros(len(page), net.width), local), -1)


def train_arm(data, mode, arm, output):
    dest = output / mode / arm
    parent = PARENT / mode
    binding = dict(
        mode=mode,
        arm="context_cloze",
        curriculum=arm,
        seed=SEED,
        steps=STEPS,
        parent=sha(parent / "writer.safetensors"),
        reservation=sha(PUB / "reservation.json"),
        training_split=sha(TRAIN_SOURCE / "split.json"),
    )
    if dest.exists():
        d = json.loads((dest / "weights.json").read_text())
        if any(d[k] != v for k, v in binding.items()) or d["sha256"] != sha(
            dest / "writer.safetensors"
        ):
            raise ValueError("Existing fit mismatch")
        return d["qa_schedule"]
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    clock, meter = monotonic(), snapshot()
    base = load(parent, "cuda")
    net = (
        ContextCloze(
            base.palette,
            mode,
            base.prototype_cells,
            base.prototype_lengths,
            base.prototype_features,
        )
        .cuda()
        .eval()
    )
    missing, extra = net.load_state_dict(base.state_dict(), strict=False)
    if extra or any(not k.startswith(("local_gate.", "local_query.")) for k in missing):
        raise ValueError("Unexpected parent tensors")
    for k, p in net.named_parameters():
        p.requires_grad_(
            k.startswith(("local_gate.", "local_query.", "retrieve."))
            or k in ("keys", "prior")
        )
    params = [p for p in net.parameters() if p.requires_grad]
    face, unit, _ = representation(mode)
    ds = corpus(data, "train", net, face, unit)
    slots, counts, centres = inventory(net, ds)
    which = net.lookup(ds.bank)
    prep = timed(clock, meter)
    rng = np.random.default_rng(SEED)
    windows = np.random.default_rng(SEED + 1)
    digest = hashlib.sha256()
    page_digest = hashlib.sha256()
    seen_windows = set()
    seen = set()
    opt = torch.optim.AdamW(params, lr=0.001)
    trace = []
    clock, meter = monotonic(), snapshot()
    torch.cuda.reset_peak_memory_stats()
    lengths = ds.al.tolist()
    b = torch.arange(32, device=ds.device)
    for step in range(1, STEPS + 1):
        typical = rng.integers(len(ds.rows), size=16)
        targeted = slots[rng.integers(len(slots), size=16)]
        ix = np.concatenate((typical, targeted[:, 0]))
        t = np.concatenate(
            (rng.integers(np.asarray([lengths[i] for i in typical])), targeted[:, 1])
        )
        hiding = rng.random(32)
        # Full, partially visible and single-concealed target configurations.
        visibility = rng.random((32, net.max_words))
        digest.update(
            ix.astype("<i8").tobytes()
            + t.astype("<i8").tobytes()
            + hiding.tobytes()
            + visibility.tobytes()
        )
        seen.update(ix.tolist())
        t = torch.as_tensor(t, device=ds.device)
        with torch.no_grad():
            x, v = ds.inputs(ix)
            mem, _ = net.memory(*x)
            n = ds.al[ix]
            w = int(n.max())
            a = ds.ans[ix, :w].long()
            shown = torch.as_tensor(
                visibility[:, :w]
                < np.where(
                    hiding[:, None] < 0.34, 0, np.where(hiding[:, None] < 0.67, 0.5, 1)
                ),
                device=ds.device,
            )
            shown &= torch.arange(w, device=ds.device)[None] < n[:, None]
            shown[b, t] = False
            _, _, z = net.paint(mem, ds.bank.features[a, :50], shown, n)
            state = z[b, t]
            target = a[b, t]
            labels = which[target]
            absent = (labels >= 0) & ~(v.flatten(1) == target[:, None]).any(1)
        if not absent[16:].all():
            raise ValueError("Target sampler admitted an available word")
        classification = F.cross_entropy(
            net.retrieve_logits(state[absent]), labels[absent]
        )
        gate = F.binary_cross_entropy_with_logits(
            net.gate_logits(state[:16])[:, 0], absent[:16].float()
        )
        loss = classification + gate
        if arm == "passage":
            pick = windows.integers(len(centres), size=64)
            seen_windows.update(pick.tolist())
            page_digest.update(pick.astype("<i8").tobytes())
            coords = centres[pick]
            zpage = page_state(net, ds, coords)
            truth = which[ds.page[coords[:, 0], coords[:, 1]].long()]
            loss = loss + F.cross_entropy(net.retrieve_logits(zpage), truth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3, error_if_nonfinite=True)
        opt.step()
        if step % 500 == 0:
            trace.append([step, float(loss.detach())])
            print(mode, arm, trace[-1], flush=True)
            if (
                monotonic() - clock > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Fit exceeds resource ceiling")
    cost = timed(clock, meter)
    trained = {k for k, p in net.named_parameters() if p.requires_grad}
    if any(
        not torch.equal(net.state_dict()[k], value)
        for k, value in base.state_dict().items()
        if k not in trained
    ):
        raise ValueError("A frozen tensor changed")
    dest.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        path = Path(tmp) / "complete"
        path.mkdir()
        f = path / "writer.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
            str(f),
        )
        dump(
            path / "weights.json",
            binding
            | dict(
                sha256=sha(f),
                qa_schedule=digest.hexdigest(),
                distinct_qa=len(seen),
                qa_positions=STEPS * 32,
                eligible_absent_slots=len(slots),
                absent_counts=counts,
                eligible_page_centres=len(centres),
                distinct_page_centres=len(seen_windows),
                page_presentations=STEPS * 64 if arm == "passage" else 0,
                page_schedule=page_digest.hexdigest(),
                max_passage_words=448,
                bytes_on_disk=f.stat().st_size,
                parameters=sum(p.numel() for p in net.parameters()),
                trainable=sum(p.numel() for p in params),
                trace=trace,
                preparation=prep,
                fit=cost,
            ),
        )
        path.rename(dest)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL)
    args = parser.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        reg = read_data()
        extra = prepared()["extra"]
        data = dict(
            rows=reg["rows"] + extra["rows"], pages=reg["pages"] | extra["pages"]
        )
        schedules = [
            train_arm(data, m, a, args.output)
            for m in ("ink", "byte")
            for a in ("qa", "passage")
        ]
        if len(set(schedules)) != 1:
            raise ValueError("QA schedules differ")
