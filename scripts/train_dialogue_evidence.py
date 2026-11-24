"""Train source-grounded visual and byte students with a matched human control."""

import argparse
from pathlib import Path
from time import monotonic
import json
import numpy as np
import torch
from safetensors.torch import save_file
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.train_dialogue_ink import dialogue_loss
from scripts.train_dialogue_surface import representation, corpus, ROOT_MODEL
from scripts.qualify_dialogue_teacher import PUB, WORK, ROOT, dump, sha
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot
from scripts.train_dialogue_subword import timed

RELEASE = ROOT / "models/dialogue_evidence"
SEED = 420121


def evidence_mask(evidence, slots):
    # Flattened candidates are ordered by word position, then four physical boundary edits.
    pos = torch.arange(slots, device=evidence.device) // 4
    return (
        (pos[None] >= evidence[:, 0, None])
        & (pos[None] <= evidence[:, 1, None])
        & (evidence[:, 0, None] >= 0)
    )


def citation_loss(logits, matches, evidence):
    hits = matches & evidence_mask(evidence, logits.size(-1))[:, None]
    active = hits.any(-1)
    if not active.any():
        return logits.new_zeros(()), 0
    p = logits[active].log_softmax(-1)
    return -p.masked_fill(~hits[active], -torch.inf).logsumexp(-1).mean(), int(
        active.sum()
    )


def fitting_data(data):
    q = PUB / "current_last/qualification.json"
    if not q.exists():
        raise ValueError("Finish the second teacher qualification first")
    passed = json.loads(q.read_text())["admitted"]
    if not passed:
        return data, None
    receipt = json.loads((PUB / "current_last/labels.json").read_text())
    f = WORK / "current_last/accepted.json"
    if sha(f) != receipt["accepted_sha"]:
        raise ValueError("Changed teacher labels")
    labels = json.loads(f.read_text())
    if not labels:
        raise ValueError("No training labels qualified")
    rows = [r for r in data["rows"] if r[0] == "train"]
    extras = []
    ids = []
    for key, target in sorted(labels.items(), key=lambda p: int(p[0])):
        i = int(key)
        r = list(rows[i])
        r[5] = target["answer"]
        r[6] = target["evidence"]
        extras.append(r)
        ids.append(i)
    return dict(pages=data["pages"], rows=rows + extras), np.array(ids)


def fit(mode, source, data, targets, root, seed=SEED):
    dest = root / mode / source
    if dest.exists():
        raise FileExistsError(dest)
    parent = ROOT_MODEL / mode
    net = load(parent, "cuda").train()
    face, unit, _ = representation(mode)
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    train = corpus(data, "train", net, face, unit)
    palette, labels = train.glyph_targets()
    if not torch.equal(palette, net.palette.cpu()):
        raise ValueError("Teacher changed the writable alphabet")
    prep = timed(start, meter)
    original = len(train.rows) if targets is None else len(train.rows) - len(targets)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0002)
    rng = np.random.default_rng(seed)
    trace = []
    active = 0
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 1501):
        if targets is None:
            ix = rng.integers(original, size=32)
        else:
            general = rng.integers(original, size=16)
            choice = rng.integers(len(targets), size=16)
            augmented = original + choice if source == "teacher" else targets[choice]
            ix = np.concatenate((general, augmented))
        loss, _, d, _, _ = dialogue_loss(net, train, ix, labels)
        cite, count = citation_loss(d["ptr"], d["matches"], train.ev[ix])
        if source != "plain":
            loss = loss + 0.35 * cite
        active += count
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        if step % 250 == 0:
            trace.append([step, float(loss.detach()), float(cite.detach())])
            print(mode, source, trace[-1], round(monotonic() - start, 1), flush=True)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Exceeded fit budget")
    cost = timed(start, meter)
    dest.mkdir(parents=True)
    f = dest / "writer.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(f)
    )
    dump(
        dest / "weights.json",
        dict(
            sha256=sha(f),
            parent=sha(parent / "writer.safetensors"),
            mode=mode,
            labels=source,
            teacher_admitted=targets is not None,
            accepted_cases=0 if targets is None else len(targets),
            seed=seed,
            steps=1500,
            presentations=48000,
            parameters=sum(p.numel() for p in net.parameters()),
            fit=cost,
            preparation=prep,
            evidence_positions=active,
            max_passage_words=448,
            trace=trace,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=RELEASE)
    args = p.parse_args()
    with gpu_owner():
        data, targets = fitting_data(read_data())
        for mode in ("ink", "byte"):
            for arm in (
                ("human", "teacher") if targets is not None else ("plain", "human")
            ):
                fit(mode, arm, data, targets, args.output)
