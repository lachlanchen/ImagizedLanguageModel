"""Fit question–predicate alignment from rendered training data on a single GPU."""

from collections import defaultdict
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.gloss_focus import cell_features
from ilm.visual_lm.pictured_action_lookup import PicturedActionLookup
from scripts.pictured_action_questions import TRAIN_FRAMES
from scripts.prepare_pictured_action_lookup import (
    PUBLIC,
    WORK,
    ROOT,
    SEED,
    read,
    dump,
    sha,
)
from scripts.prepare_pronoun_writer import window
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_asked_gloss import packed_batch
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore


def question_bank(strings, face, device):
    # Share identical physical cells across examples; these indices are used
    # only to assemble images. The model receives the 256 optical measurements.
    ink = sorted({face.cell(c) for s in strings for c in s} | {bytes(128)})
    lookup = {b: i for i, b in enumerate(ink)}
    positions = np.full(
        (len(strings), max(map(len, strings))), lookup[bytes(128)], np.int32
    )
    for i, s in enumerate(strings):
        positions[i, : len(s)] = [lookup[face.cell(c)] for c in s]
    raw = np.frombuffer(b"".join(ink), np.uint8).reshape(-1, 128).copy()
    return (
        cell_features(torch.from_numpy(raw).to(device)),
        torch.from_numpy(positions).to(device),
        torch.tensor(list(map(len, strings)), device=device),
    )


def matching_loss(score, group):
    positive = group[:, None] == group[None, :]
    masked = score.masked_fill(~positive, -torch.inf)
    return sum((score.logsumexp(d) - masked.logsumexp(d)).mean() for d in (0, 1)) / 2


def train():
    split = read(PUBLIC / "split.json")
    assert sha(WORK / "train.json") == split["train_sha256"]
    rows = read(WORK / "train.json")
    face = Face(FACES[0])
    writer = restore(device="cuda").requires_grad_(False)
    prep = monotonic()
    labels = sorted({r["gloss"][0][0] for r in rows})
    lookup = {a: i for i, a in enumerate(labels)}
    ids = np.array([lookup[r["gloss"][0][0]] for r in rows])
    heads = defaultdict(lambda: defaultdict(list))
    states = []
    with torch.no_grad():
        for a in range(0, len(rows), 128):
            batch = rows[a : a + 128]
            raw, _ = packed_batch(
                [window(r["page"], r["focus"][0]) for r in batch], face, "cuda"
            )
            states.append(writer.encode(cell_features(raw)))
    states = torch.cat(states)
    for i, r in enumerate(rows):
        heads[face.cell(r["page"][r["focus"][0]])][ids[i]].append(i)
    difficult = [list(senses.values()) for senses in heads.values() if len(senses) > 1]
    strings = [f.format(v) for v in labels for f in TRAIN_FRAMES]
    letters, bitmap, lengths = question_bank(strings, face, "cuda")
    dump(
        PUBLIC / "preparation.json",
        dict(
            seconds=monotonic() - prep,
            ambiguous_heads=len(difficult),
            questions=len(strings),
        ),
    )
    for arm in ("aligned", "exchanged"):
        folder = WORK / arm
        folder.mkdir(exist_ok=False)
        torch.manual_seed(SEED)
        net = PicturedActionLookup().cuda()
        optim = torch.optim.AdamW(net.parameters(), lr=0.00085)
        start = monotonic()
        energy = snapshot()
        torch.cuda.reset_peak_memory_stats()
        trace = []
        steps = 0
        for epoch in range(10):
            rng = np.random.default_rng(SEED + epoch)
            order = rng.permutation(len(rows))
            total = 0.0
            n = 0
            for pos in range(0, len(order), 96):
                indices = list(order[pos : pos + 96])
                if difficult:
                    for j in rng.integers(len(difficult), size=8):
                        group = difficult[j]
                        for k in rng.choice(len(group), size=2, replace=False):
                            indices.append(rng.choice(group[k]))
                qids = ids[indices].copy()
                frames = rng.integers(len(TRAIN_FRAMES), size=len(indices))
                shuffled = rng.permutation(len(indices))
                if arm == "exchanged":
                    qids = qids[shuffled]
                qi = qids * len(TRAIN_FRAMES) + frames
                width = int(lengths[qi].max())
                logits = (
                    net.scores(
                        states[indices], letters[bitmap[qi, :width].long()], lengths[qi]
                    )
                    / 0.055
                )
                loss = matching_loss(logits, torch.from_numpy(qids).cuda())
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total += float(loss.detach()) * len(indices)
                n += len(indices)
                steps += 1
                if (
                    monotonic() - start > 300
                    or torch.cuda.max_memory_allocated() > 6 * 2**30
                ):
                    raise RuntimeError("Registered fit ceiling exceeded")
            trace.append(total / n)
        torch.cuda.synchronize()
        file = folder / "lookup.safetensors"
        save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(file)
        )
        spec = dict(
            architecture="pictured_action_lookup_v1",
            sha256=sha(file),
            parent_reader=sha(ROOT / "models/visual_clause_roles/focus.safetensors"),
            parent_writer=sha(ROOT / "models/visual_word_senses/writer.safetensors"),
            adapter=sha(ROOT / "models/pronoun_writer/adapter.safetensors"),
            parameters=sum(p.numel() for p in net.parameters()),
            training=split["train_sha256"],
            steps=steps,
            loss=trace,
            seconds=monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            energy=consumed(energy, snapshot()),
        )
        dump(folder / "weights.json", spec)
        dump(PUBLIC / (arm + "_fit.json"), spec)
        print(arm, spec["seconds"], trace, flush=True)


if __name__ == "__main__":
    with gpu_owner():
        train()
