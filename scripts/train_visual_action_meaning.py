"""Train a visual English meaning encoder and two matched Chinese action readers."""

import argparse
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.gloss_focus import cell_features
from ilm.visual_lm.visual_action_meaning import OpticalWordMeaning, VisualActionMeaning
from scripts.answer_pictured_action import restore_lookup
from scripts.prepare_visual_action_meaning import WORK, PARENT, ROOT, read, dump, sha
from scripts.pictured_action_questions import TRAIN_FRAMES
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES
from scripts.train_pictured_action_lookup import matching_loss
from scripts.train_asked_gloss import packed_batch
from scripts.prepare_pronoun_writer import window
from scripts.write_visual_gloss import restore

SEED = 38241
D = "cuda"


def save(net, path):
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
        str(path),
    )


def positions(strings, face, width):
    chars = sorted({c for s in strings for c in s})
    alphabet = [" "] + [c for c in chars if c != " "]
    ids = {c: i for i, c in enumerate(alphabet)}
    raw, _ = packed_batch(alphabet, face, D)
    x = np.zeros((len(strings), width), np.int32)
    for i, s in enumerate(strings):
        x[i, : min(width, len(s))] = [ids[c] for c in s[:width]]
    return cell_features(raw[:, 0]), torch.tensor(x, device=D)


def fit_words(face):
    names = read(WORK / "words.json")
    target = torch.from_numpy(np.load(WORK / "targets.npy")).to(D)
    # Sparse suffix/prefix perturbations expose punctuation attached to a word.
    forms = [
        lambda s: s,
        lambda s: s + "?",
        lambda s: "'" + s + "',",
        lambda s: s + ".",
        lambda s: s + ",",
        lambda s: "'" + s + "'.",
    ]
    forms = [fn(s) for fn in forms for s in names]
    letters, ids = positions(forms, face, OpticalWordMeaning.width)
    torch.manual_seed(SEED)
    net = OpticalWordMeaning().to(D)
    optim = torch.optim.AdamW(net.parameters(), lr=0.001)
    start = monotonic()
    energy = snapshot()
    torch.cuda.reset_peak_memory_stats()
    trace = []
    for step in range(10000):
        ix = torch.randint(len(names), (384,), device=D)
        variant = torch.randint(10, (len(ix),), device=D).sub(4).clamp_min(0)
        pred = net(net.cell(letters)[ids[ix + len(names) * variant].long()])
        loss = (1 - (pred * target[ix]).sum(-1)).mean()
        for group in optim.param_groups:
            group["lr"] = 0.00018 + 0.00082 * (1 + np.cos(step * np.pi / 10000)) / 2
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % 500 == 0:
            trace.append(float(loss.detach()))
            print("word step", step + 1, trace[-1], flush=True)
        if monotonic() - start > 300 or torch.cuda.max_memory_allocated() > 6 * 2**30:
            raise RuntimeError("Optical word fit exceeded ceiling")
    torch.cuda.synchronize()
    elapsed = monotonic() - start
    cost = consumed(energy, snapshot())
    peak = torch.cuda.max_memory_allocated()
    with torch.no_grad():
        held = torch.tensor(
            np.random.default_rng(SEED + 1).permutation(len(names))[:1024], device=D
        )
        pred = net(net.cell(letters)[ids[held].long()])
        nearest = (pred @ target.T).argmax(1)
        quality = dict(
            examples=len(held),
            top1=int((nearest == held).sum()),
            mean_cosine=float((pred * target[held]).sum(1).mean()),
        )
    save(net, WORK / "optical_words.safetensors")
    dump(
        WORK / "word_fit.json",
        dict(
            updates=10000,
            presentations=3840000,
            parameters=sum(p.numel() for p in net.parameters()),
            seconds=elapsed,
            peak_bytes=peak,
            energy=cost,
            loss=trace,
            seen_word_reconstruction=quality,
            sha256=sha(WORK / "optical_words.safetensors"),
        ),
    )
    return net.eval().requires_grad_(False)


@torch.no_grad()
def query_cache(queries, face, net):
    split = [s.split() for s in queries]
    forms = sorted({s for q in split for s in q})
    ix = {s: i + 1 for i, s in enumerate(forms)}
    letters, ids = positions(forms, face, net.width)
    project = net.cell(letters)
    zero = letters.new_zeros(1, 50)
    allwords = torch.cat(
        [zero] + [net(project[chunk.long()]) for chunk in ids.split(512)]
    )
    q = np.zeros((len(split), max(map(len, split))), np.int32)
    for i, s in enumerate(split):
        q[i, : len(s)] = [ix[w] for w in s]
    return (
        allwords,
        torch.tensor(q, device=D),
        torch.tensor(list(map(len, split)), device=D),
    )


def fit(rows_path):
    if sha(rows_path) != read(PARENT / "split.json")["train_sha256"]:
        raise ValueError("Unexpected source rows")
    # Training outputs are immutable; never replace finished fits on a retry.
    if any(
        (WORK / n).exists()
        for n in ("optical_words.safetensors", "aligned", "exchanged")
    ):
        raise FileExistsError(WORK)
    torch.set_float32_matmul_precision("high")
    face = Face(FACES[0])
    net = fit_words(face)
    rows = read(rows_path)
    writer = restore(device=D)
    keys = []
    with torch.no_grad():
        for a in range(0, len(rows), 128):
            raw, _ = packed_batch(
                [window(r["page"], r["focus"][0]) for r in rows[a : a + 128]], face, D
            )
            keys.append(writer.encode(cell_features(raw)))
    states = torch.cat(keys)
    descr = read(WORK / "descriptions.json")
    labels = sorted(descr)
    lookup = {s: i for i, s in enumerate(labels)}
    ids = np.array([lookup[r["gloss"][0][0]] for r in rows])
    picks = []
    queries = []
    literal = []
    for s in labels:
        literal.append(np.arange(len(queries), len(queries) + len(TRAIN_FRAMES)))
        queries.extend(f.format(s) for f in TRAIN_FRAMES)
        start = len(queries)
        queries.extend(f.format(term) for term in descr[s] for f in TRAIN_FRAMES)
        picks.append(
            np.arange(start, len(queries)) if start < len(queries) else literal[-1]
        )
    words, q, counts = query_cache(queries, face, net)
    literal = np.array(literal)
    vocabulary = {s: i for i, s in enumerate(read(WORK / "words.json"))}
    vectors = np.load(WORK / "targets.npy")
    target = torch.from_numpy(
        np.stack(
            [
                vectors[vocabulary[s]] if s in vocabulary else np.zeros(50, np.float32)
                for s in labels
            ]
        )
    ).to(D)
    parent = ROOT / "models/pictured_action_lookup"
    genuine = restore_lookup(parent, D)
    for arm in ("aligned", "exchanged"):
        folder = WORK / arm
        folder.mkdir()
        torch.manual_seed(SEED + 2)
        model = VisualActionMeaning().to(D)
        model.words.load_state_dict(net.state_dict())
        model.words.requires_grad_(False)
        model.parent.load_state_dict(genuine.state_dict())
        model.parent.requires_grad_(False)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )

        def embed(indices):
            lens = counts[indices]
            table = q[indices, : int(lens.max())]
            return model.meaning(words[table.long()], lens)

        start = monotonic()
        energy = snapshot()
        torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(SEED + 3)
        trace = []
        for step in range(2200):
            ix = rng.integers(len(ids), size=128)
            lab = ids[ix]
            group = torch.as_tensor(lab, device=D)
            ground = target[group]
            a = literal[lab, rng.integers(len(TRAIN_FRAMES), size=len(lab))]
            b = np.array([picks[k][rng.integers(len(picks[k]))] for k in lab])
            perm = rng.permutation(len(b))
            if arm == "exchanged":
                b = b[perm]
            literal_q = embed(a)
            description = embed(b)
            src = F.normalize(model.source(states[ix]), dim=-1)
            align = lambda x, y: 1 - (x * y).sum(-1).mean()
            loss = (
                matching_loss(src @ literal_q.T / 0.075, group)
                + matching_loss(src @ description.T / 0.075, group)
            ) / 2
            loss += (
                0.5 * align(literal_q, description)
                + 0.5 * align(src, ground)
                + 0.35 * align(description, ground)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            if (step + 1) % 200 == 0:
                trace.append(float(loss.detach()))
            if (
                monotonic() - start > 300
                or torch.cuda.max_memory_allocated() > 6 * 2**30
            ):
                raise RuntimeError("Comparator exceeded ceiling")
        torch.cuda.synchronize()
        elapsed = monotonic() - start
        cost = consumed(energy, snapshot())
        file = folder / "lookup.safetensors"
        save(model, file)
        spec = read(parent / "weights.json")
        spec.update(
            architecture="visual_action_meaning_v1",
            sha256=sha(file),
            inherited_lookup=sha(parent / "lookup.safetensors"),
            parameters=sum(p.numel() for p in model.parameters()),
            trained_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            semantic_words=sha(WORK / "optical_words.safetensors"),
            steps=2200,
            loss=trace,
            seconds=elapsed,
            peak_bytes=torch.cuda.max_memory_allocated(),
            energy=cost,
        )
        dump(folder / "weights.json", spec)
        print(arm, trace[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True, type=Path)
    with gpu_owner():
        fit(p.parse_args().rows)
