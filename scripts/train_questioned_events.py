"""Train matched image-to-event heads on a single GPU."""

from time import monotonic
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.questioned_events import QuestionedEvents
from scripts.answer_visual_gloss import load_focus
from scripts.answer_pictured_action import restore_lookup
from scripts.prepare_questioned_events import PUB, WORK, ROOT, read, dump, sha
from scripts.pictured_action_questions import TRAIN_FRAMES
from scripts.train_asked_gloss import packed_batch
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore
from scripts.measure_writer_energy import snapshot, consumed

SEED = 510031
PARENT = ROOT / "models/visual_action_meaning/aligned"


@torch.no_grad()
def prepare(rows, face, writer, syntax, lookup, described):
    lengths = torch.tensor([len(r["page"]) for r in rows], device="cuda")
    width = int(lengths.max())
    n = len(rows)
    states = torch.zeros(n, width, 160, dtype=torch.float16, device="cuda")
    priors = torch.full(
        (n, width, width), -torch.inf, dtype=torch.float16, device="cuda"
    )
    for i in range(0, n, 96):
        batch = rows[i : i + 96]
        raw, l = packed_batch([r["page"] for r in batch], face, "cuda")
        h = encoding_windows(writer, cell_features(raw))
        a = syntax.arc_scores(h, l)
        w = h.shape[1]
        states[i : i + len(batch), :w] = h
        priors[i : i + len(batch), :w, :w] = a
    names = sorted({e[2] for r in rows for e in r["events"]})
    import csv

    with (PUB / "natural_questions.tsv").open() as f:
        held = {
            r[k] for r in csv.DictReader(f, delimiter="\t") for k in ("first", "second")
        }

    def training_question(frame, term):
        question = frame.format(term)
        if described and question in held:
            return next(
                f.format(term) for f in TRAIN_FRAMES if f.format(term) not in held
            )
        return question

    sentences = [training_question(f, name) for name in names for f in TRAIN_FRAMES]
    alternatives = []
    if described:
        file = PUB / "wordnet_descriptions.json"
        assert (
            sha(file)
            == read(
                ROOT / "references/evidence/visual_action_meaning_v1/preparation.json"
            )["dictionary_sha256"]
        )
        bank_terms = read(file)
        rng = np.random.default_rng(SEED + 1)
        for name in names:
            terms = bank_terms.get(name, [])
            indices = []
            for term in rng.permutation(terms)[:16]:
                for f in rng.choice(TRAIN_FRAMES, 2, replace=False):
                    indices.append(len(sentences))
                    sentences.append(training_question(f, term))
            alternatives.append(indices)
        assert held.isdisjoint(sentences)
    length = max(map(len, sentences))
    bank = torch.zeros(len(sentences), length, 146, device="cuda", dtype=torch.float16)
    mask = torch.zeros(len(sentences), length, device="cuda", dtype=torch.bool)
    for i in range(0, len(sentences), 64):
        pix, l = packed_batch(sentences[i : i + 64], face, "cuda")
        q, m = lookup.questions(cell_features(pix), l)
        bank[i : i + len(q), : q.shape[1]] = q
        mask[i : i + len(q), : q.shape[1]] = m
    return (
        states,
        priors,
        lengths,
        bank,
        mask,
        {s: i for i, s in enumerate(names)},
        alternatives,
    )


def fit(folder, described=False):
    path = folder / "train.json"
    reg = read(PUB / "split.json")
    if sha(path) != reg["training_sha256"]:
        raise ValueError("Unexpected training pages")
    folder = folder / "described" if described else folder
    folder.mkdir(exist_ok=True)
    if any((folder / a).exists() for a in ("aligned", "exchanged")):
        raise FileExistsError(folder)
    rows = read(path)
    face = Face(FACES[0])
    writer = restore(device="cuda").requires_grad_(False)
    syntax = load_focus(ROOT / "models/visual_clause_roles", "cuda").requires_grad_(
        False
    )
    lookup = restore_lookup(PARENT, "cuda").requires_grad_(False)
    started = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    states, priors, lengths, words, mask, ids, alternatives = prepare(
        rows, face, writer, syntax, lookup, described
    )
    torch.cuda.synchronize()
    dump(
        folder / "cache_cost.json",
        dict(
            seconds=monotonic() - started,
            energy=consumed(meter, snapshot()),
            peak_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    ne = np.array([len(r["events"]) for r in rows])
    single = np.flatnonzero(ne == 1)
    multi = np.flatnonzero(ne > 1)
    truth = np.zeros((len(rows), max(ne), 3), np.int64)
    for i, r in enumerate(rows):
        for j, (v, o, name, _) in enumerate(r["events"]):
            truth[i, j] = [v, o, ids[name]]
    for arm in ("aligned", "exchanged"):
        target = folder / arm
        target.mkdir()
        torch.manual_seed(SEED)
        net = QuestionedEvents().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=0.001)
        view_rng = np.random.default_rng(SEED + 2)
        rng = np.random.default_rng(SEED)
        losses = []
        start = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        for step in range(1600):
            ids_batch = np.r_[rng.choice(multi, 48), rng.choice(single, 16)]
            j = rng.integers(ne[ids_batch])
            v, o, n = truth[ids_batch, j].T
            if arm == "exchanged" and not described:
                o = truth[ids_batch, (j + 1) % ne[ids_batch], 1]
            frame = rng.integers(len(TRAIN_FRAMES), size=64)
            qi = n * len(TRAIN_FRAMES) + frame
            w = int(lengths[ids_batch].max())
            h = states[ids_batch, :w].float()
            a, k = net.page(h, priors[ids_batch, :w, :w].float(), lookup.predicates(h))
            score = net.scores(a, k, words[qi].float(), mask[qi])
            loss = F.cross_entropy(
                score.flatten(1), torch.as_tensor(v * w + o, device="cuda")
            )
            if described:
                other = qi.copy()
                eligible = np.flatnonzero([bool(alternatives[c]) for c in n])
                other[eligible] = [
                    view_rng.choice(alternatives[n[k]]) for k in eligible
                ]
                shuffled = view_rng.permutation(eligible)
                if arm == "exchanged":
                    other[eligible] = other[shuffled]
                second = net.scores(a, k, words[other].float(), mask[other])
                loss = (
                    loss
                    + F.cross_entropy(
                        second.flatten(1), torch.as_tensor(v * w + o, device="cuda")
                    )
                ) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            if (step + 1) % 200 == 0:
                losses.append(float(loss.detach()))
            if (
                monotonic() - start > 300
                or torch.cuda.max_memory_allocated() > 6 * 2**30
            ):
                raise RuntimeError("Fit exceeded registered resources")
        torch.cuda.synchronize()
        elapsed = monotonic() - start
        energy = consumed(meter, snapshot())
        file = target / "events.safetensors"
        save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(file)
        )
        spec = dict(
            architecture="questioned_events_v1",
            arm=arm,
            sha256=sha(file),
            reader=sha(PARENT / "lookup.safetensors"),
            training=sha(path),
            updates=1600,
            presentations=102400,
            question_exposures=204800 if described else 102400,
            description_training=described,
            parameters=sum(x.numel() for x in net.parameters()),
            seconds=elapsed,
            energy=energy,
            peak_bytes=torch.cuda.max_memory_allocated(),
            losses=losses,
        )
        dump(target / "weights.json", spec)
        print(arm, elapsed, losses[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--descriptions", action="store_true")
    args = parser.parse_args()
    with gpu_owner():
        fit(args.work, args.descriptions)
