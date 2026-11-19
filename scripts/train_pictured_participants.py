"""Matched versus label-exchanged image training for Chinese grammatical participants."""

import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
from opencc import OpenCC
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.interval_writer import MAX_RUN
from ilm.visual_lm.participant_reader import ParticipantReader
from scripts.pictured_participant_scenes import EVENTS, FRAMES, PREFIXES, QUESTIONS
from scripts.prepare_pictured_grammatical_roles import DEST as GRAMMAR
from scripts.train_interval_writer import (
    DEVICE, PUBLIC as PREVIOUS_PUBLIC, ROOT, START, WORK as PREVIOUS, PairCorpus, SentenceCorpus,
    load, load_visual_writer, memory_bank, physical_tables, sha, to_batch, write_json,
)
from scripts.train_visual_role_writer import Tiles
from scripts.visual_role_examples import TRAIN_NAMES

WORK = ROOT / "artifacts/pictured_participant_reader_v1"
PUBLIC = ROOT / "references/evidence/pictured_participant_reader_20261119"
PRIOR = PREVIOUS / "calibrated/interval.safetensors"
SEED = 53231
EPOCHS = 10


def binding():
    files = [Path(__file__), ROOT / "scripts/evaluate_pictured_participants.py",
             ROOT / "scripts/pictured_participant_scenes.py", ROOT / "scripts/explicit_participant_cases.py",
             ROOT / "ilm/visual_lm/participant_reader.py", ROOT / "references/pictured_participant_protocol.md",
             GRAMMAR / "receipt.json", PRIOR, START / "final.safetensors",
             PREVIOUS_PUBLIC / "calibrated.json", PREVIOUS_PUBLIC / "grammar_source.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def treebank(bank, split):
    directory = GRAMMAR / split
    receipt = load(PREVIOUS_PUBLIC / "grammar_source.json")
    for name, digest in receipt["splits"][split]["files"].items():
        assert sha(directory / name) == digest
    remap = bank.merge(np.load(directory / "pixels.npy"))
    def read(field):
        with np.load(directory / (field + ".npz")) as packed:
            offsets, cells = packed["offsets"], remap[packed["cells"]]
        return [cells[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    meta = load(directory / "rows.json")
    streams = [read(k) for k in ("sentence", "subject_question", "object_question", "subject", "object")]
    pairs = []
    for (p, qa, qb, ya, yb), row in zip(zip(*streams, strict=True), meta, strict=True):
        a, b = tuple(row["subject"]), tuple(row["object"])
        assert np.array_equal(ya, p[slice(*a)]) and np.array_equal(yb, p[slice(*b)])
        pairs.append((p, qa, qb, a, b))
    return pairs, meta


def illustrated(bank):
    raster = Tiles(bank)
    simplified = OpenCC("t2s")
    rng = np.random.default_rng(SEED)
    pairs = []
    for verb, agent, patient in EVENTS:
        terms = dict(v=verb, agent=agent, patient=patient)
        for frame in FRAMES:
            for q1, q2 in QUESTIONS:
                for convert in (str, simplified.convert):
                    questions = [raster(convert(q.format(**terms))) for q in (q1, q2)]
                    for _ in range(2):
                        names = rng.choice(TRAIN_NAMES, 2, replace=False)
                        prefix = rng.choice(PREFIXES)
                        for left, right in (names, names[::-1]):
                            a, b = convert(left), convert(right)
                            text = convert(prefix + frame.format(a=left, b=right, v=verb))
                            assert text.count(a) == text.count(b) == 1
                            x, y = text.index(a), text.index(b)
                            pairs.append((raster(text), *questions, (x, x+len(a)), (y, y+len(b))))
    return pairs


def make_model(seed):
    torch.manual_seed(seed)
    head = ParticipantReader().to(DEVICE)
    state = load_file(str(PRIOR))
    assert sha(PRIOR) == load(PREVIOUS / "calibrated/fit.json")["weights"]
    missing, extra = head.load_state_dict(state, strict=False)
    assert not extra and set(missing) == {n for n, p in head.named_parameters() if p.requires_grad}
    return head


def fit(arm, natural, events, replay, writer, ink):
    dest = WORK / arm
    dest.mkdir()
    head = make_model(SEED)
    trainable = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=.0005)
    count = math.ceil(len(events) / 32)
    start = monotonic()
    trace = []
    presentations = [0, 0, 0]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(SEED+epoch)
        shuffle = np.random.default_rng(SEED+1000+epoch)
        chunks = [np.array_split(rng.permutation(len(ds)), count) for ds in (natural, events)]
        total_cost = 0
        for t, (gs, es) in enumerate(zip(*chunks, strict=True)):
            step = epoch * count + t + 1
            opt.param_groups[0]["lr"] = (.00008 + .00042*(1+math.cos(math.pi*step/(count*EPOCHS)))/2) * min(1, step/30)
            rows = []
            for items, indices in ((natural, gs), (events, es)):
                for idx in indices:
                    p, q1, q2, a, b = items[idx]
                    exchange = shuffle.integers(2)
                    if arm == "exchanged" and exchange:
                        a, b = b, a
                    rows.extend(((p, q1, [a], b), (p, q2, [b], a)))
            for idx in rng.integers(len(replay.data.questions), size=24):
                p, q, spans = replay.common(idx, epoch)
                if spans:
                    rows.append((p, q, spans, None))
            presentations[0] += 2*len(gs)
            presentations[1] += 2*len(es)
            presentations[2] += len(rows) - 2*(len(gs)+len(es))
            rows.sort(key=lambda row: len(row[0]))
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(rows), 32):
                chunk = rows[i:i+32]
                ps, qs, gold, other = zip(*chunk, strict=True)
                p, q = to_batch(ps, qs)
                with torch.no_grad():
                    encoded = writer.encode(ink, p, q)
                _, scores = head(encoded, p)
                flat = scores.flatten(1)
                encode = lambda span: span[0]*MAX_RUN + span[1]-span[0]-1
                width = max(map(len, gold))
                indices = torch.tensor([[encode(s) for s in spans]+[-1]*(width-len(spans))
                                        for spans in gold], device=DEVICE)
                positive = flat.gather(1, indices.clamp_min(0)).masked_fill(indices<0, -torch.inf).logsumexp(1)
                competitors = torch.tensor([encode(s) if s is not None else 0 for s in other], device=DEVICE)
                valid = torch.tensor([s is not None for s in other], device=DEVICE)
                rival = flat.gather(1, competitors[:, None]).squeeze(1)
                loss = ((flat.logsumexp(1)-positive).sum()+F.softplus((rival-positive)[valid]).sum())/len(rows)
                loss.backward()
                total_cost += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            opt.step()
            if monotonic()-start > 900 or torch.cuda.max_memory_allocated() > 6*1024**3:
                raise RuntimeError("Training resource ceiling reached")
        trace.append(total_cost/count)
        print(arm, epoch+1, trace[-1], flush=True)
    state = {k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}
    assert all(torch.equal(state[k], v) for k,v in load_file(str(PRIOR)).items())
    path = dest / "reader.safetensors"
    save_file(state, str(path))
    write_json(dest/"fit.json", dict(weights=sha(path), epochs=trace, updates=EPOCHS*count,
        seconds=monotonic()-start, peak_bytes=torch.cuda.max_memory_allocated(), presentations=presentations,
        trainable_parameters=sum(p.numel() for p in trainable)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    reg = PUBLIC / "registration.json"
    if parser.parse_args().register:
        PUBLIC.mkdir(parents=True, exist_ok=True)
        if reg.exists():
            raise FileExistsError(reg)
        write_json(reg, binding())
        return
    assert binding() == load(reg) and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        bank = memory_bank()
        natural, _ = treebank(bank, "train")
        events = illustrated(bank)
        replay = SentenceCorpus(PairCorpus(bank, "train"), bank, "train")
        writer = load_visual_writer(START, DEVICE).requires_grad_(False)
        features, _, _ = physical_tables(bank, DEVICE)
        ink = writer.ink(features).detach()
        print(f"Loaded {len(natural)} training predicates and {len(events)} pictured question pairs", flush=True)
        for arm in ("matched", "exchanged"):
            fit(arm, natural, events, replay, writer, ink)
        from scripts.evaluate_pictured_participants import evaluate
        evaluate()


if __name__ == "__main__":
    main()
