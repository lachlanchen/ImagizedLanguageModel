"""Fit optical evidence probabilities to complete stopped human responses."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.answer_evidence import answer_marginal
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.optical_competition import CandidateCompetition
from scripts.check_story_answer_objective import likelihoods
from scripts.prepare_story_answer_fit import (
    ROOT,
    WORK,
    PUB,
    MODEL,
    TRAIN,
    PARENT,
    PARENT_PUB,
    PROTOCOL,
    sha,
    read,
    dump,
)
from scripts.prepare_fact_channel import Ink
from scripts.train_fact_channel import join_regions, gpu_owner, SEED
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.write_directional_reading import DirectionalAnswer
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli

MODES = ("single", "marginal")


def binding():
    names = [
        Path(__file__),
        PROTOCOL,
        PUB / "reservation.json",
        ROOT / "scripts/check_story_answer_objective.py",
        ROOT / "ilm/visual_lm/answer_evidence.py",
        ROOT / "formal/span_list/AnswerConditionedEvidence.lean",
        ROOT / "ilm/visual_lm/optical_competition.py",
        PARENT_PUB / "corpus.json",
        *[
            ROOT / "models" / n / "weights.safetensors"
            for n in (
                "directional_reading/signed",
                "fact_channel/human",
                "fact_span_ranking/whole",
                "fact_channel/reader",
            )
        ],
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in names}


@torch.inference_mode()
def prepare():
    record = PUB / "bank.json"
    cache = WORK / "bank.pt"
    bound = binding()
    if record.exists() and cache.exists():
        rc = read(record)
        if rc["sources"] != bound or rc["sha256"] != sha(cache):
            raise ValueError("Bank changed")
        return torch.load(cache, mmap=True, weights_only=True)
    spec = read(PUB / "reservation.json")
    for f, checksum in spec["files"].items():
        if sha(WORK / f) != checksum:
            raise ValueError(f)
    corpus = read(PARENT_PUB / "corpus.json")["files"]
    for f in ("strings.npz", "tiles.npy", "writer.json"):
        if sha(PARENT / f) != corpus[f]:
            raise ValueError(f)
    if sha(TRAIN / "training.json") != spec["story_source"]:
        raise ValueError("Story material changed")
    ink = Ink(np.load(PARENT / "tiles.npy"))
    indices = read(WORK / "indices.json")
    sr = read(TRAIN / "training.json")
    rows = []
    for i in indices["stories"]:
        r = sr[i]
        rows.append(
            (
                r["task"],
                ink.encode(r["q"]),
                join_regions([ink.encode(s) for s in r["blocks"]]),
                list(dict.fromkeys(tuple(ink.encode(a)) + (1,) for a in r["answers"])),
            )
        )
    prior = read(PARENT / "writer.json")
    with np.load(PARENT / "strings.npz") as z:
        ids, ptr = z["ids"], z["ptr"]

    def seq(i):
        return ids[ptr[i] : ptr[i + 1]].tolist()

    for group, picks in indices["replay"].items():
        for i in picks:
            r = prior[i]
            assert r[-2] == group
            rows.append(
                (
                    group,
                    seq(r[0]),
                    join_regions([seq(k) for k in r[1]]),
                    list(dict.fromkeys(tuple(seq(k)) + (1,) for k in r[3])),
                )
            )
    net = DirectionalAnswer("signed", "cuda")
    features = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
    n = len(rows)
    store = dict(
        images=torch.zeros(n, 32, 3, 128, dtype=torch.float16),
        base=torch.full((n, 32), -torch.inf, dtype=torch.float64),
        lengths=torch.ones(n, 32, dtype=torch.long),
        likelihood=torch.full((n, 32), -torch.inf, dtype=torch.float64),
    )
    t = monotonic()
    meter = snapshot()
    counts = Counter()
    num_paths = 0
    torch.cuda.reset_peak_memory_stats()
    for i, (group, q, p, refs) in enumerate(rows):
        im, base, l, v = likelihoods(
            net, features, q, p, refs, "cuda", return_inputs=True
        )
        k = len(v)
        for name, value in zip(store, (im, base, l, v), strict=True):
            store[name][i, :k] = value.cpu()
        counts[group] += int(v.isfinite().any())
        num_paths += k * len(refs)
        if i % 512 == 0:
            print("score human wordings", i, n, flush=True)
            check_budget(t)
    torch.cuda.synchronize()
    sec = monotonic() - t
    energy = consumed(meter, snapshot())["joules"]
    atomic_save(lambda p: torch.save(store, p), cache)
    dump(
        record,
        dict(
            sha256=sha(cache),
            sources=bound,
            rows=n,
            supported=counts,
            paths=num_paths,
            seconds=sec,
            joules=energy,
            peak_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    return store


def fit(mode, bank):
    root = MODEL / mode
    root.mkdir(parents=True, exist_ok=True)
    path = root / "weights.safetensors"
    rc = root / "fit.json"
    bound = binding()
    bank_hash = read(PUB / "bank.json")["sha256"]
    if rc.exists():
        if read(rc)["sources"] != bound or read(rc)["bank"] != bank_hash:
            raise ValueError("Completed fit mismatch")
        restore(mode, "cpu")
        return
    torch.manual_seed(SEED)
    head = CandidateCompetition("independent").cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, fused=True)
    ix = np.flatnonzero(bank["likelihood"].isfinite().any(1).cpu().numpy())
    order = np.random.default_rng(SEED).permuted(np.tile(ix, (8, 1)), axis=1)
    schedule = sha256(order.astype("<i8").tobytes()).hexdigest()
    resume = WORK / (mode + "-resume.pt")
    hist = []
    if resume.exists():
        ck = torch.load(resume, weights_only=True, map_location="cpu")
        if (ck["sources"], ck["bank"], ck["mode"], ck["order"]) != (
            bound,
            bank_hash,
            mode,
            schedule,
        ):
            raise ValueError("Resume mismatch")
        hist = ck["epochs"]
        head.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
    for epoch in range(len(hist), 8):
        t = monotonic()
        meter = snapshot()
        joint_nll = loss_total = 0.0
        torch.cuda.reset_peak_memory_stats()
        for batch in np.array_split(order[epoch], (len(ix) + 95) // 96):
            z = head(bank["images"][batch], bank["base"][batch], bank["lengths"][batch])
            v = bank["likelihood"][batch]
            joint = -answer_marginal(z, v)
            loss = (
                joint.mean()
                if mode == "marginal"
                else -(z.double().log_softmax(-1) + v).max(-1).values.mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            joint_nll += float(joint.detach().sum())
            loss_total += float(loss.detach()) * len(batch)
            check_budget(t, sum(r["seconds"] for r in hist))
        hist.append(
            dict(
                epoch=epoch + 1,
                rows=len(ix),
                mean_joint_nll=joint_nll / len(ix),
                mean_objective=loss_total / len(ix),
                seconds=monotonic() - t,
                joules=consumed(meter, snapshot())["joules"],
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        atomic_save(
            lambda p: torch.save(
                dict(
                    net=head.state_dict(),
                    opt=opt.state_dict(),
                    epochs=hist,
                    sources=bound,
                    mode=mode,
                    order=schedule,
                    bank=bank_hash,
                ),
                p,
            ),
            resume,
        )
        print(mode, hist[-1], flush=True)
    atomic_save(
        lambda p: save_file(
            {k: v.cpu().contiguous() for k, v in head.state_dict().items()}, p
        ),
        path,
    )
    dump(
        rc,
        dict(
            sources=bound,
            mode=mode,
            sha256=sha(path),
            order=schedule,
            bank=bank_hash,
            epochs=hist,
            parameters=sum(p.numel() for p in head.parameters()),
        ),
    )


def restore(mode, device):
    folder = MODEL / mode
    rc = read(folder / "fit.json")
    path = folder / "weights.safetensors"
    if rc["mode"] != mode or sha(path) != rc["sha256"]:
        raise ValueError("Story weights changed")
    for name, checksum in rc["sources"].items():
        if name.startswith("models/") and sha(ROOT / name) != checksum:
            raise ValueError(name)
    net = CandidateCompetition("independent")
    net.load_state_dict(load_file(str(path)))
    return net.to(device).eval()


if __name__ == "__main__":
    configure_cuda_cli()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    with gpu_owner():
        data = {k: v.cuda() for k, v in prepare().items()}
        for m in MODES:
            fit(m, data)
