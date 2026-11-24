"""Train matched independent and interacting answer-span scorers on real optical pages."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.fact_ranking import candidate_loss
from ilm.visual_lm.optical_competition import CandidateCompetition, shortlist
from scripts.prepare_optical_competition import (
    ROOT,
    WORK,
    PUB,
    MODEL,
    PROTOCOL,
    read,
    dump,
    sha,
)
from scripts.prepare_directional_reading import MODEL as PRIOR
from scripts.train_directional_reading import restore as directional
from scripts.train_fact_channel import Data, join_regions, pack, gpu_owner, SEED
from scripts.train_fact_ranking import (
    atomic_save,
    check_budget,
    restore as earlier,
    ORIGINAL,
    DATA,
    MODEL as RANKER,
)
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


def binding():
    files = [
        Path(__file__),
        PROTOCOL,
        PUB / "reservation.json",
        DATA / "corpus.json",
        ROOT / "formal/span_list/CandidateCompetition.lean",
        ROOT / "ilm/visual_lm/optical_competition.py",
        PRIOR / "signed/weights.safetensors",
        ORIGINAL / "reader/weights.safetensors",
        ORIGINAL / "human/weights.safetensors",
        RANKER / "whole/weights.safetensors",
    ]
    return {str(f.relative_to(ROOT)): sha(f) for f in files}


@torch.inference_mode()
def bank():
    bound = binding()
    cache = WORK / "training-lists.pt"
    receipt = PUB / "lists.json"
    if receipt.exists() and cache.exists():
        record = read(receipt)
        if record["sources"] != bound or record["sha256"] != sha(cache):
            raise ValueError("Training lists changed")
        return torch.load(cache, mmap=True, weights_only=True)
    reader = directional("signed", "cuda")
    parent = earlier("whole", "cuda")
    data = Data("reader", "cuda")
    n = len(data.rows)
    store = dict(
        images=torch.zeros(n, 32, 3, 128, dtype=torch.float16),
        base=torch.full((n, 32), -torch.inf),
        lengths=torch.ones(n, 32, dtype=torch.int64),
        correct=torch.zeros(n, 32, dtype=torch.bool),
    )
    order = sorted(range(n), key=lambda i: sum(len(v) for v in data.pages([i])[0]))
    domains = Counter()
    start = monotonic()
    meter = snapshot()
    for j in range(0, n, 32):
        ix = order[j : j + 32]
        q = pack([data.seq[data.rows[i][0]] for i in ix], "cuda")
        p = pack([join_regions(rs) for rs in data.pages(ix)], "cuda")
        im, base, s, l = shortlist(reader, data.features, q, p)
        scores = parent(im, base, l)
        k = scores.size(1)
        store["images"][ix, :k] = im.cpu()
        store["base"][ix, :k] = scores.cpu()
        store["lengths"][ix, :k] = l.cpu()
        for u, (i, ss, ll, valid) in enumerate(
            zip(ix, s.tolist(), l.tolist(), base.isfinite().tolist(), strict=True)
        ):
            accepted = set(map(tuple, data.rows[i][2]))
            hits = torch.tensor(
                [
                    v and (x, y) in accepted
                    for x, y, v in zip(ss, ll, valid, strict=True)
                ]
            )
            store["correct"][i, :k] = hits
            domains[data.rows[i][-2]] += int(hits.any())
        check_budget(start)
    torch.cuda.synchronize()
    seconds = monotonic() - start
    energy = consumed(meter, snapshot())["joules"]
    atomic_save(lambda p: torch.save(store, p), cache)
    dump(
        receipt,
        dict(
            rows=n,
            supported_by_domain=domains,
            sources=bound,
            sha256=sha(cache),
            seconds=seconds,
            joules=energy,
        ),
    )
    return store


def restore(mode, device):
    folder = MODEL / mode
    rc = read(folder / "fit.json")
    if rc["mode"] != mode or sha(folder / "weights.safetensors") != rc["sha256"]:
        raise ValueError("Altered scorer")
    for p, d in rc["sources"].items():
        if p.startswith("models/") and sha(ROOT / p) != d:
            raise ValueError(p)
    net = CandidateCompetition(mode)
    net.load_state_dict(load_file(str(folder / "weights.safetensors")))
    return net.to(device).eval()


def train(mode, store):
    folder = MODEL / mode
    folder.mkdir(exist_ok=True, parents=True)
    sources = binding()
    weights = folder / "weights.safetensors"
    fit = folder / "fit.json"
    if fit.exists():
        if read(fit)["sources"] != sources:
            raise ValueError("Mismatched completed fit")
        restore(mode, "cpu")
        return
    torch.manual_seed(SEED)
    net = CandidateCompetition(mode).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, fused=True)
    rows = np.flatnonzero(store["correct"].any(1).cpu().numpy())
    order = np.random.default_rng(SEED).permuted(np.tile(rows, (8, 1)), axis=1)
    schedule = sha256(order.astype("<i8").tobytes()).hexdigest()
    ck = WORK / (mode + "-resume.pt")
    history = []
    if ck.exists():
        state = torch.load(ck, weights_only=True, map_location="cpu")
        if (
            state["sources"] != sources
            or state["order"] != schedule
            or state["mode"] != mode
        ):
            raise ValueError("Resume state changed")
        history = state["history"]
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
    for epoch in range(len(history), 8):
        loss_sum = 0.0
        good = 0
        t = monotonic()
        energy = snapshot()
        torch.cuda.reset_peak_memory_stats()
        for ix in np.array_split(order[epoch], (len(rows) + 127) // 128):
            logits = net(store["images"][ix], store["base"][ix], store["lengths"][ix])
            y = store["correct"][ix]
            loss = candidate_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            loss_sum += float(loss.detach()) * len(ix)
            good += int(y.gather(1, logits.detach().argmax(1, keepdim=True)).sum())
            check_budget(t, sum(z["seconds"] for z in history))
        torch.cuda.synchronize()
        history.append(
            dict(
                epoch=epoch + 1,
                questions=len(rows),
                mean_loss=loss_sum / len(rows),
                correct=good,
                seconds=monotonic() - t,
                joules=consumed(energy, snapshot())["joules"],
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        atomic_save(
            lambda p: torch.save(
                dict(
                    net=net.state_dict(),
                    opt=opt.state_dict(),
                    order=schedule,
                    mode=mode,
                    history=history,
                    sources=sources,
                ),
                p,
            ),
            ck,
        )
        print(mode, history[-1], flush=True)
    atomic_save(
        lambda p: save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
        ),
        weights,
    )
    dump(
        fit,
        dict(
            mode=mode,
            sources=sources,
            sha256=sha(weights),
            schedule=schedule,
            epochs=history,
            parameters=sum(v.numel() for v in net.parameters()),
            bank=read(PUB / "lists.json")["sha256"],
        ),
    )


if __name__ == "__main__":
    configure_cuda_cli()
    torch.set_num_threads(3)
    with gpu_owner():
        b = {k: v.cuda() for k, v in bank().items()}
        for mode in CandidateCompetition.modes:
            train(mode, b)
