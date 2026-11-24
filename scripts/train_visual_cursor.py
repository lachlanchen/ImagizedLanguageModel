"""Train matched optical writers on full human responses, books and instructions."""

import hashlib
import math
from time import monotonic
import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.source_cursor import SourceCursor
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_cursor import WORK, PUBLIC, MODEL, read, dump, sha
from scripts.train_visual_discourse import Corpus, load_model
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

STEPS = 2400
SEED = 44129
TAIL = 144


def pad(seqs):
    arr = np.zeros((len(seqs), max(map(len, seqs))), np.int32)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return arr


def batch(data, pages, step):
    rng = np.random.default_rng([SEED, step])
    x = []
    y = []
    if (step - 1) % 8 < 6:
        k = ("coqa", "cmrc", "drcd")[(step - 1) % 8 % 3]
        for idx in rng.choice(pages[k], 24):
            a, b = pages["u"][idx : idx + 2]
            c, d = pages["v"][idx : idx + 2]
            x.append(pages["x"][a:b])
            y.append(pages["y"][c : min(d, c + 48)])
    else:
        for k in (
            "book-en",
            "book-zh",
            "instruction-en",
            "instruction-zh",
            "reference",
        ):
            for idx in rng.choice(
                data.groups[k], 4 if k == "reference" else 5, p=data.books.get(k)
            ):
                s = data.seq(idx)
                cut = data.rows[idx]["cut"]
                if k.startswith("book"):
                    if len(s) < 3:
                        continue
                    cut = int(rng.integers(1, len(s) - 1))
                x.append(s[max(0, cut - 768) : cut])
                y.append(s[cut : cut + 48])
    return x, y


def base_batch(x, y):
    xs = [s[-TAIL:] for s in x]
    ids = pad([np.concatenate((p, a)) for p, a in zip(xs, y, strict=True)])
    p = (np.asarray(list(map(len, xs)))[:, None] + np.arange(max(map(len, y)))).clip(
        0, ids.shape[1] - 1
    )
    return ids, p


def load_data():
    receipt = read(PUBLIC / "corpus.json")
    if receipt["pages"] != sha(WORK / "pages.npz"):
        raise ValueError("Changed QA pages")
    with np.load(WORK / "pages.npz") as file:
        return {k: file[k] for k in file.files}


def fit(data, pages, base, basebank, features, persistent):
    name = "persistent" if persistent else "independent"
    path = MODEL / (name + ".safetensors")
    receipt = PUBLIC / (name + ".json")
    if receipt.exists():
        if read(receipt)["weights"] != sha(path):
            raise ValueError("Changed final fit")
        return
    torch.manual_seed(SEED)
    model = SourceCursor().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, fused=True, weight_decay=0.02)
    digest = bytes(32)
    trace = []
    positions = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    meter = snapshot()
    start = monotonic()
    for step in range(1, STEPS + 1):
        x, y = batch(data, pages, step)
        a, b = pad(x), pad(y)
        positions += sum(map(len, y))
        digest = hashlib.sha256(digest + a.tobytes() + b.tobytes()).digest()
        ids, target = (torch.as_tensor(v, device="cuda") for v in (a, b))
        bi, bp = (torch.as_tensor(v, device="cuda") for v in base_batch(x, y))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mass = base.log_prob(basebank, bi, data.n, bp).exp()
        opt.param_groups[0]["lr"] = (
            0.00085
            * min(1, step / 80)
            * (0.15 + 0.85 * (1 + math.cos(math.pi * step / STEPS)) / 2)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = model.likelihood(features, ids, target, mass, data.n, persistent)
            loss = -((lp * (target > 0)).sum(1) / (target > 0).sum(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 100 == 0:
            trace.append([step, float(loss.detach())])
            print(name, step, trace[-1][1], round(monotonic() - start, 1), flush=True)
            if torch.cuda.max_memory_allocated() > 20 * 2**30:
                raise RuntimeError("Device ceiling")
    torch.cuda.synchronize()
    seconds = monotonic() - start
    energy = consumed(meter, snapshot())
    save_file(model.cpu().state_dict(), str(path))
    dump(
        receipt,
        dict(
            weights=sha(path),
            steps=STEPS,
            seconds=seconds,
            energy=energy,
            positions=int(positions),
            schedule=digest.hex(),
            corpus=sha(WORK / "pages.npz"),
            parameters=sum(p.numel() for p in model.parameters()),
            allocated=torch.cuda.max_memory_allocated(),
            trace=trace,
        ),
    )
    return


def main():
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    MODEL.mkdir(parents=True, exist_ok=True)
    data = Corpus()
    pages = load_data()
    with gpu_owner():
        base = load_model("ink-pair", "cuda").requires_grad_(False)
        features = torch.from_numpy(
            observations({"tiles": pages["tiles"]}, "ink")
        ).cuda()
        basebank = base.encode(features).detach()
        for p in (False, True):
            fit(data, pages, base, basebank, features, p)


if __name__ == "__main__":
    main()
