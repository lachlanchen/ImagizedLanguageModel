"""Compare equal-exposure bilingual visual writers with and without whole-sentence alignment."""

from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.bilingual_alignment import BilingualTraining
from scripts.prepare_bilingual_ink import (
    WORK,
    PUB,
    MODEL,
    BASE,
    FIRST,
    ROOT,
    read,
    dump,
    sha,
)
from scripts.evaluate_proposition_translation import model as parent
from scripts.train_span_choice import gpu_owner
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


def arrays(core, corelengths, replay, pairs, epoch):
    rng = np.random.default_rng(618073 + epoch)
    extra = [
        pool[i] for pool in replay for i in rng.choice(len(pool), 12000, replace=False)
    ]
    for i in rng.choice(len(pairs), min(len(pairs), 6000), replace=False):
        extra.extend(pairs[i]["cells"])
    data = np.zeros((len(core) + len(extra), 2, 96), np.uint16)
    lengths = np.zeros((len(data), 2), np.uint8)
    data[: len(core)] = core
    lengths[: len(core)] = corelengths
    for i, pair in enumerate(extra, len(core)):
        for t, seq in enumerate(pair):
            data[i, t, : len(seq)] = seq
            lengths[i, t] = len(seq)
    bins = lengths[:, 0].astype(int) // 12 * 20 + lengths[:, 1].astype(int) // 8
    order = rng.permutation(len(data))
    order = order[np.argsort(bins[order], kind="stable")]
    groups = np.split(order, np.where(np.diff(bins[order]))[0] + 1)
    batches = [g[i : i + 128] for g in groups for i in range(0, len(g), 128)]
    # Equality codes refer to complete, padded physical streams. They are used only to mask
    # duplicate translations within a minibatch of contrastive training examples.
    ids = np.empty((len(data), 2), np.int32)
    for t in (0, 1):
        raw = np.ascontiguousarray(data[:, t, :]).view(np.dtype((np.void, 192))).ravel()
        ids[:, t] = np.unique(raw, return_inverse=True)[1]
    return data, lengths, ids, [batches[j] for j in rng.permutation(len(batches))]


def main():
    configure_cuda_cli()
    binds = {
        s: sha(ROOT / s)
        for s in (
            "scripts/train_bilingual_ink.py",
            "scripts/prepare_bilingual_ink.py",
            "ilm/visual_lm/bilingual_alignment.py",
            "ilm/visual_lm/ordered_translation.py",
            "references/bilingual_ink_alignment_protocol.md",
            "references/evidence/bilingual_ink_v1/corpus.json",
            "models/proposition_translation/likelihood/weights.safetensors",
            "references/evidence/proposition_translation_v1/design.json",
        )
    }
    for base, files in (
        (WORK, read(PUB / "corpus.json")["files"]),
        (
            FIRST,
            {
                k + ".json": v
                for k, v in read(
                    ROOT / "references/evidence/proposition_translation_v1/design.json"
                )["files"].items()
            },
        ),
    ):
        for name, h in files.items():
            if sha(base / name) != h:
                raise ValueError(name)
    core = np.load(WORK / "images.npy", mmap_mode="r")
    lens = np.load(WORK / "lengths.npy", mmap_mode="r")
    pairs = read(FIRST / "contrasts.json")
    replay = read(FIRST / "replay.json")
    with gpu_owner():
        torch.set_num_threads(2)
        f = torch.tensor(
            observations({"tiles": np.load(BASE / "palette.npy")}, "ink"), device="cuda"
        )
        inv = [torch.tensor(v, device="cuda") for v in read(BASE / "inventories.json")]
        local = [torch.full((len(f),), -100, device="cuda") for _ in inv]
        for l, v in zip(local, inv):
            l[v] = torch.arange(len(v), device="cuda")
        for mode in ("likelihood", "aligned"):
            path = MODEL / mode
            path.mkdir(parents=True, exist_ok=True)
            receipt = path / "fit.json"
            weights = path / "weights.safetensors"
            if receipt.exists():
                if (
                    read(receipt)["binding"] != binds
                    or sha(weights) != read(receipt)["weights"]
                ):
                    raise ValueError(mode)
                continue
            torch.manual_seed(512903)
            net = BilingualTraining(parent("likelihood", "cuda")).cuda().train()
            opt = torch.optim.AdamW(net.parameters(), lr=0.00055, fused=True)
            ck = WORK / (mode + "-resume.pt")
            hist = []
            if ck.exists():
                d = torch.load(ck, map_location="cpu", weights_only=True)
                if d["binds"] != binds:
                    raise ValueError("Restart mismatch")
                net.load_state_dict(d["net"])
                opt.load_state_dict(d["opt"])
                hist = d["hist"]
            for ep in range(len(hist), 4):
                start = monotonic()
                meter = snapshot()
                torch.cuda.reset_peak_memory_stats()
                data, lengths, identities, batches = arrays(
                    core, lens, replay, pairs, ep
                )
                gpu = torch.as_tensor(data.astype(np.int32), device="cuda")
                ids = torch.as_tensor(identities, device="cuda")
                nll = closs = 0.0
                positions = 0
                opt.param_groups[0]["lr"] = [0.00055, 0.00042, 0.00030, 0.00018][ep]
                for idx in batches:
                    maxi = lengths[idx].max(0)
                    j = torch.as_tensor(idx, device="cuda")
                    n = torch.as_tensor(lengths[idx].astype(int), device="cuda")
                    rows = gpu[j].long()
                    sides = [rows[:, t, : int(m)] for t, m in enumerate(maxi)]
                    target = [
                        torch.nn.functional.pad(z, (0, 1)).scatter_(
                            1, n[:, t : t + 1], 1
                        )
                        for t, z in enumerate(sides)
                    ]
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ce, alignment, count = net(
                            f, sides, target, inv, local, ids[j], mode == "aligned"
                        )
                        loss = ce + 0.12 * alignment
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), 1, error_if_nonfinite=True
                    )
                    opt.step()
                    c = int(count)
                    nll += float(ce.detach()) * c
                    closs += float(alignment.detach()) * len(j)
                    positions += c
                torch.cuda.synchronize()
                check_budget(start, sum(r["seconds"] for r in hist))
                hist.append(
                    dict(
                        epoch=ep + 1,
                        pairs=len(data),
                        positions=positions,
                        nll=nll / positions,
                        alignment=closs / len(data),
                        seconds=monotonic() - start,
                        joules=consumed(meter, snapshot())["joules"],
                        peak=torch.cuda.max_memory_allocated(),
                    )
                )
                atomic_save(
                    lambda p: torch.save(
                        dict(
                            net=net.state_dict(),
                            opt=opt.state_dict(),
                            binds=binds,
                            hist=hist,
                        ),
                        p,
                    ),
                    ck,
                )
                print(mode, hist[-1], flush=True)
            atomic_save(
                lambda p: save_file(
                    {
                        k: v.cpu().contiguous()
                        for k, v in net.writer.state_dict().items()
                    },
                    p,
                ),
                weights,
            )
            projection = WORK / "alignment-projection.safetensors"
            if mode == "aligned":
                atomic_save(
                    lambda p: save_file(
                        {
                            k: v.cpu().contiguous()
                            for k, v in net.project.state_dict().items()
                        },
                        p,
                    ),
                    projection,
                )
            dump(
                receipt,
                dict(
                    binding=binds,
                    weights=sha(weights),
                    epochs=hist,
                    parameters=sum(p.numel() for p in net.writer.parameters()),
                    training_projection=sha(projection) if mode == "aligned" else None,
                ),
            )


if __name__ == "__main__":
    main()
