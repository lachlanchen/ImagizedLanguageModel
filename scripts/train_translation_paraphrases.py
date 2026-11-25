"""Train from varied or canonical optical sources with exactly the same target stream."""

from collections import defaultdict
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_translation_paraphrases import (
    ROOT,
    BASE,
    FIRST,
    LARGE,
    WORK,
    MODEL,
    PUB,
    read,
    dump,
    sha,
)
from scripts.evaluate_bilingual_ink import model
from scripts.train_ordered_translation import run
from scripts.train_span_choice import gpu_owner
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


def schedule(examples, replay, images, lengths, epoch, varied):
    rng = np.random.default_rng(42821 + epoch)
    addition = [
        pool[i] for pool in replay for i in rng.choice(len(pool), 6000, replace=False)
    ]
    for i in rng.choice(len(images), 12000, replace=False):
        addition.append([images[i, t, : lengths[i, t]].tolist() for t in (0, 1)])
    rows = []
    groups = defaultdict(list)
    for t in (0, 1):
        for r in examples:
            z = r["target"]
            v = r["variants"][1 - t]
            alt = v[rng.integers(len(v))]
            switch = rng.random() < 0.75
            cells = z.copy()
            if varied and switch:
                cells[1 - t] = alt
            bucket = max(len(z[1 - t]), len(alt)) // 16
            groups[t, bucket, len(z[t]) // 12].append(len(rows))
            rows.append(dict(cells=cells))
        for z in addition:
            groups[t, len(z[1 - t]) // 16, len(z[t]) // 12].append(len(rows))
            rows.append(dict(cells=z))
    batches = []
    for (t, _, _), ids in groups.items():
        vs = rng.permutation(ids)
        batches.extend((t, vs[j : j + 96]) for j in range(0, len(vs), 96))
    return rows, [batches[i] for i in rng.permutation(len(batches))]


def main():
    configure_cuda_cli()
    sources = [
        "scripts/train_translation_paraphrases.py",
        "scripts/prepare_translation_paraphrases.py",
        "references/translation_paraphrase_protocol.md",
        "references/evidence/translation_paraphrases_v1/corpus.json",
        "models/bilingual_ink/likelihood/weights.safetensors",
        "scripts/train_ordered_translation.py",
        "references/evidence/proposition_translation_v1/design.json",
        "references/evidence/bilingual_ink_v1/corpus.json",
    ]
    bound = {s: sha(ROOT / s) for s in sources}
    for base, files in (
        (WORK, {k + ".json": v for k, v in read(PUB / "corpus.json")["files"].items()}),
        (LARGE, read(ROOT / sources[-1])["files"]),
        (FIRST, {"replay.json": read(ROOT / sources[-2])["files"]["replay"]}),
    ):
        for k, d in files.items():
            if sha(base / k) != d:
                raise ValueError(k)
    examples = read(WORK / "examples.json")
    replay = read(FIRST / "replay.json")
    imgs = np.load(LARGE / "images.npy", mmap_mode="r")
    ln = np.load(LARGE / "lengths.npy", mmap_mode="r")
    with gpu_owner():
        torch.set_num_threads(2)
        f = torch.tensor(
            observations({"tiles": np.load(BASE / "palette.npy")}, "ink"), device="cuda"
        )
        inv = [torch.tensor(v, device="cuda") for v in read(BASE / "inventories.json")]
        loc = [torch.full((len(f),), -100, device="cuda") for _ in inv]
        for v, l in zip(inv, loc):
            l[v] = torch.arange(len(v), device="cuda")
        for mode in ("canonical", "varied"):
            folder = MODEL / mode
            folder.mkdir(parents=True, exist_ok=True)
            rc = folder / "fit.json"
            weights = folder / "weights.safetensors"
            if rc.exists():
                if read(rc)["binding"] != bound or sha(weights) != read(rc)["weights"]:
                    raise ValueError(mode)
                continue
            net = model("likelihood", "cuda")
            opt = torch.optim.AdamW(net.parameters(), fused=True)
            hist = []
            ck = WORK / (mode + "-resume.pt")
            if ck.exists():
                st = torch.load(ck, weights_only=True, map_location="cpu")
                if st["binding"] != bound:
                    raise ValueError("Resume differs")
                net.load_state_dict(st["writer"])
                opt.load_state_dict(st["optimizer"])
                hist = st["history"]
            for ep in range(len(hist), 4):
                start = monotonic()
                meter = snapshot()
                torch.cuda.reset_peak_memory_stats()
                rows, batches = schedule(
                    examples, replay, imgs, ln, ep, mode == "varied"
                )
                stats = run(net, f, rows, batches, inv, loc, opt, 8 + ep)
                torch.cuda.synchronize()
                check_budget(start, sum(x["seconds"] for x in hist))
                hist.append(
                    dict(
                        epoch=ep + 1,
                        train=stats,
                        seconds=monotonic() - start,
                        joules=consumed(meter, snapshot())["joules"],
                        peak=torch.cuda.max_memory_allocated(),
                    )
                )
                atomic_save(
                    lambda p: torch.save(
                        dict(
                            binding=bound,
                            writer=net.state_dict(),
                            optimizer=opt.state_dict(),
                            history=hist,
                        ),
                        p,
                    ),
                    ck,
                )
                print(mode, hist[-1], flush=True)
            atomic_save(
                lambda p: save_file(
                    {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
                ),
                weights,
            )
            dump(
                rc,
                dict(
                    binding=bound,
                    weights=sha(weights),
                    parameters=sum(p.numel() for p in net.parameters()),
                    epochs=hist,
                ),
            )


if __name__ == "__main__":
    main()
