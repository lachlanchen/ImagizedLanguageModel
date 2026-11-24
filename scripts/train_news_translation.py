"""Matched continued optical translation on conversational replay or new article groups."""

from collections import Counter, defaultdict
import gzip
from functools import lru_cache
from urllib.request import urlopen
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_ordered_translation import (
    ROOT,
    MODEL,
    PUB,
    WORK,
    read,
    dump,
    sha,
    key,
    hashed,
    Ink,
)
from scripts.write_ordered_translation import restore
from scripts.train_ordered_translation import batches, run
from scripts.train_fact_ranking import check_budget, atomic_save
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.measure_writer_energy import snapshot, consumed

ARCHIVE = ROOT / "data/raw/news_commentary/v18.en-zh.tsv.gz"
URL = "https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.en-zh.tsv.gz"
DIGEST = "47288c33b6b3f6c2494da0bda3c66e5ff2ba5891a1b2334a309ec4275255543b"


def prepare():
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=90) as r:
            payload = r.read()
        from hashlib import sha256

        if sha256(payload).hexdigest() != DIGEST:
            raise ValueError("New archive revision")
        ARCHIVE.write_bytes(payload)
    if sha(ARCHIVE) != DIGEST:
        raise ValueError("Archive hash differs")
    inks = Ink(np.load(MODEL / "palette.npy"))
    shape = lru_cache(None)(lambda c: inks.encode(c)[0])
    inv = [set(a) for a in read(MODEL / "inventories.json")]
    seen = [set(), set()]
    for k in ("train", "test", "dev"):
        for row in read(WORK / (k + ".json")):
            for j, s in enumerate(row["text"]):
                seen[j].add(key(s))
    docs = defaultdict(list)
    counts = Counter()
    title = None
    with gzip.open(ARCHIVE, "rt") as f:
        for line in f:
            sides = line.rstrip("\n").split("\t")
            if len(sides) != 2 or not all(s.strip() for s in sides):
                title = None
                continue
            if title is None:
                title = hashed("news-article-v1/" + key(sides[0]))
            if max(map(len, sides)) > 96:
                counts["over_length"] += 1
                continue
            norm = list(map(key, sides))
            if not all(norm) or any(v in seen[j] for j, v in enumerate(norm)):
                counts["empty_or_duplicate"] += 1
                continue
            cells = [list(map(shape, s)) for s in sides]
            if any(c in inks.bad for s in sides for c in s) or any(
                not set(z) <= inv[j] for j, z in enumerate(cells)
            ):
                counts["outside_palette"] += 1
                continue
            for j, v in enumerate(norm):
                seen[j].add(v)
            docs[title].append(dict(family=title, text=sides, cells=cells))
    ordered = sorted(docs)
    held = [r for title in ordered[:128] for r in docs[title]]
    pool = [r for title in ordered[128:] for r in docs[title]]
    for k, rows in (("train", pool), ("held", held)):
        dump(WORK / ("news-" + k + ".json"), rows)
    dump(
        PUB / "news-corpus.json",
        dict(
            archive=DIGEST,
            excluded=counts,
            train=len(pool),
            held=len(held),
            train_articles=len(ordered) - 128,
            held_articles=128,
            files={k: sha(WORK / ("news-" + k + ".json")) for k in ("train", "held")},
        ),
    )


def main():
    configure_cuda_cli()
    torch.set_num_threads(2)
    if not (PUB / "news-corpus.json").exists():
        prepare()
    news = read(WORK / "news-train.json")
    old = read(WORK / "train.json")
    binds = {
        p: sha(ROOT / p)
        for p in (
            "references/news_visual_translation_protocol.md",
            "scripts/train_news_translation.py",
            "models/ordered_translation/1/weights.safetensors",
            "references/evidence/ordered_translation_v1/news-corpus.json",
            "scripts/train_ordered_translation.py",
        )
    }
    if (
        sha(WORK / "news-train.json")
        != read(PUB / "news-corpus.json")["files"]["train"]
        or sha(WORK / "train.json") != read(PUB / "corpus.json")["files"]["train.json"]
    ):
        raise ValueError("Training rows changed")
    with gpu_owner():
        torch.set_num_threads(2)
        f = torch.tensor(
            observations({"tiles": np.load(MODEL / "palette.npy")}, "ink"),
            device="cuda",
        )
        inv = [torch.tensor(v, device="cuda") for v in read(MODEL / "inventories.json")]
        maps = [torch.full((len(f),), -100, device="cuda") for _ in inv]
        for m, v in zip(maps, inv):
            m[v] = torch.arange(len(v), device="cuda")
        for mode in ("replay", "news"):
            folder = MODEL / mode
            folder.mkdir(exist_ok=True)
            receipt = folder / "fit.json"
            if receipt.exists():
                if (
                    read(receipt)["binding"] != binds
                    or sha(folder / "weights.safetensors") != read(receipt)["weights"]
                ):
                    raise ValueError("Released fit changed")
                continue
            net = restore(1, "cuda")
            opt = torch.optim.AdamW(net.parameters(), fused=True)
            history = []
            ck = WORK / (mode + "-resume.pt")
            if ck.exists():
                resume = torch.load(ck, map_location="cpu", weights_only=True)
                if resume["bindings"] != binds:
                    raise ValueError("Resume binding differs")
                net.load_state_dict(resume["net"])
                opt.load_state_dict(resume["opt"])
                history = resume["history"]
            for ep in range(len(history), 4):
                rng = np.random.default_rng(170195 + ep)
                rows = [old[i] for i in rng.choice(len(old), 25000, replace=False)]
                pool = news if mode == "news" else old
                rows.extend(
                    pool[i]
                    for i in rng.choice(len(pool), min(len(news), 25000), replace=False)
                )
                start = monotonic()
                meter = snapshot()
                torch.cuda.reset_peak_memory_stats()
                meters = run(
                    net, f, rows, batches(rows, 27 + ep), inv, maps, opt, 7 + ep
                )
                torch.cuda.synchronize()
                check_budget(start, sum(r["seconds"] for r in history))
                history.append(
                    dict(
                        epoch=ep + 1,
                        train=meters,
                        seconds=monotonic() - start,
                        joules=consumed(meter, snapshot())["joules"],
                        peak=torch.cuda.max_memory_allocated(),
                    )
                )
                atomic_save(
                    lambda p: torch.save(
                        dict(
                            bindings=binds,
                            net=net.state_dict(),
                            opt=opt.state_dict(),
                            history=history,
                        ),
                        p,
                    ),
                    ck,
                )
                print(mode, ep + 1, meters["mean_nll"], flush=True)
            weights = folder / "weights.safetensors"
            atomic_save(
                lambda p: save_file(
                    {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
                ),
                weights,
            )
            dump(
                receipt,
                dict(
                    weights=sha(weights),
                    binding=binds,
                    epochs=history,
                    parameters=sum(p.numel() for p in net.parameters()),
                ),
            )


if __name__ == "__main__":
    main()
