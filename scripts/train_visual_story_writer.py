"""Fit matched strip and whole-page glyph writers on one GPU, with epoch-level restart."""

from collections import Counter, defaultdict
from hashlib import sha256
from time import monotonic
import math
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from ilm.visual_lm.visual_story_writer import VisualStoryWriter
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_story_writer import (
    WORK,
    PUB,
    MODEL,
    ROOT,
    OUT,
    training,
    read,
    dump,
    sha,
)
from scripts.write_directional_reading import DirectionalAnswer
from scripts.train_fact_channel import pack, join_regions, gpu_owner
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli

SEED = 917231
MODES = ("strip", "page")


def source(mode, fact, regions):
    return fact if mode == "strip" else join_regions([fact, *regions])


def binding():
    src = [
        "scripts/train_visual_story_writer.py",
        "scripts/prepare_visual_story_writer.py",
        "ilm/visual_lm/visual_story_writer.py",
        "ilm/visual_lm/region_writer.py",
        "references/visual_story_writer_protocol.md",
        "references/evidence/visual_story_writer_v1/reservation.json",
        "references/evidence/story_answer_fit_v1/reservation.json",
        "data/chinese_origin_library_v1/manifest.json",
        "models/directional_reading/signed/weights.safetensors",
        "models/fact_span_ranking/whole/weights.safetensors",
        "models/fact_channel/human/weights.safetensors",
    ]
    return {s: sha(ROOT / s) for s in src}


def restore(mode, device):
    folder = MODEL / mode
    rc = read(folder / "fit.json")
    weights = folder / "weights.safetensors"
    if rc["mode"] != mode or sha(weights) != rc["sha256"]:
        raise ValueError("Writer weights changed")
    for name, h in rc["sources"].items():
        if name.startswith("models/") and sha(ROOT / name) != h:
            raise ValueError(name)
    model = VisualStoryWriter()
    model.load_state_dict(load_file(str(weights)))
    return model.to(device).eval()


@torch.inference_mode()
def select(net, features, rows):
    facts = [None] * len(rows)
    ids = sorted(range(len(rows)), key=lambda i: sum(map(len, rows[i]["p"])))
    for j in range(0, len(ids), 20):
        batch = ids[j : j + 20]
        choices = net.select(
            features, [rows[i]["q"] for i in batch], [rows[i]["p"] for i in batch]
        )
        for i, (_, _, v) in zip(batch, choices, strict=True):
            facts[i] = v
    return facts


def fit(mode, net, features, rows, facts):
    folder = MODEL / mode
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "weights.safetensors"
    final = folder / "fit.json"
    resume = WORK / (mode + ".pt")
    bound = binding()
    if final.exists():
        if read(final)["sources"] != bound:
            raise ValueError("Completed trial changed")
        restore(mode, "cpu")
        return
    torch.manual_seed(SEED)
    writer = VisualStoryWriter()
    writer.gru.recurrent.load_state_dict(net.writer.gru.state_dict())
    prior = {
        k: v for k, v in net.writer.state_dict().items() if not k.startswith("gru.")
    }
    missing, unexpected = writer.load_state_dict(prior, strict=False)
    assert not unexpected and all(k.startswith("gru.") for k in missing)
    writer = writer.cuda().train()
    optimizer = torch.optim.AdamW(writer.parameters(), lr=0.00035, fused=True)
    bins = defaultdict(list)
    for i, r in enumerate(rows):
        for y in dict.fromkeys(map(tuple, r["y"])):
            bins[(sum(map(len, r["p"])) // 230, len(y) // 32)].append((i, y))

    def schedule(epoch):
        rng = np.random.default_rng(SEED + epoch)
        batches = []
        for (p, t), items in bins.items():
            size = 24 if t < 3 and p < 4 else 12
            order = rng.permutation(len(items))
            batches.extend(
                [items[k] for k in order[j : j + size]]
                for j in range(0, len(items), size)
            )
        return [batches[k] for k in rng.permutation(len(batches))]

    hist = []
    if resume.exists():
        rc = torch.load(resume, weights_only=True, map_location="cpu")
        if rc["sources"] != bound:
            raise ValueError("Resume binding changed")
        hist = rc["history"]
        writer.load_state_dict(rc["net"])
        optimizer.load_state_dict(rc["optimizer"])
    for epoch in range(len(hist), 4):
        t = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        stats = Counter()
        unavailable = Counter()
        total = 0.0
        digest = sha256()
        batches = schedule(epoch)
        for j, batch in enumerate(batches):
            ids = [i for i, y in batch]
            digest.update(np.array(ids, dtype="<i4").tobytes())
            q = pack([rows[i]["q"] for i in ids], "cuda")
            p = pack([source(mode, facts[i], rows[i]["p"]) for i in ids], "cuda")
            y = pack([(*a, 1) for _, a in batch], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx, h = writer.encode(features, net.n, q, p)
                losses, unsupported = writer.objective(ctx, h, y)
                loss = losses.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                writer.parameters(), 1, error_if_nonfinite=True
            )
            frac = (epoch + j / len(batches)) / 4
            optimizer.param_groups[0]["lr"] = 0.00035 * (
                0.15 + 0.85 * (1 + math.cos(math.pi * frac)) / 2
            )
            optimizer.step()
            for i, n in zip(ids, unsupported.tolist(), strict=True):
                stats[rows[i]["task"]] += 1
                unavailable[rows[i]["task"]] += n
            total += float(loss.detach()) * len(ids)
            check_budget(t, sum(v["seconds"] for v in hist))
        torch.cuda.synchronize()
        hist.append(
            dict(
                epoch=epoch + 1,
                loss=total / sum(stats.values()),
                examples=stats,
                unreachable=unavailable,
                updates=len(batches),
                batch_order=digest.hexdigest(),
                seconds=monotonic() - t,
                joules=consumed(meter, snapshot())["joules"],
                peak=torch.cuda.max_memory_allocated(),
            )
        )
        # Keep primitive types in resumable checkpoints.
        hist[-1].update(examples=dict(stats), unreachable=dict(unavailable))
        atomic_save(
            lambda p: torch.save(
                dict(
                    sources=bound,
                    net=writer.state_dict(),
                    optimizer=optimizer.state_dict(),
                    history=hist,
                ),
                p,
            ),
            resume,
        )
        print(mode, hist[-1], flush=True)
    atomic_save(
        lambda p: save_file(
            {k: v.cpu().contiguous() for k, v in writer.state_dict().items()}, p
        ),
        weights,
    )
    dump(
        final,
        dict(
            mode=mode,
            sha256=sha(weights),
            sources=bound,
            epochs=hist,
            writer_parameters=sum(p.numel() for p in writer.parameters()),
            complete_parameters=sum(
                p.numel()
                for m in (writer, net.reader, net.ranker)
                for p in m.parameters()
            ),
        ),
    )


def main():
    configure_cuda_cli()
    with gpu_owner():
        if (
            sha(OUT / "origin.sqlite3")
            != read(OUT / "manifest.json")["database_sha256"]
        ):
            raise ValueError("Corpus checksum failed")
        bound = binding()
        ink, rows = training()
        net = DirectionalAnswer("signed", "cuda")
        f = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
        start = monotonic()
        meter = snapshot()
        facts = select(net, f, rows)
        receipt = dict(
            sources=bound,
            rows=dict(Counter(r["task"] for r in rows)),
            seconds=monotonic() - start,
            joules=consumed(meter, snapshot())["joules"],
            selected_sha256=sha256(
                b"".join(np.array([len(v), *v], dtype="<i4").tobytes() for v in facts)
            ).hexdigest(),
        )
        old = PUB / "selections.json"
        if old.exists():
            if any(
                receipt[k] != read(old)[k]
                for k in ("sources", "selected_sha256", "rows")
            ):
                raise ValueError("Selections changed")
        else:
            dump(old, receipt)
        for mode in MODES:
            fit(mode, net, f, rows, facts)


if __name__ == "__main__":
    main()
