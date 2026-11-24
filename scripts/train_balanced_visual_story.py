"""Test counterbalanced period requests against matched continued training."""

from collections import Counter
from hashlib import sha256
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.catalogue_ink import print_forms
from scripts.prepare_visual_story_writer import (
    ROOT,
    MODEL,
    STAGES,
    training,
    dump,
    read,
    sha,
)
from scripts.train_visual_story_writer import restore, select, source
from scripts.train_fact_channel import pack, gpu_owner
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.write_directional_reading import DirectionalAnswer
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.evaluate_visual_story_writer import forms, evaluate

WORK = ROOT / "artifacts/visual_form_balance_v1"
PUB = ROOT / "references/evidence/visual_form_balance_v1"
MODES = ("fixed", "balanced")


def condition(row, ink, epoch, mode):
    if mode == "fixed" or row["task"] != "forms":
        return row
    bits = sha256(row["family"].encode()).digest()[0]
    second = int(row["images"][1] == row["y"][0][0])
    english = second ^ ((epoch >> 1) & 1) ^ ((bits >> 1) & 1)
    reverse = (epoch & 1) ^ (bits & 1)
    c = row["char"]
    zh, en = STAGES[row["stage"]]
    q = f"Show the {en} form of {c}." if english else f"请写出{c}的{zh}字形。"
    head, first, last = row["p"]
    return dict(row, q=ink.encode(q), p=[head, last, first] if reverse else row["p"])


def fit(mode, ink, rows, reader):
    parent = read(MODEL / "page/fit.json")
    bound = dict(parent["sources"])
    for name in [
        "scripts/train_balanced_visual_story.py",
        "references/visual_form_balance_protocol.md",
        "ilm/visual_lm/catalogue_ink.py",
        "models/visual_story_writer/page/weights.safetensors",
    ]:
        bound[name] = sha(ROOT / name)
    folder = MODEL / mode
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists():
        if read(folder / "fit.json")["sources"] != bound:
            raise ValueError("Completed experiment binding changed")
        restore(mode, "cpu")
        return
    torch.manual_seed(7161)
    writer = restore("page", "cuda").train()
    opt = torch.optim.AdamW(writer.parameters(), lr=0.00018, fused=True)
    logs = []
    for epoch in range(4):
        t = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        presented = [condition(r, ink, epoch, mode) for r in rows]
        features = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
        facts = select(reader, features, presented)
        rng = np.random.default_rng(23071 + epoch)
        # Bucket before shuffling to keep wide modern pages out of short form batches.
        bins = {}
        for i, row in enumerate(rows):
            for y in dict.fromkeys(map(tuple, row["y"])):
                bins.setdefault(
                    (sum(map(len, row["p"])) // 200, len(y) // 32), []
                ).append((i, y))
        batches = []
        for items in bins.values():
            order = rng.permutation(len(items))
            batches.extend(
                [items[k] for k in order[j : j + 24]] for j in range(0, len(items), 24)
            )
        examples = Counter()
        total = 0
        digest = sha256()
        for j in rng.permutation(len(batches)):
            batch = batches[j]
            ids = [i for i, _ in batch]
            digest.update(np.asarray(ids, dtype="<i4").tobytes())
            q = pack([presented[i]["q"] for i in ids], "cuda")
            p = pack([source("page", facts[i], presented[i]["p"]) for i in ids], "cuda")
            y = pack([(*a, 1) for _, a in batch], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx, h = writer.encode(features, reader.n, q, p)
                losses, missing = writer.objective(ctx, h, y)
                loss = losses.mean()
            if missing.any():
                raise ValueError("A reply contains an unavailable target")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                writer.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            examples.update(rows[i]["task"] for i in ids)
            total += float(loss.detach()) * len(ids)
            check_budget(t, sum(v["seconds"] for v in logs))
        torch.cuda.synchronize()
        logs.append(
            dict(
                epoch=epoch + 1,
                examples=dict(examples),
                updates=len(batches),
                batch_order=digest.hexdigest(),
                loss=total / sum(examples.values()),
                seconds=monotonic() - t,
                joules=consumed(meter, snapshot())["joules"],
                peak=torch.cuda.max_memory_allocated(),
            )
        )
        print(mode, logs[-1], flush=True)
    weights = folder / "weights.safetensors"
    atomic_save(
        lambda path: save_file(
            {k: v.detach().cpu().contiguous() for k, v in writer.state_dict().items()},
            path,
        ),
        weights,
    )
    dump(
        folder / "fit.json",
        dict(
            mode=mode,
            sources=bound,
            epochs=logs,
            sha256=sha(weights),
            complete_parameters=parent["complete_parameters"],
            writer_parameters=parent["writer_parameters"],
        ),
    )


def main():
    configure_cuda_cli()
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    with gpu_owner():
        ink, pool = training()
        pool = print_forms(ink, pool)
        counts = Counter()
        rows = []
        for r in pool:
            t = r["task"]
            if t == "forms" or counts[t] < 512:
                counts[t] += 1
                rows.append(r)
        reader = DirectionalAnswer("signed", "cuda")
        for mode in MODES:
            fit(mode, ink, rows, reader)
        del reader
    forms(("page", *MODES), "sealed", WORK, PUB, printed=True)
    evaluate(None, MODES, WORK, PUB)


if __name__ == "__main__":
    main()
