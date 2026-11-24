"""Train exact visual copying inside natural Mandarin/English translations."""

import re
import sqlite3
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.catalogue_ink import print_cell
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.specimen_translation import SpecimenTranslator
from scripts.prepare_ordered_translation import ROOT, WORK, MODEL, read, dump, sha
from scripts.train_ordered_translation import pack, batches
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


def names(row):
    en, zh = row["text"]
    for name, zh_names in (("Tom", ("汤姆", "湯姆")), ("Mary", ("玛丽", "瑪麗"))):
        matches = list(re.finditer(r"\b" + name + r"\b", en))
        z = [s for s in zh_names if zh.count(s) == 1]
        if len(matches) == len(z) == 1:
            a = matches[0].span()
            b = zh.index(z[0])
            return a, (b, b + len(z[0]))
    return None


def materials():
    pixels = list(map(bytes, np.load(WORK / "inputs.npy")))
    seen = set(pixels)
    db = sqlite3.connect(
        (ROOT / "data/chinese_origin_library_v1/origin.sqlite3").resolve().as_uri()
        + "?mode=ro",
        uri=True,
    )
    choices = {}
    labels = {}
    for split, upper in (("train", 1024), ("sealed", 256)):
        choices[split] = []
        records = db.execute(
            """SELECT svg_sha,pixels32,label FROM form JOIN image USING(svg_sha)
        JOIN entry USING(headword,entry_id) WHERE split=? AND has_ink=1 AND stage='oracle' ORDER BY svg_sha,label""",
            (split,),
        )
        for h, p, label in records:
            raw = print_cell(p)
            if raw in seen:
                continue
            seen.add(raw)
            choices[split].append(len(pixels))
            labels[len(pixels)] = label
            pixels.append(raw)
            if len(choices[split]) == upper:
                break
        if len(choices[split]) != upper:
            raise ValueError("Insufficient distinct glyphs")
    db.close()
    return np.frombuffer(b"".join(pixels), np.uint8).reshape(-1, 128), choices, labels


def changed(r, extent, mark):
    cells = [
        z[:a] + [mark] + z[b:] for z, (a, b) in zip(r["cells"], extent, strict=True)
    ]
    return dict(r, cells=cells, mark=mark)


def main():
    configure_cuda_cli()
    folder = MODEL / "specimen"
    folder.mkdir(exist_ok=True)
    final = folder / "fit.json"
    weight = folder / "weights.safetensors"
    src = [
        "scripts/train_specimen_translation.py",
        "ilm/visual_lm/specimen_translation.py",
        "references/open_specimen_translation_protocol.md",
        "models/ordered_translation/4/weights.safetensors",
        "data/chinese_origin_library_v1/origin.sqlite3",
        "references/evidence/ordered_translation_v1/corpus.json",
    ]
    bound = {p: sha(ROOT / p) for p in src}
    if final.exists():
        if read(final)["binding"] != bound or read(final)["weights"] != sha(weight):
            raise ValueError("Binding changed")
        return
    with gpu_owner():
        torch.set_num_threads(2)
        torch.manual_seed(31067)
        tiles, ids, _ = materials()
        f = torch.tensor(observations({"tiles": tiles}, "ink"), device="cuda")
        inv = [torch.tensor(v, device="cuda") for v in read(MODEL / "inventories.json")]
        local = [torch.full((len(f),), -1, device="cuda") for _ in inv]
        for m, v in zip(local, inv):
            m[v] = torch.arange(len(v), device="cuda")
        net = SpecimenTranslator().cuda()
        missing, extra = net.load_state_dict(
            load_file(str(MODEL / "4/weights.safetensors")), strict=False
        )
        if extra or any(
            k.split(".")[0] not in ("address", "request", "route") for k in missing
        ):
            raise ValueError("Parent disagrees")
        opt = torch.optim.AdamW(net.parameters(), lr=0.00022, fused=True)
        train = read(WORK / "train.json")
        eligible = [(i, v) for i, r in enumerate(train) if (v := names(r))]
        hist = []
        resume = WORK / "specimen-resume.pt"
        if resume.exists():
            state = torch.load(resume, weights_only=True, map_location="cpu")
            if state["bound"] != bound:
                raise ValueError("Checkpoint differs")
            hist = state["history"]
            net.load_state_dict(state["net"])
            opt.load_state_dict(state["optimizer"])
        for epoch in range(len(hist), 4):
            rng = np.random.default_rng(31107 + epoch)
            # Every changed example has a human-aligned pair; replay is freshly sampled each pass.
            rows = [
                train[i]
                for i in rng.choice(len(train), size=2 * len(eligible), replace=False)
            ]
            rows += [
                changed(train[i], v, int(rng.choice(ids["train"]))) for i, v in eligible
            ]
            start = monotonic()
            meter = snapshot()
            torch.cuda.reset_peak_memory_stats()
            loss_total = 0
            targets = 0
            updates = 0
            for t, ix in batches(rows, epoch):
                p = pack([rows[i]["cells"][1 - t] for i in ix], "cuda")
                y = pack([rows[i]["cells"][t] + [1] for i in ix], "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net.loss(f, p, y, inv[t], local[t], t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 1, error_if_nonfinite=True
                )
                opt.step()
                n = int((y != 0).sum())
                loss_total += float(loss.detach()) * n
                targets += n
                updates += 1
            torch.cuda.synchronize()
            check_budget(start, sum(v["seconds"] for v in hist))
            hist.append(
                dict(
                    epoch=epoch + 1,
                    targets=targets,
                    updates=updates,
                    directed_replacements=2 * len(eligible),
                    directed_replay=4 * len(eligible),
                    nll=loss_total / targets,
                    seconds=monotonic() - start,
                    joules=consumed(meter, snapshot())["joules"],
                    peak=torch.cuda.max_memory_allocated(),
                )
            )
            atomic_save(
                lambda p: torch.save(
                    dict(
                        bound=bound,
                        net=net.state_dict(),
                        optimizer=opt.state_dict(),
                        history=hist,
                    ),
                    p,
                ),
                resume,
            )
            print(hist[-1], flush=True)
        atomic_save(
            lambda p: save_file(
                {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
            ),
            weight,
        )
        dump(
            final,
            dict(
                epochs=hist,
                weights=sha(weight),
                binding=bound,
                parameters=sum(p.numel() for p in net.parameters()),
            ),
        )


if __name__ == "__main__":
    main()
