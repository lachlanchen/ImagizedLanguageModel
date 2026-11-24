"""Fit matched full-answer writers with a fixed visual evidence reader on one GPU."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import math
from pathlib import Path
import shutil
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.sentence_context import ContextWriter, scoring_positions
from scripts.prepare_sentence_context import (
    ROOT,
    WORK,
    PUB,
    MODEL,
    BANK,
    PREVIOUS,
    PREVIOUS_PUB,
    Ink,
    read,
    dump,
    sha,
    prepared_files,
)
from scripts.train_fact_channel import pack, join_regions, SEED, load, gpu_owner
from scripts.write_fact_ranking import RankedAnswer
from scripts.train_fact_ranking import MODEL as RANK
from scripts.prepare_fact_channel import MODEL as BASE
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli

ARMS = ("direct", "surround")
CACHE = WORK / "optical"
CACHE_FILES = ("rows.json", "seq.npz", "tiles.npy", "readback.json", "choices.npy")


def assemble(stage):
    spec = read(PUB / "corpus.json")
    base = read(PREVIOUS_PUB / "corpus.json")
    for name, h in spec["files"].items():
        if sha(WORK / name) != h:
            raise ValueError(name)
    for name in ("writer.json", "tiles.npy", "strings.npz", "readback.json"):
        if sha(PREVIOUS / name) != base["files"][name]:
            raise ValueError(name)
    if sha(BANK) != spec["palette"]:
        raise ValueError("Changed training inventory")
    palette = np.load(BANK)
    ink = Ink(palette)
    old = np.load(PREVIOUS / "tiles.npy")
    remap = [0, 1]
    for raw in map(bytes, old[2:]):
        if raw not in ink.byink:
            ink.byink[raw] = len(ink.tiles)
            ink.tiles.append(raw)
        remap.append(ink.byink[raw])
    names = read(PREVIOUS / "readback.json")
    chars = ["�"] * len(palette)
    for i, c in enumerate(names):
        if remap[i] < len(chars):
            chars[remap[i]] = c
    with np.load(PREVIOUS / "strings.npz") as z:
        seq = np.array(remap, np.int32)[z["ids"]]
        ptr = z["ptr"]
    strings = [seq[a:b] for a, b in zip(ptr[:-1], ptr[1:])]
    rows = [[q, bs, ys, f, t] for q, bs, f, ys, t, _ in read(PREVIOUS / "writer.json")]
    stats = defaultdict(Counter)
    extra_ids = {}

    def number(s):
        if s not in extra_ids:
            extra_ids[s] = len(strings)
            strings.append(np.array(ink.encode(s), np.int32))
        return extra_ids[s]

    for r in read(WORK / "additional.json"):
        t = r["task"]
        q = number(r["q"])
        bs = [number(b) for b in r["blocks"]]
        visible = set(strings[q])
        for b in bs:
            visible.update(strings[b])
        ys = []
        for s in r["answers"]:
            y = number(s)
            if ink.bad.intersection(r["q"] + "".join(r["blocks"]) + s):
                stats[t]["font_excluded"] += 1
            elif any(v >= len(palette) and v not in visible for v in strings[y]):
                stats[t]["output_support_excluded"] += 1
            else:
                ys.append(y)
                stats[t]["wordings"] += 1
        if ys:
            rows.append([q, bs, ys, -1, t])
    for c, i in ink.chars.items():
        if i < len(chars) and chars[i] == "�":
            chars[i] = c
    tiles = ink.array()
    np.save(stage / "tiles.npy", tiles)
    np.savez(
        stage / "seq.npz",
        ids=np.concatenate(strings),
        ptr=np.cumsum([0] + list(map(len, strings))),
    )
    dump(stage / "rows.json", rows)
    dump(stage / "readback.json", chars)
    reader = RankedAnswer("whole", "cuda")
    feat = torch.from_numpy(observations({"tiles": tiles}, "ink")).to("cuda")
    chosen = np.zeros((len(rows), 3), np.int32)
    order = sorted(
        range(len(rows)), key=lambda i: sum(len(strings[b]) for b in rows[i][1])
    )
    clock = monotonic()
    meter = snapshot()
    with torch.inference_mode():
        for j in range(0, len(order), 32):
            indices = order[j : j + 32]
            spans = reader.select(
                feat,
                [strings[rows[i][0]] for i in indices],
                [[strings[b] for b in rows[i][1]] for i in indices],
            )
            chosen[indices] = [(b, s, len(f)) for b, s, f in spans]
    torch.cuda.synchronize()
    np.save(stage / "choices.npy", chosen)
    print(
        "training sources selected",
        len(rows),
        round(monotonic() - clock, 1),
        "seconds",
        consumed(meter, snapshot()),
        flush=True,
    )
    return dict(
        files={f: sha(stage / f) for f in CACHE_FILES},
        extra=stats,
        questions=len(rows),
        wordings=sum(len(r[2]) for r in rows),
        corpus=sha(PUB / "corpus.json"),
        palette=sha(BANK),
        trained_reader=sha(BASE / "reader/weights.safetensors"),
        ranker=sha(RANK / "whole/weights.safetensors"),
    )


def extended_page(regions, span):
    return join_regions([span, *regions])


def setup():
    prepared_files(
        CACHE, PUB / "optical.json", {f: ("files", f) for f in CACHE_FILES}, assemble
    )
    model_bank = MODEL / "palette.npy"
    MODEL.mkdir(exist_ok=True, parents=True)
    for f, src in (
        (model_bank, BANK),
        (MODEL / "readback.json", CACHE / "readback.json"),
    ):
        if f.exists() and sha(f) != sha(src):
            raise ValueError(str(f))
        if not f.exists():
            shutil.copyfile(src, f)


class Training:
    def __init__(self):
        rows = read(CACHE / "rows.json")
        picks = np.load(CACHE / "choices.npy")
        tiles = np.load(CACHE / "tiles.npy")
        self.bank = torch.from_numpy(observations({"tiles": tiles}, "ink")).to("cuda")
        self.n = len(np.load(MODEL / "palette.npy"))
        with np.load(CACHE / "seq.npz") as z:
            ids, ptr = z["ids"], z["ptr"]
        strings = [ids[a:b] for a, b in zip(ptr[:-1], ptr[1:])]
        self.data = []
        self.inputs = []
        self.counts = Counter()
        bins = defaultdict(list)
        for (q, bs, ys, f, t), (b, s, l) in zip(rows, picks, strict=True):
            blocks = [strings[k] for k in bs]
            span = blocks[b][s : s + l]
            if len(span) != l or not l:
                raise ValueError("Invalid selected fact")
            ix = len(self.inputs)
            source = extended_page(blocks, span)
            self.inputs.append((strings[q], source))
            for y in ys:
                target = np.r_[strings[y], np.int32(1)]
                aux = None
                if f >= 0 and len(strings[f]) < len(target) - 1:
                    # Binary serialization uses a multiple-of-four check to keep glyph boundaries.
                    src = target[:-1].tobytes()
                    needle = strings[f].tobytes()
                    hits = [
                        k
                        for k in range(len(target) - len(strings[f]))
                        if target[k] == strings[f][0] and src.startswith(needle, 4 * k)
                    ]
                    if len(hits) == 1:
                        aux = scoring_positions(target, hits[0], len(strings[f]))
                token = len(self.data)
                self.data.append((ix, target, aux))
                bins[(len(source) // 128, len(target) // 32)].append(token)
                self.counts["positions"] += len(target)
                if aux:
                    self.counts["auxiliary_answers"] += 1
                    self.counts["auxiliary_positions"] += len(aux[1])
        self.batches = []
        rng = np.random.default_rng(SEED)
        for epoch in range(2):
            epoch_batches = []
            for (plen, ylen), indices in bins.items():
                # Limit physical logits and source alignment work for long passages.
                size = max(
                    1, min(48, 4096 // ((ylen + 1) * 32), 28000 // ((plen + 1) * 128))
                )
                perm = rng.permutation(indices)
                epoch_batches.extend(
                    perm[j : j + size] for j in range(0, len(perm), size)
                )
            self.batches.extend(
                epoch_batches[k] for k in rng.permutation(len(epoch_batches))
            )

    def loss(self, net, mode, indices):
        rs = [self.data[i] for i in indices]
        q, p = zip(*(self.inputs[i] for i, _, _ in rs), strict=True)
        y = pack([v for _, v, _ in rs], "cuda")
        c, h = net.encode(self.bank, self.n, pack(q, "cuda"), pack(p, "cuda"))
        aux = [(j, a) for j, (_, _, a) in enumerate(rs) if a is not None]
        js = torch.tensor([j for j, _ in aux], dtype=torch.long, device="cuda")
        if not aux:
            return net.objective(mode, c, h, y, js, None, None)
        compact = pack([a[0] for _, a in aux], "cuda")
        mask = np.zeros((len(aux), y.shape[1]), bool)
        for i, (_, a) in enumerate(aux):
            mask[i, a[1]] = True
        return net.objective(
            mode, c, h, y, js, compact, torch.from_numpy(mask).to("cuda")
        )


def binding():
    files = [
        Path(__file__),
        ROOT / "ilm/visual_lm/sentence_context.py",
        ROOT / "ilm/visual_lm/region_writer.py",
        ROOT / "formal/span_list/SentenceContext.lean",
        PUB / "optical.json",
        PUB / "corpus.json",
        BASE / "human/weights.safetensors",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def fit(data, mode):
    out = MODEL / mode
    out.mkdir(exist_ok=True, parents=True)
    frozen = binding()
    weights = out / "weights.safetensors"
    if (out / "fit.json").exists():
        rc = read(out / "fit.json")
        if rc["binding"] != frozen or rc["weights"] != sha(weights):
            raise ValueError("Changed release")
        return
    torch.manual_seed(SEED)
    net = ContextWriter().cuda()
    net.load_state_dict(load("human", "cpu").state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.01, fused=True)
    checkpoint = WORK / (mode + "-resume.pt")
    first = 0
    duration = energy = peak = 0
    trace = []
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ck["binding"] != frozen:
            raise ValueError("Changed restart")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        first, duration, energy, peak, trace = [
            ck[k] for k in ("step", "seconds", "joules", "peak", "trace")
        ]
    meter = snapshot()
    clock = monotonic()
    dt = duration
    ej = energy
    torch.cuda.reset_peak_memory_stats()
    for j in range(first, len(data.batches)):
        step = j + 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = data.loss(net, mode, data.batches[j])
        opt.param_groups[0]["lr"] = (
            0.0003
            * min(1, step / 100)
            * (0.20 + 0.80 * (1 + math.cos(math.pi * step / len(data.batches))) / 2)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 250 == 0 or step == len(data.batches):
            trace.append([step, float(loss.detach())])
            print(mode, step, len(data.batches), trace[-1][1], flush=True)
        if step % 500 == 0 or step == len(data.batches):
            torch.cuda.synchronize()
            duration = dt + monotonic() - clock
            energy = ej + consumed(meter, snapshot())["joules"]
            peak = max(peak, torch.cuda.max_memory_allocated())
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(
                dict(
                    binding=frozen,
                    model=net.state_dict(),
                    opt=opt.state_dict(),
                    step=step,
                    seconds=duration,
                    joules=energy,
                    peak=peak,
                    trace=trace,
                ),
                tmp,
            )
            tmp.replace(checkpoint)
            if peak > 18 * 2**30 or duration > 21600:
                raise RuntimeError("Resource ceiling")
    tmp = weights.with_suffix(".tmp")
    save_file({k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(weights)
    order = sha256(
        b"".join(np.asarray(b, dtype="<i4").tobytes() for b in data.batches)
    ).hexdigest()
    dump(
        out / "fit.json",
        dict(
            binding=frozen,
            weights=sha(weights),
            steps=len(data.batches),
            wordings=len(data.data),
            epochs=2,
            per_epoch=data.counts,
            order=order,
            seconds=duration,
            joules=energy,
            peak_bytes=peak,
            trace=trace,
            parameters=sum(p.numel() for p in net.parameters()),
        ),
    )


if __name__ == "__main__":
    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms", nargs="+", choices=(*ARMS, "prepare"))
    args = p.parse_args()
    torch.set_num_threads(3)
    with gpu_owner():
        setup()
        if args.arms != ["prepare"]:
            data = Training()
            for mode in args.arms:
                fit(data, mode)
